"""Parquet Numpy embedding reader, read embeddings from numpy files in streaming, also read parquet files
Assumptions: the numpy and parquet files should be completely aligned
"""

import pandas as pd
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import numpy as np
import math
from collections import namedtuple
from embedding_reader.get_file_list import get_file_list
from embedding_reader.numpy_reader import get_numpy_headers
from embedding_reader.piece_builder import build_pieces, PIECES_BASE_COLUMNS
from threading import Semaphore
import pyarrow.parquet as pq


class ParquetNumpyReader:
    """Parquet numpy reader class, implements init to read the files headers and call to procuce embeddings batches"""

    def __init__(self, embeddings_folder, metadata_folder, metadata_column_names):
        self.embeddings_folder = embeddings_folder
        self.fs, embeddings_file_paths = get_file_list(embeddings_folder, "npy")
        self.metadata_folder = metadata_folder
        self.metadata_fs, metadata_file_paths = get_file_list(metadata_folder, "parquet")
        self.metadata_column_names = metadata_column_names

        headers = get_numpy_headers(embeddings_file_paths, self.fs)

        # add metadata path to headers by zipping
        headers = [[*h, m] for h, m in zip(headers, metadata_file_paths)]

        self.headers = pd.DataFrame(
            headers,
            columns=[
                "filename",
                "count",
                "count_before",
                "dimension",
                "dtype",
                "header_offset",
                "byte_per_item",
                "metadata_path",
            ],
        )

        self.count = self.headers["count"].sum()
        if self.count == 0:
            raise ValueError("No embeddings found in folder {embeddings_folder}")
        self.dimension = int(self.headers.iloc[0]["dimension"])
        self.byte_per_item = self.headers.iloc[0]["byte_per_item"]
        self.dtype = self.headers.iloc[0]["dtype"]
        self.total_size = self.count * self.byte_per_item

    def __call__(self, batch_size, start=0, end=None, max_piece_size=None, parallel_pieces=None, show_progress=True):
        if end is None:
            end = self.count

        if end > self.count:
            end = self.count
        if batch_size > end - start:
            batch_size = end - start

        if max_piece_size is None:
            max_piece_size = max(int(50 * 10**6 / (self.byte_per_item)), 1)
        if parallel_pieces is None:
            parallel_pieces = max(math.ceil(batch_size / max_piece_size), 10)

        metadata_columns = ["metadata_path", "header_offset"]
        pieces = build_pieces(
            headers=self.headers,
            batch_size=batch_size,
            start=start,
            end=end,
            max_piece_size=max_piece_size,
            metadata_columns=metadata_columns,
        )
        cols = PIECES_BASE_COLUMNS + metadata_columns
        Piece = namedtuple("Count", cols)

        def read_piece(t):
            (piece, table) = t
            try:
                start = piece.piece_start
                end = piece.piece_end
                path = piece.filename
                header_offset = piece.header_offset

                length = end - start
                id_columns = self.metadata_column_names
                table_slice = table.slice(start, length)
                ids = table_slice.select(id_columns).to_pandas()

                with self.fs.open(path, "rb") as f:
                    length = end - start
                    f.seek(header_offset + start * self.byte_per_item)
                    return (
                        None,
                        (
                            np.frombuffer(f.read(length * self.byte_per_item), dtype=self.dtype).reshape(
                                (length, self.dimension)
                            ),
                            ids,
                            piece,
                        ),
                    )
            except Exception as e:  # pylint: disable=broad-except
                return e, (None, None, piece)

        semaphore = Semaphore(parallel_pieces)

        stopped = False
        # from path to table and file
        open_parquet_files = {}

        def piece_generator(pieces, open_parquet_files):
            current_parquet_file = None
            for piece in (Piece(*parts) for parts in zip(*[pieces[col] for col in cols])):
                if stopped:
                    break
                semaphore.acquire()  # pylint: disable=consider-using-with
                if piece.metadata_path not in open_parquet_files:
                    file = self.metadata_fs.open(piece.metadata_path, "rb")
                    for i in range(5):
                        try:
                            table = pq.read_table(file, use_threads=True)
                            break
                        except Exception as e:  # pylint: disable=broad-except
                            print("Fail to read " + piece.metadata_path)
                            print(e)
                            if i != 4:
                                print(f"Retry number {i+1}/5...")
                            else:
                                raise e

                    open_parquet_files[piece.metadata_path] = {"file": file, "table": table}
                if current_parquet_file != piece.metadata_path:
                    current_parquet_file = piece.metadata_path
                yield (piece, open_parquet_files[piece.metadata_path]["table"])

        batch = None
        batch_meta = None
        batch_offset = 0

        if show_progress:
            pbar = tqdm(total=len(pieces))
        with ThreadPool(parallel_pieces) as p:
            current_parquet_file = None
            for err, (data, meta, piece) in p.imap(read_piece, piece_generator(pieces, open_parquet_files)):
                if err is not None:
                    stopped = True
                    semaphore.release()
                    raise Exception(
                        f"failed reading file {piece.filename} from {piece.piece_start} to {piece.piece_end}"
                    ) from err
                try:
                    if batch is None:
                        batch = np.empty((piece.batch_length, self.dimension), "float32")
                        batch_meta = np.empty((piece.batch_length, len(self.metadata_column_names)), dtype="object")

                    batch[batch_offset : (batch_offset + piece.piece_length)] = data
                    if self.metadata_column_names is not None:
                        batch_meta[batch_offset : (batch_offset + piece.piece_length)] = meta.to_numpy()
                    batch_offset += data.shape[0]
                    if piece.last_piece:
                        meta_batch_df = pd.DataFrame(batch_meta, columns=self.metadata_column_names).infer_objects()
                        meta_batch_df["i"] = np.arange(start=piece.batch_start, stop=piece.batch_end)
                        yield batch, meta_batch_df
                        batch = None
                        batch_meta = None
                        batch_offset = 0
                    if current_parquet_file != piece.metadata_path:
                        if current_parquet_file is not None:
                            open_parquet_files[current_parquet_file]["file"].close()
                            del open_parquet_files[current_parquet_file]
                        current_parquet_file = piece.metadata_path

                    if show_progress:
                        pbar.update(1)
                    semaphore.release()
                except Exception as e:  # pylint: disable=broad-except
                    stopped = True
                    semaphore.release()
                    raise e
            if current_parquet_file is not None:
                open_parquet_files[current_parquet_file]["file"].close()
                del open_parquet_files[current_parquet_file]

        if show_progress:
            pbar.close()

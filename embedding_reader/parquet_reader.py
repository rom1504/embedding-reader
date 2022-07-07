"""Parquet embedding reader, read embeddings from parquet files in streaming"""

import pandas as pd
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import numpy as np
import pyarrow.parquet as pq
from collections import namedtuple
from embedding_reader.get_file_list import get_file_list
from embedding_reader.piece_builder import build_pieces, PIECES_BASE_COLUMNS
from threading import Semaphore
import math


def file_to_header(filename, fs):
    try:
        with fs.open(filename, "rb") as f:
            parquet_file = pq.ParquetFile(f, memory_map=True)
            return (None, [filename, parquet_file.metadata.num_rows])
    except Exception as e:  # pylint: disable=broad-except
        return e, (filename, None)


def get_parquet_headers(fs, embeddings_file_paths):
    """get parquet headers"""
    headers = []
    count_before = 0
    with ThreadPool(10) as p:
        for err, c in tqdm(
            p.imap(lambda fp: file_to_header(fp, fs), embeddings_file_paths), total=len(embeddings_file_paths)
        ):
            if err is not None:
                raise Exception(f"failed reading file {c[0]}") from err
            if c[1] == 0:
                continue
            headers.append([*c, count_before])
            count_before += c[1]

    return headers


class ParquetReader:
    """Parquet reader class, implements init to read the files headers and call to produce embeddings batches"""

    def __init__(self, embeddings_folder, embedding_column_name, metadata_column_names=None):
        self.embeddings_folder = embeddings_folder
        self.fs, embeddings_file_paths = get_file_list(embeddings_folder, "parquet")

        self.metadata_column_names = metadata_column_names
        self.embedding_column_name = embedding_column_name

        # read one non empty file to get the dimension
        for filename in embeddings_file_paths:
            with self.fs.open(filename, "rb") as f:
                parquet_file = pq.ParquetFile(f, memory_map=True)
                batches = parquet_file.iter_batches(batch_size=1, columns=[embedding_column_name])
                try:
                    embedding = next(batches).to_pandas()[embedding_column_name].to_numpy()[0]
                    self.dimension = int(embedding.shape[0])
                    break
                except StopIteration:
                    continue

        headers = get_parquet_headers(self.fs, embeddings_file_paths)
        self.headers = pd.DataFrame(headers, columns=["filename", "count", "count_before"])
        self.count = self.headers["count"].sum()
        if self.count == 0:
            raise ValueError(f"No embeddings found in folder {embeddings_folder}")
        self.nb_files = len(self.headers["count"])
        self.byte_per_item = 4 * self.dimension
        self.total_size = self.count * self.byte_per_item
        self.max_file_size = max(self.headers["count"]) * self.byte_per_item

    def __call__(
        self,
        batch_size,
        start=0,
        end=None,
        max_piece_size=None,
        parallel_pieces=None,
        show_progress=True,
        max_ram_usage_in_bytes=2**31,
    ):
        if end is None:
            end = self.count

        if end > self.count:
            end = self.count
        if batch_size > end - start:
            batch_size = end - start

        if max_piece_size is None:
            # Take x embeddings per pieces so that the max piece size is max(50MB, size of a batch)
            max_piece_size = max(int(50 * 10**6 / self.byte_per_item), batch_size, 1)

        if parallel_pieces is None:
            # read_piece function can load the whole file in memory to take only max_piece_size bytes
            read_piece_ram_usage = self.max_file_size
            # We try to parallelize a maximum as long at it fits the ram constraint
            parallel_pieces = min(max(int(math.ceil(max_ram_usage_in_bytes / read_piece_ram_usage)), 1), 50)
        pieces = build_pieces(
            headers=self.headers, batch_size=batch_size, start=start, end=end, max_piece_size=max_piece_size
        )

        cols = PIECES_BASE_COLUMNS
        Piece = namedtuple("Count", cols)

        def read_piece(piece):
            try:
                start = piece.piece_start
                end = piece.piece_end
                path = piece.filename

                with self.fs.open(path, "rb") as f:
                    length = end - start
                    table = pq.read_table(f, use_threads=False)
                    id_columns = self.metadata_column_names
                    table_slice = table.slice(start, length)
                    embeddings_raw = table_slice[self.embedding_column_name].to_numpy()
                    ids = table_slice.select(id_columns).to_pandas() if id_columns else None

                    return (
                        None,
                        (
                            np.stack(embeddings_raw),
                            ids,
                            piece,
                        ),
                    )
            except Exception as e:  # pylint: disable=broad-except
                return e, (None, None, piece)

        semaphore = Semaphore(parallel_pieces)
        stopped = False

        def piece_generator(pieces):
            for piece in (Piece(*parts) for parts in zip(*[pieces[col] for col in cols])):
                if stopped:
                    break
                semaphore.acquire()  # pylint: disable=consider-using-with
                yield piece

        batch = None
        batch_meta = None
        batch_offset = 0

        if show_progress:
            pbar = tqdm(total=len(pieces))
        with ThreadPool(parallel_pieces) as p:
            for err, (data, meta, piece) in p.imap(read_piece, piece_generator(pieces)):
                if err is not None:
                    stopped = True
                    semaphore.release()
                    raise Exception(
                        f"failed reading file {piece.filename} from {piece.piece_start} to {piece.piece_end}"
                    ) from err
                try:
                    if batch is None:
                        batch = np.empty((piece.batch_length, self.dimension), "float32")
                        if self.metadata_column_names is not None:
                            batch_meta = np.empty((piece.batch_length, len(self.metadata_column_names)), dtype="object")
                    batch[batch_offset : (batch_offset + piece.piece_length)] = data
                    if self.metadata_column_names is not None:
                        batch_meta[batch_offset : (batch_offset + piece.piece_length)] = meta.to_numpy()
                    batch_offset += data.shape[0]
                    if piece.last_piece:
                        if self.metadata_column_names is not None:
                            meta_batch_df = pd.DataFrame(batch_meta, columns=self.metadata_column_names).infer_objects()
                            meta_batch_df["i"] = np.arange(start=piece.batch_start, stop=piece.batch_end)
                        else:
                            meta_batch_df = pd.DataFrame(
                                np.arange(start=piece.batch_start, stop=piece.batch_end), columns=["i"]
                            )
                        yield batch, meta_batch_df
                        batch = None
                        if self.metadata_column_names is not None:
                            batch_meta = None
                        batch_offset = 0

                    if show_progress:
                        pbar.update(1)
                    semaphore.release()
                except Exception as e:  # pylint: disable=broad-except
                    stopped = True
                    semaphore.release()
                    raise e

        if show_progress:
            pbar.close()

"""Numpy embedding reader, read embeddings from numpy files in streaming

The main logic of this reader is:
* read file headers to know the length and dimensions
* compute pieces to read from each file depending on batch size and max piece length
* read pieces in parallel
* concatenate pieces
"""

import pandas as pd
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import numpy as np
import re
import math
from collections import namedtuple
from embedding_reader.get_file_list import get_file_list
from embedding_reader.piece_builder import build_pieces, PIECES_BASE_COLUMNS
from threading import Semaphore


def read_numpy_header(f):
    """Read the header of a numpy file"""
    f.seek(0)
    file_size = f.size if isinstance(f.size, int) else f.size()
    first_line = f.read(min(file_size, 300)).split(b"\n")[0]
    result = re.search(r"'shape': \(([0-9]+), ([0-9]+)\)", str(first_line))
    shape = (int(result.group(1)), int(result.group(2)))
    dtype = re.search(r"'descr': '([<f0-9]+)'", str(first_line)).group(1)
    end = len(first_line) + 1  # the first line content and the endline
    f.seek(0)
    byte_per_item = np.dtype(dtype).itemsize * shape[1]
    return (shape[0], shape[1], dtype, end, byte_per_item)


def file_to_header(filename, fs):
    try:
        with fs.open(filename, "rb") as f:
            return (None, [filename, *read_numpy_header(f)])
    except Exception as e:  # pylint: disable=broad-except
        return e, (filename, None)


def get_numpy_headers(embeddings_file_paths, fs):
    """get numpy headers"""
    headers = []
    count_before = 0
    with ThreadPool(128) as p:
        for err, c in tqdm(
            p.imap(lambda f: file_to_header(f, fs), embeddings_file_paths), total=len(embeddings_file_paths)
        ):
            if err is not None:
                raise Exception(f"failed reading file {c[0]}") from err
            if c[1] == 0:
                continue
            headers.append([*c[0:2], count_before, *c[2:]])
            count_before += c[1]

    return headers


class NumpyReader:
    """Numpy reader class, implements init to read the files headers and call to procuce embeddings batches"""

    def __init__(self, embeddings_folder):
        self.embeddings_folder = embeddings_folder
        self.fs, embeddings_file_paths = get_file_list(embeddings_folder, "npy")

        headers = get_numpy_headers(embeddings_file_paths, self.fs)
        self.headers = pd.DataFrame(
            headers,
            columns=["filename", "count", "count_before", "dimension", "dtype", "header_offset", "byte_per_item"],
        )

        self.count = self.headers["count"].sum()
        if self.count == 0:
            raise ValueError(f"No embeddings found in folder {embeddings_folder}")
        self.nb_files = len(self.headers["count"])
        self.dimension = int(self.headers.iloc[0]["dimension"])
        self.byte_per_item = self.headers.iloc[0]["byte_per_item"]
        self.dtype = self.headers.iloc[0]["dtype"]
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
        max_ram_usage_in_bytes=2**32,
    ):
        if end is None:
            end = self.count

        if end > self.count:
            end = self.count
        if batch_size > end - start:
            batch_size = end - start

        if max_piece_size is None:
            # Take x embeddings per pieces so that the max piece size is 50MB
            max_piece_size = max(int(50 * 10**6 / self.byte_per_item), 1)

        if parallel_pieces is None:
            # We try to parallelize a maximum as long at it fits the ram constraint.
            # Since pieces are read with imap and that files parts are only read once,
            # we can estimate the ram usage for n pieces read in parallel as being (n*max_piece_size // self.max_file_size + 1 ) * self.max_file_size
            parallel_pieces = min(
                max(
                    math.floor(max_ram_usage_in_bytes / min(max_piece_size * self.byte_per_item, self.max_file_size)), 1
                ),
                50,
            )

        metadata_columns = ["header_offset"]
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

        def read_piece(piece):
            try:
                start = piece.piece_start
                end = piece.piece_end
                path = piece.filename
                header_offset = piece.header_offset

                with self.fs.open(path, "rb") as f:
                    length = end - start
                    f.seek(header_offset + start * self.byte_per_item)
                    return (
                        None,
                        (
                            np.frombuffer(f.read(length * self.byte_per_item), dtype=self.dtype).reshape(
                                (length, self.dimension)
                            ),
                            piece,
                        ),
                    )
            except Exception as e:  # pylint: disable=broad-except
                return e, (None, piece)

        semaphore = Semaphore(parallel_pieces)
        stopped = False

        def piece_generator(pieces):
            for piece in (Piece(*parts) for parts in zip(*[pieces[col] for col in cols])):
                if stopped:
                    break
                semaphore.acquire()  # pylint: disable=consider-using-with
                yield piece

        batch = None
        batch_offset = 0

        if show_progress:
            pbar = tqdm(total=len(pieces))
        with ThreadPool(parallel_pieces) as p:
            for err, (data, piece) in p.imap(read_piece, piece_generator(pieces)):
                if err is not None:
                    stopped = True
                    semaphore.release()
                    raise Exception(
                        f"failed reading file {piece.filename} from {piece.piece_start} to {piece.piece_end}"
                    ) from err
                try:
                    if batch is None:
                        batch = np.empty((piece.batch_length, self.dimension), "float32")

                    batch[batch_offset : (batch_offset + piece.piece_length)] = data
                    batch_offset += data.shape[0]
                    if piece.last_piece:
                        meta_batch_df = pd.DataFrame(
                            np.arange(start=piece.batch_start, stop=piece.batch_end), columns=["i"]
                        )
                        yield batch, meta_batch_df
                        batch = None
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

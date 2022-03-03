"""piece builder module provides the build_pieces function"""

import pandas as pd
from collections import namedtuple


def build_pieces(headers, batch_size, start, end, max_piece_size=100000, metadata_columns=None):
    """
    Build pieces function takes as input a list of headers and
    returns a list of pieces splitted in size maximum max_piece_size.
    Input: (filename, count, count_before)
    Output: (filename:str, piece_start:int, piece_end:int, batch_id:int, batch_length:int, last_piece:bool)

    This function is the main feature of embedding reader, it makes it possible to read many files
    in parallel and abstract away all the batching
    """

    if metadata_columns is None:
        metadata_columns = []

    columns = ["filename", "count", "count_before"] + metadata_columns

    filecount = namedtuple("FileCount", columns)

    items = [filecount(*args) for args in zip(*[headers[col] for col in columns])]

    header_i = 0
    while header_i < len(items) and items[header_i].count_before < start:
        continue

    read_current_file = 0
    read_current_batch = 0
    read_from_beginning = 0
    pieces = []

    for batch_id, batch_start in enumerate(range(start, end, batch_size)):
        batch_length = min(batch_size, end - batch_start)

        # building all pieces of this batch
        while header_i < len(items) and read_current_batch < batch_length:
            count_before = items[header_i].count_before
            count = items[header_i].count
            piece_start = batch_start + read_current_batch - count_before
            piece_length = min(count - read_current_file, batch_length - read_current_batch, max_piece_size)
            piece_end = piece_start + piece_length

            read_current_file += piece_length
            read_current_batch += piece_length
            read_from_beginning += piece_length
            piece_filename = items[header_i].filename
            last_piece = read_current_batch == batch_length
            batch_end = batch_start + batch_length
            piece = (
                piece_filename,
                piece_start,
                piece_end,
                piece_length,
                batch_id,
                batch_start,
                batch_end,
                batch_length,
                last_piece,
            )
            piece = piece + tuple(items[header_i][3 + col] for col in range(len(metadata_columns)))

            pieces.append(piece)

            if read_current_file == items[header_i].count:
                read_current_file = 0
                header_i += 1

        read_current_batch = 0

    return pd.DataFrame(
        pieces,
        columns=[
            "filename",
            "piece_start",
            "piece_end",
            "piece_length",
            "batch_id",
            "batch_start",
            "batch_end",
            "batch_length",
            "last_piece",
        ]
        + metadata_columns,
    )

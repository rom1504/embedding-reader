from unittest import result
from embedding_reader.piece_builder import build_pieces

import numpy as np
import random
import pandas as pd


def build_random_filecounts():
    count_before = 0
    results = []
    for i in range(1000):
        r = random.randint(100, 10000)
        results.append([r, count_before, str(i) + ".npy", "someval" + str(i)])

    return pd.DataFrame(results, columns=["count", "count_before", "filename", "custommeta"])


def test_piece_builder():
    # generate random file counts
    # call piece builder
    # check sum is correct
    # check group by file is fully correct
    # check each piece has a reasonable size

    batch_size = 1000
    start = 0
    end = 10000
    max_piece_size = 100

    file_counts = build_random_filecounts()
    pieces = build_pieces(
        file_counts, batch_size, start, end, max_piece_size=max_piece_size, metadata_columns=["custommeta"]
    )
    pieces["piece_length"].sum() == file_counts["count"].sum()

    # check group by filename of pieces is the same as file_counts
    pieces_grouped = pieces.groupby("filename")
    file_counts_grouped = file_counts.groupby("filename")
    for filename, pieces_group in pieces_grouped:
        file_counts_group = file_counts_grouped.get_group(filename)
        pieces_group["piece_length"].sum() == file_counts_group["count"].sum()
        pieces_group["custommeta"].values == file_counts_group["custommeta"].values

    # check each piece has a reasonable size
    pieces["piece_length"].max() <= max_piece_size

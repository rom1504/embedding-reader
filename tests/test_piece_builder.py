from embedding_reader.piece_builder import build_pieces

import random
import pytest
import pandas as pd


def build_random_filecounts(min_count=100, max_count=10000):
    count_before = 0
    results = []
    for i in range(1000):
        r = random.randint(min_count, max_count)
        results.append([r, count_before, str(i) + ".npy", "someval" + str(i)])
        count_before += r

    return pd.DataFrame(results, columns=["count", "count_before", "filename", "custommeta"])


@pytest.mark.parametrize(["start", "end"], [(0, 100000), (100, 100000), (10000, 300000)])
def test_piece_builder(start, end):
    # generate random file counts
    # call piece builder
    # check sum is correct
    # check each piece has a reasonable size

    batch_size = 1000
    max_piece_size = 100

    file_counts = build_random_filecounts()
    pieces = build_pieces(
        file_counts, batch_size, start, end, max_piece_size=max_piece_size, metadata_columns=["custommeta"]
    )
    assert pieces["piece_length"].sum() == end - start

    filename_to_count = {filename: count for count, filename in zip(file_counts["count"], file_counts["filename"])}

    for piece_start, piece_end, piece_length, batch_start, batch_end, batch_length, filename in zip(
        pieces["piece_start"],
        pieces["piece_end"],
        pieces["piece_length"],
        pieces["batch_start"],
        pieces["batch_end"],
        pieces["batch_length"],
        pieces["filename"],
    ):
        assert 0 < piece_length <= max_piece_size
        assert piece_start >= 0
        assert piece_start < piece_end
        assert batch_start < batch_end
        assert batch_length <= batch_size
        assert batch_end <= end
        assert batch_end - batch_start <= batch_size
        assert piece_end <= filename_to_count[filename]

    # check each piece has a reasonable size
    assert pieces["piece_length"].max() <= max_piece_size


@pytest.mark.parametrize(["start", "end"], [(9, 15)])
def test_piece_builder_with_empty_file(start, end):
    # generate random file counts
    # call piece builder
    # check piece length is not empty

    batch_size = 1000
    max_piece_size = 100

    file_counts = build_random_filecounts(min_count=0, max_count=1)
    pieces = build_pieces(
        file_counts, batch_size, start, end, max_piece_size=max_piece_size, metadata_columns=["custommeta"]
    )
    for piece_start, piece_end, piece_length, batch_start, batch_end, batch_length, filename in zip(
        pieces["piece_start"],
        pieces["piece_end"],
        pieces["piece_length"],
        pieces["batch_start"],
        pieces["batch_end"],
        pieces["batch_length"],
        pieces["filename"],
    ):
        assert piece_length != 0
        assert piece_start < piece_end

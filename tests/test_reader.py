from embedding_reader import EmbeddingReader
import random
import numpy as np
import pytest
import pandas as pd

from tests.fixtures import build_test_collection_numpy, build_test_collection_parquet, build_test_collection_parquet_npy


@pytest.mark.parametrize("file_format", ["npy", "parquet", "parquet_npy"])
@pytest.mark.parametrize("collection_kind", ["random", "simple"])
def test_embedding_reader(file_format, collection_kind, tmpdir):
    min_size = 1
    max_size = 1024
    dim = 512
    nb_files = 5

    if file_format == "npy":
        build_test_collection = build_test_collection_numpy
    elif file_format == "parquet":
        build_test_collection = build_test_collection_parquet
    elif file_format == "parquet_npy":
        build_test_collection = build_test_collection_parquet_npy
    tmp_dir, sizes, dim, expected_array, expected_meta = build_test_collection(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files, kind=collection_kind
    )
    batch_size = random.randint(min_size, max_size)
    embedding_reader = EmbeddingReader(
        tmp_dir,
        file_format=file_format,
        embedding_column="embedding",
        meta_columns=["id", "id2"],
        metadata_folder=tmp_dir,
    )

    assert embedding_reader.dimension == (dim,)
    assert embedding_reader.count == sum(sizes)
    assert embedding_reader.byte_per_item == dim * 4
    assert embedding_reader.total_size == embedding_reader.count * embedding_reader.byte_per_item

    it = embedding_reader(batch_size=batch_size)
    all_batches = list(it)
    all_shapes = [x[0].shape for x in all_batches]
    actual_array = np.vstack([x[0] for x in all_batches])
    if all_batches[0][1] is not None:
        actual_ids = pd.concat([x[1] for x in all_batches])

    assert all(s[0] == batch_size and s[1] == 512 for s in all_shapes[:-1])
    assert all_shapes[-1][0] <= batch_size and all_shapes[-1][1] == 512
    assert actual_array.shape == expected_array.shape
    np.testing.assert_almost_equal(actual_array, expected_array)
    if file_format == "parquet" or file_format == "parquet_npy":
        pd.testing.assert_frame_equal(
            actual_ids.reset_index(drop=True), expected_meta[["id", "id2", "i"]].reset_index(drop=True)
        )
    else:
        pd.testing.assert_frame_equal(actual_ids.reset_index(drop=True), expected_meta[["i"]].reset_index(drop=True))

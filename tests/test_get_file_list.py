from embedding_reader.get_file_list import get_file_list
from tests.fixtures import build_test_collection_parquet
from fsspec.implementations.local import LocalFileSystem


def test_get_file_list_with_single_input(tmpdir):
    tmp_dir = build_test_collection_parquet(tmpdir, tmpdir_name="a", nb_files=2)[0]
    fs, paths = get_file_list(path=tmp_dir, file_format="parquet")
    assert isinstance(fs, LocalFileSystem)
    assert len(paths) == 2


def test_get_file_list_with_multiple_inputs(tmpdir):
    tmp_dir1 = build_test_collection_parquet(tmpdir, tmpdir_name="a", nb_files=2)[0]
    tmp_dir2 = build_test_collection_parquet(tmpdir, tmpdir_name="b", nb_files=2)[0]
    fs, paths = get_file_list(path=[tmp_dir1, tmp_dir2], file_format="parquet")
    assert isinstance(fs, LocalFileSystem)
    assert len(paths) == 4


def test_get_file_list_with_multiple_multiple_levels_input(tmpdir):
    tmp_dir1 = build_test_collection_parquet(tmpdir, tmpdir_name="a", nb_files=2)[0]
    build_test_collection_parquet(tmpdir, tmpdir_name="a/1", nb_files=2)
    build_test_collection_parquet(tmpdir, tmpdir_name="a/1/2", nb_files=2)
    fs, paths = get_file_list(path=tmp_dir1, file_format="parquet")
    assert isinstance(fs, LocalFileSystem)
    assert len(paths) == 6

"""get_file_list module gets the file list from a path for both readers"""

import fsspec
from typing import List, Tuple, Union
import os


def get_file_list(path: Union[str, List[str]], file_format: str) -> Tuple[fsspec.AbstractFileSystem, List[str]]:
    """
    Get the file system and all the file paths that matches `file_format` under the given `path`.
    The `path` could a single folder or multiple folders.
    :raises ValueError: if file system is inconsistent under different folders.
    """
    if isinstance(path, str):
        return _get_file_list(path, file_format)
    all_file_paths = []
    fs = None
    for p in path:
        cur_fs, file_paths = _get_file_list(p, file_format, sort_result=False)
        if fs is None:
            fs = cur_fs
        elif fs != cur_fs:
            raise ValueError(
                f"The file system in different folder are inconsistent.\n" f"Got one {fs} and the other {cur_fs}"
            )
        all_file_paths.extend(file_paths)
    all_file_paths.sort()
    return fs, all_file_paths


def make_path_absolute(path: str) -> str:
    fs, p = fsspec.core.url_to_fs(path)
    if fs.protocol == "file":
        return os.path.abspath(p)
    return path


def _get_file_list(
    path: str, file_format: str, sort_result: bool = True
) -> Tuple[fsspec.AbstractFileSystem, List[str]]:
    """Get the file system and all the file paths that matches `file_format` given a single path."""
    path = make_path_absolute(path)
    fs, path_in_fs = fsspec.core.url_to_fs(path)
    prefix = path[: path.index(path_in_fs)]
    glob_pattern = path.rstrip("/") + f"/**.{file_format}"
    file_paths = fs.glob(glob_pattern)
    if sort_result:
        file_paths.sort()
    file_paths_with_prefix = [prefix + file_path for file_path in file_paths]
    return fs, file_paths_with_prefix

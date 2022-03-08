import numpy as np
import pandas as pd
import os
import py
import random
import math


def build_test_collection_numpy(
    tmpdir: py.path,
    min_size=2,
    max_size=10000,
    dim=512,
    nb_files=5,
    tmpdir_name: str = "autofaiss_numpy",
    kind="random",
):
    tmp_path = tmpdir.mkdir(tmpdir_name)
    sizes = [random.randint(min_size, max_size) for _ in range(nb_files)]
    dim = dim
    all_arrays = []
    file_paths = []
    for i, size in enumerate(sizes):
        if kind == "random":
            arr = np.random.rand(size, dim).astype("float32")
        elif kind == "simple":
            arr = (i * np.ones((size, dim))).astype("float32")
        all_arrays.append(arr)
        filename = str(i).zfill(int(math.log10(len(str(nb_files)))) + 1)
        file_path = os.path.join(tmp_path, f"{filename}.npy")
        file_paths.append(file_path)
        np.save(file_path, arr)
    all_arrays = np.vstack(all_arrays)
    return str(tmp_path), sizes, dim, all_arrays, None, file_paths


def build_test_collection_parquet(
    tmpdir: py.path,
    min_size=2,
    max_size=10000,
    dim=512,
    nb_files=5,
    tmpdir_name: str = "autofaiss_parquet",
    consecutive_ids=False,
    kind="random",
):
    tmp_path = tmpdir.mkdir(tmpdir_name)
    sizes = [random.randint(min_size, max_size) for _ in range(nb_files)]
    all_dfs = []
    file_paths = []
    n = 0
    for i, size in enumerate(sizes):
        if kind == "random":
            arr = np.random.rand(size, dim).astype("float32")
        elif kind == "simple":
            arr = (i * np.ones((size, dim))).astype("float32")
        if consecutive_ids:
            # ids would be consecutive from 0 to N-1
            ids = list(range(n, n + size))
        else:
            ids = np.random.randint(max_size * nb_files * 10, size=size)
        id2 = np.random.randint(max_size * nb_files * 10, size=size)
        df = pd.DataFrame({"embedding": list(arr), "id": ids, "id2": id2})
        all_dfs.append(df)
        filename = str(i).zfill(int(math.log10(len(str(nb_files)))) + 1)
        file_path = os.path.join(tmp_path, f"{filename}.parquet")
        df.to_parquet(file_path)
        file_paths.append(file_path)
        n += len(df)
    all_dfs = pd.concat(all_dfs)
    expected_array = np.vstack(all_dfs["embedding"])
    all_dfs["i"] = np.arange(start=0, stop=len(all_dfs))
    return str(tmp_path), sizes, dim, expected_array, all_dfs[["id", "id2", "i"]], file_paths


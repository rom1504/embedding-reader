"""
This is an example on how to use embedding reader to do an inference over a set of billion
of clip vit-l/14 embeddings to predict aesthetic clip embeddings that can be used inside knn or clip guiding
"""


from builtins import ValueError
from embedding_reader import EmbeddingReader
import fire
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import fsspec
import pandas as pd
import pyarrow.parquet as pq
import numpy as np


dir_path = os.path.dirname(os.path.realpath(__file__))

import torch
import torch.nn as nn
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m


import mmh3


def compute_hash(url, text):
    if url is None:
        url = ""

    if text is None:
        text = ""

    total = (url + text).encode("utf-8")
    return mmh3.hash64(total)[0]


def main(
    embedding_folder="https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/img_emb/",
    output_folder="/media/hd2/aethetic_emb_other",
    batch_size=10**6,
    end=10**7,
    model="vit_l_14"
):
    """main function"""
    reader = EmbeddingReader(embedding_folder, file_format="npy")
    fs, relative_output_path = fsspec.core.url_to_fs(output_folder)
    fs.mkdirs(relative_output_path, exist_ok=True)

    model = get_aesthetic_model(model)

    if model == "vit_l_14":
        embs = {i: np.zeros((1, 768), "float32") for i in range(10)}
    elif model == "vit_b_32":
        embs = {i: np.zeros((1, 512), "float32") for i in range(10)}
    else:
        raise ValueError(f"no model {model}")

    for i, (embeddings, _) in enumerate(reader(batch_size=batch_size, start=0, end=end)):
        with torch.no_grad():
            predictions = model(torch.tensor(embeddings)).cpu().numpy()
        for k in embs.keys():
            w = np.argwhere((predictions >= k) & (predictions <= k + 1))[:, 0]
            embs[k] += np.sum(np.take(embeddings, w, axis=0), axis=0)

        if i % 100 == 0:
            for k in embs.keys():
                v = embs[k]
                v = v / np.linalg.norm(v)
                np.save(os.path.join(relative_output_path, f"pre{i:05d}_rating{k}.npy"), v)

    for k in embs.keys():
        v = embs[k]
        v = v / np.linalg.norm(v)
        np.save(os.path.join(relative_output_path, f"rating{k}.npy"), v)


if __name__ == "__main__":
    fire.Fire(main)

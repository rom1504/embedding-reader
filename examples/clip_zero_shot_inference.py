"""
This is an example on how to use embedding reader to do an inference over a set of billion
of clip vit-l/14 embeddings to predict whether the coco classes for the images
"""


from embedding_reader import EmbeddingReader
import fire
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import fsspec
import math
import pandas as pd
import functools
import pyarrow.parquet as pq
import torch
import clip


@functools.lru_cache(maxsize=None)
def get_coco_classes():
    fs, p = fsspec.core.url_to_fs("https://github.com/pjreddie/darknet/raw/master/data/coco.names")
    with fs.open(p, "r") as f:
        cls = f.read().split("\n")[:-1]
    return cls


def get_prompt_embeddings():
    """load the safety model"""

    device = "cpu"
    model, _ = clip.load("ViT-L/14", device=device)

    CLASSES = get_coco_classes()
    text = clip.tokenize(["A photo of a " + c for c in CLASSES]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return np.transpose(text_features.cpu().numpy())


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
    metadata_folder="https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/laion2B-en-metadata/",
    output_folder="/media/hd2/cocov2",
    batch_size=10**6,
    end=None,
):
    """main function"""
    reader = EmbeddingReader(
        embedding_folder, metadata_folder=metadata_folder, file_format="parquet_npy", meta_columns=["url", "caption"]
    )
    fs, relative_output_path = fsspec.core.url_to_fs(output_folder)
    fs.mkdirs(relative_output_path, exist_ok=True)

    prompts = get_prompt_embeddings()

    total = reader.count
    batch_count = math.ceil(total // batch_size)
    padding = int(math.log10(batch_count)) + 1
    CLASSES = [c.replace(" ", "_") for c in get_coco_classes()]

    for i, (embeddings, ids) in enumerate(reader(batch_size=batch_size, start=0, end=end)):
        predictions = np.dot(embeddings, prompts)
        padded_id = str(i).zfill(padding)
        output_file_path = os.path.join(relative_output_path, padded_id + ".parquet")
        d = {c: predictions[:, i].tolist() for i, c in enumerate(CLASSES)}
        df = pd.DataFrame(d)
        df["hash"] = [compute_hash(x, y) for x, y in zip(ids["url"], ids["caption"])]
        df["url"] = ids["url"]
        with fs.open(output_file_path, "wb") as f:
            df.to_parquet(f)


if __name__ == "__main__":
    fire.Fire(main)

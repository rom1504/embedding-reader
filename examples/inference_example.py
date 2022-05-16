"""
This is an example on how to use embedding reader to do an inference over a set of billion
of clip vit-l/14 embeddings to predict whether the corresponding images are safe or not
"""


from embedding_reader import EmbeddingReader
import fire
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import fsspec
import math
import pandas as pd
import gc


def load_safety_model():
    """load the safety model"""
    import autokeras as ak  # pylint: disable=import-outside-toplevel
    from tensorflow.keras.models import load_model  # pylint: disable=import-outside-toplevel
    from os.path import expanduser  # pylint: disable=import-outside-toplevel

    home = expanduser("~")

    cache_folder = home + "/.cache/clip_retrieval"
    model_dir = cache_folder + "/clip_autokeras_binary_nsfw"
    if not os.path.exists(model_dir):
        os.makedirs(cache_folder, exist_ok=True)

        from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

        path_to_zip_file = cache_folder + "/clip_autokeras_binary_nsfw.zip"
        url_model = (
            "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_binary_nsfw.zip"
        )
        urlretrieve(url_model, path_to_zip_file)
        import zipfile  # pylint: disable=import-outside-toplevel

        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(cache_folder)

    loaded_model = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS)
    loaded_model.predict(np.random.rand(10**3, 768).astype("float32"), batch_size=10**3)

    return loaded_model


import mmh3


def compute_hash(url, text):
    if url is None:
        url = ""

    if text is None:
        text = ""

    total = (url + text).encode("utf-8")
    return mmh3.hash64(total)[0]


def main(
    embedding_folder="https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion1B-nolang/img_emb/",
    metadata_folder="https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion1B-nolang/laion1B-nolang-metadata/",
    output_folder="output",
    batch_size=10**5,
    end=None,
):
    """main function"""
    reader = EmbeddingReader(
        embedding_folder, metadata_folder=metadata_folder, file_format="parquet_npy", meta_columns=["url", "caption"]
    )
    fs, relative_output_path = fsspec.core.url_to_fs(output_folder)
    fs.mkdirs(relative_output_path, exist_ok=True)

    model = load_safety_model()

    total = reader.count
    batch_count = math.ceil(total // batch_size)
    padding = int(math.log10(batch_count)) + 1
    import tensorflow as tf  # pylint: disable=import-outside-toplevel

    for i, (embeddings, ids) in enumerate(reader(batch_size=batch_size, start=0, end=end)):
        predictions = model.predict_on_batch(embeddings)
        batch = np.hstack(predictions)
        padded_id = str(i).zfill(padding)
        output_file_path = os.path.join(relative_output_path, padded_id + ".parquet")
        df = pd.DataFrame(batch, columns=["prediction"])
        df["hash"] = [compute_hash(x, y) for x, y in zip(ids["url"], ids["caption"])]
        df["url"] = ids["url"]
        with fs.open(output_file_path, "wb") as f:
            df.to_parquet(f)


if __name__ == "__main__":
    fire.Fire(main)

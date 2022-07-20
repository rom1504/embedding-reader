# embedding_reader
[![pypi](https://img.shields.io/pypi/v/embedding_reader.svg)](https://pypi.python.org/pypi/embedding_reader)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rom1504/embedding_reader/blob/master/notebook/embedding_reader_getting_started.ipynb)
[![Try it on gitpod](https://img.shields.io/badge/try-on%20gitpod-brightgreen.svg)](https://gitpod.io/#https://github.com/rom1504/embedding_reader)

Embedding reader is a module to make it easy to read efficiently a large collection of embeddings stored in any file system.
* 400GB of embeddings read in 8min using an nvme drive
* 400GB of embeddings read in 40min using an hdd drive
* 400GB of embeddings read in 1.3h from aws s3

## Install

pip install embedding_reader

## Python examples

Checkout these examples to call this as a lib:
* [example.py](examples/example.py)

### Simple example

```python
from embedding_reader import EmbeddingReader

embedding_reader = EmbeddingReader(embeddings_folder="embedding_folder", file_format="npy")

print("embedding count", embedding_reader.count)
print("dimension", embedding_reader.dimension)
print("total size", embedding_reader.total_size)
print("byte per item", embedding_reader.byte_per_item)

for emb, meta in embedding_reader(batch_size=10 ** 6, start=0, end=embedding_reader.count):
    print(emb.shape)
```

### Laion5B example

In [laion5B](https://laion.ai/laion-5b-a-new-era-of-open-large-scale-multi-modal-datasets/) you can find 5B ViT-L/14 image embeddings, you can read them with that code:

```python
from embedding_reader import EmbeddingReader

embedding_reader = EmbeddingReader(embeddings_folder="https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/img_emb/", file_format="npy")

print("embedding count", embedding_reader.count)
print("dimension", embedding_reader.dimension)
print("total size", embedding_reader.total_size)
print("byte per item", embedding_reader.byte_per_item)

for emb, meta in embedding_reader(batch_size=10 ** 6, start=0, end=embedding_reader.count):
    print(emb.shape)
```
It takes about 3h to read laion2B-en embeddings at 300MB/s

### Numpy & Parquet Metadata Example

The parquet_npy format supports reading from both a .npy collection and a .parquet collection that are in the same order.
Here is an example of usage:

```python
from embedding_reader import EmbeddingReader

embedding_reader = EmbeddingReader(
    embeddings_folder="embedding_folder",
    metadata_folder="metadata_folder",
    meta_columns=['image_path', 'caption'],
    file_format="parquet_npy"
)

for emb, meta in embedding_reader(batch_size=10 ** 6, start=0, end=embedding_reader.count):
    print(emb.shape)
    print(meta["image_path"], meta["caption"])
```
`emb` is a numpy array like the previous examples while `meta` is a pandas dataframe with the columns requested in `meta_columns`.

## Who is using embedding reader?

Some use cases of embedding reader include:
* building knn indices in [autofaiss](https://github.com/criteo/autofaiss)
* computing zero shot attributes using clip
* running training or inferences of linear layer models on top of embeddings

Embeddings are a powerful concept, they allow turning highly complex data into point in a linearly separable space.
Embeddings are also much smaller and more efficient to manipulate than usual data (images, audio, video, text, interaction items, ...)

To learn more about embeddings read [Semantic search blogpost](https://rom1504.medium.com/semantic-search-with-embeddings-index-anything-8fb18556443c)

## File system support

Thanks to [fsspec](https://filesystem-spec.readthedocs.io/en/latest/), embedding_reader supports reading and writing files in [many file systems](https://github.com/fsspec/filesystem_spec/blob/6233f315548b512ec379323f762b70764efeb92c/fsspec/registry.py#L87).
To use it, simply use the prefix of your filesystem before the path. For example `hdfs://`, `s3://`, `http://`, or `gcs://`.
Some of these file systems require installing an additional package (for example s3fs for s3, gcsfs for gcs).
See fsspec doc for all the details.

## API

This module exposes one class:

### EmbeddingReader(folder, file_format, embedding_column="embedding", meta_columns=None, metadata_folder=None)

initialize the reader by listing all files and retrieving their metadata

* **folder** embeddings folder. Can also be a list of folders. (*required*)
* **file_format** parquet, npy or parquet_npy. (*required*)
* **embedding_column** embedding column in parquet. (*default embedding*)
* **meta_columns** meta columns in parquet. (*default None*)
* **metadata_folder** metadata folder, used by the parquet_npy reader (*default None*)

#### .embeddings_folder

the embedding folder

#### .count

total number of embedding in this folder

#### .dimension

dimension of one embedding

#### .byte_per_item

size of one embedding in bytes

#### .total_size

size in bytes of the collection

#### .nb_files

total number of embedding files in this folder

#### .max_file_size

max size in bytes of the embedding files of the collection

#### __call__(batch_size, start=0, end=None, max_piece_size=None, parallel_pieces=None, show_progress=True, max_ram_usage_in_bytes=2**31)

Produces an iterator that yields tuples (data, meta) with the given batch_size

* **batch_size** amount of embeddings in one batch. (*required*)
* **start** start of the subset of the collection to read. (default *0*)
* **end** end of the subset of the collection to read. (default *end of collection*)
* **max_piece_size** maximum size of a piece. The default value works for most cases. Increase or decrease based on your file system performances (default *max(number of embedding for 50MB, batch size in MB)*)
* **parallel_pieces** Number of pieces to read in parallel. Increase or decrease depending on your filesystem. (default *~min(round(max_ram_usage_in_bytes/max_piece_size*byte_per_item), 50)*)
* **show_progress** Display a tqdm bar with the number of pieces done. (default *True*)
* **max_ram_usage_in_bytes** Constraint the ram usage of embedding reader. The exact max ram usage is *min(max_ram_usage_in_bytes, size of a batch in bytes)*. (default 4GB)


## Architecture notes and benchmark

The main architecture choice of this lib is the `build_pieces` function that builds decently sizes pieces of embedding files (typically 50MB) initially.
These pieces metadata can then be used to fetch in parallel these pieces, which are then used to build the embedding batches and provided to the user.
In order to reach the maximal speed, it is better to read files of equal size. The number of threads used is constrained by the maximum size of your embeddings files: the lower the size, the more threads are used (you can also set a custom number of threads, but ram consumption will be higher).

In practice, it has been observed speed of up to 100MB/s when fetching embeddings from s3, 1GB/s when fetching from an nvme drive.
That means reading 400GB of embeddings in 8 minutes (400M embeddings in float16 and dimension 512)
The memory usage stays low and flat thanks to the absence of copy. Decreasing the batch size decreases the amount of memory consumed, you can also set max_ram_usage_in_bytes to have a better control on the ram usage.


## For development

Either locally, or in [gitpod](https://gitpod.io/#https://github.com/rom1504/embedding_reader) (do `export PIP_USER=false` there)

Setup a virtualenv:

```
python3 -m venv .env
source .env/bin/activate
pip install -e .
```

to run tests:
```
pip install -r requirements-test.txt
```
then 
```
make lint
make test
```

You can use `make black` to reformat the code

`python -m pytest -x -s -v tests -k "dummy"` to run a specific test


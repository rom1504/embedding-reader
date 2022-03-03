from embedding_reader import EmbeddingReader
from tqdm import tqdm

embedding_reader = EmbeddingReader(embeddings_folder="embedding_folder", file_format="npy")

print("embedding count", embedding_reader.count)
print("dimension", embedding_reader.dimension)
print("total size", embedding_reader.total_size)
print("byte per item", embedding_reader.byte_per_item)

for emb, meta in tqdm(
    embedding_reader(batch_size=10 ** 6, start=0, end=embedding_reader.count), total=embedding_reader.count // 10 ** 6
):
    print(emb.shape)

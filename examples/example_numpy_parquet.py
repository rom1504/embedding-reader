from embedding_reader import EmbeddingReader

embedding_reader = EmbeddingReader(
    embeddings_folder="https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion1B-nolang/img_emb/",
    metadata_folder="https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion1B-nolang/laion1B-nolang-metadata/",
    meta_columns=["url", "caption"],
    file_format="parquet_npy",
)
print("embedding count", embedding_reader.count)
print("dimension", embedding_reader.dimension)
print("total size", embedding_reader.total_size)
print("byte per item", embedding_reader.byte_per_item)
for emb, meta in embedding_reader(batch_size=10 ** 6, start=0, end=embedding_reader.count):
    print(emb.shape)
    print(meta["url"], meta["caption"])

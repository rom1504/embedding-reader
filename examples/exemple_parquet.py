from embedding_reader import EmbeddingReader

print("Starting")
embedding_reader = EmbeddingReader(
    embeddings_folder="hdfs://root/user/cailimage/prod/image-text-pipeline/EU/prediction/clip/embeddings/with_image=true",
    embedding_column="image_embs",
    file_format="parquet",
)
print("embedding count", embedding_reader.count)
print("dimension", embedding_reader.dimension)
print("total size", embedding_reader.total_size)
print("byte per item", embedding_reader.byte_per_item)


for emb, meta in embedding_reader(
    batch_size=10**6, start=0, end=embedding_reader.count, max_ram_usage_in_bytes=2**30
):
    pass

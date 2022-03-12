"""Reader module exposes the reading functionality of all formats"""

from embedding_reader.numpy_reader import NumpyReader
from embedding_reader.parquet_reader import ParquetReader


class EmbeddingReader:
    """reader class, implements init to read the files headers and call to produce embeddings batches"""

    def __init__(self, embeddings_folder, file_format="npy", embedding_column="embedding", meta_columns=None):
        if file_format == "npy":
            self.reader = NumpyReader(embeddings_folder)
        elif file_format == "parquet":
            self.reader = ParquetReader(
                embeddings_folder, embedding_column_name=embedding_column, metadata_column_names=meta_columns
            )
        else:
            raise ValueError("format must be npy or parquet")

        self.dimension = self.reader.dimension
        self.count = self.reader.count
        self.byte_per_item = self.reader.byte_per_item
        self.total_size = self.reader.total_size
        self.embeddings_folder = embeddings_folder

    def __call__(self, batch_size, start=0, end=None, max_piece_size=None, parallel_pieces=None, show_progress=True):
        return self.reader(batch_size, start, end, max_piece_size, parallel_pieces, show_progress)

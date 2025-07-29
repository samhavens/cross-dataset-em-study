"""
Drop-in replacement for SentenceTransformer that supports both local and TEI/OpenAI endpoints
"""
from typing import List, Union

import numpy as np

from tqdm import tqdm


class EmbeddingProvider:
    """
    Drop-in replacement for SentenceTransformer that supports:
    1. Local SentenceTransformer models (when base_url=None)
    2. TEI/OpenAI compatible endpoints (when base_url provided)

    Maintains exact same interface as SentenceTransformer.encode()
    """

    def __init__(self, model_name_or_path: str, base_url: str = None):
        """
        Initialize embedding provider

        Args:
            model_name_or_path: Model name (for SentenceTransformer locally, or model name for API endpoint)
            base_url: If provided, use OpenAI client with this base URL. If None, use SentenceTransformer
        """
        self.model_name = model_name_or_path
        self.base_url = base_url
        self.client = None
        self.model = None

        if base_url:
            # Use OpenAI client for any compatible endpoint
            try:
                from openai import OpenAI
                self.client = OpenAI(base_url=f"{base_url}/v1")
                print(f"Using OpenAI-compatible endpoint: {base_url} with model: {model_name_or_path}")
            except ImportError:
                raise ImportError("openai package required for API endpoint. Install with: pip install openai")
        else:
            # Use SentenceTransformer
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name_or_path)
                print(f"Using local SentenceTransformer: {model_name_or_path}")
            except ImportError:
                raise ImportError("sentence-transformers package required for local models. Install with: pip install sentence-transformers")

    def encode(self, sentences: Union[str, List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = False,
               **kwargs) -> np.ndarray:
        """
        Encode sentences to embeddings

        Args:
            sentences: Single string or list of strings to encode
            batch_size: Batch size for processing
            show_progress_bar: Whether to show progress bar
            **kwargs: Additional arguments (ignored for TEI, passed to SentenceTransformer)

        Returns:
            numpy array of embeddings
        """
        # Handle single string input
        if isinstance(sentences, str):
            sentences = [sentences]

        if self.client:
            # Use TEI/OpenAI endpoint
            return self._encode_tei(sentences, batch_size, show_progress_bar)
        # Use SentenceTransformer
        return self.model.encode(sentences, batch_size=batch_size,
                               show_progress_bar=show_progress_bar, **kwargs)

    def _encode_tei(self, sentences: List[str], batch_size: int, show_progress_bar: bool) -> np.ndarray:
        """Encode using TEI/OpenAI endpoint with batching"""
        embeddings = []

        # Create batches
        batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]

        # Process with optional progress bar
        iterator = tqdm(batches, desc="Encoding") if show_progress_bar else batches

        for batch in iterator:
            try:
                # API supports batch input
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )

                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

            except Exception as e:
                print(f"Warning: TEI embedding failed for batch: {e}")
                # Fallback: return zero embeddings for this batch
                batch_embeddings = [[0.0] * 384] * len(batch)  # Assume 384 dim, adjust if needed
                embeddings.extend(batch_embeddings)

        return np.array(embeddings)

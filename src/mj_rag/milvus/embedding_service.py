from typing import Optional

from mj_rag.interfaces import EmbeddingServiceInterface
from pymilvus import (
    model,
)


class OpenAIEmbeddingService(model.dense.OpenAIEmbeddingFunction, EmbeddingServiceInterface):

    def __init__(self,
                 model_name: str = "text-embedding-3-small",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 dimensions: Optional[int] = 1536,
                 **kwargs):
        model.dense.OpenAIEmbeddingFunction.__init__(
            self,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            dimensions=dimensions # Set the embedding dimensionality according to MRL feature.
        )
        self.dimensions = dimensions

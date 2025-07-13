from pydantic import BaseModel, Field
from typing import List, Optional, Any


class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the RAG pipeline.")


class RAGUsedImage(BaseModel):
    image_url: str = Field(..., description="The URL of the image used in the RAG response.")
    price: Optional[float] = Field(..., description="The price of the item.")
    description: str = Field(..., description="A description of the item.")


class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The request ID.")
    answer: str = Field(..., description="The content of the RAG response.")
    used_image_urls: List[RAGUsedImage]


from pydantic import BaseModel, Field
from typing import List, Optional


class IngestRequest(BaseModel):
    pdf_path: str = Field(default="data/aws_customer_agreement.pdf")


class IngestResponse(BaseModel):
    message: str
    chunks_created: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Question to ask about the document")


class SourceChunk(BaseModel):
    chunk_id: int
    text: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceChunk]
    answer_found: bool
    response_time_ms: float


class FrequentQuery(BaseModel):
    query: str
    count: int


class UnansweredQuery(BaseModel):
    query: str
    count: int


class AnalyticsResponse(BaseModel):
    total_queries: int
    avg_response_latency_ms: float
    most_frequent_queries: List[FrequentQuery]
    unanswered_queries: List[UnansweredQuery]

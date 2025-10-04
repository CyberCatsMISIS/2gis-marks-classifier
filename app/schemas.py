from typing import List

from pydantic import BaseModel


class TagRequest(BaseModel):
    text: str


class TagResponse(BaseModel):
    tags: List[str]

from fastapi import FastAPI

from app.models import predict_tags
from app.schemas import TagRequest, TagResponse

app = FastAPI(title="2GIS Marks Classifier")


@app.post("/parse", response_model=TagResponse)
async def parse_tags(text: TagRequest):
    tags = predict_tags(text.text)
    return TagResponse(tags=tags)

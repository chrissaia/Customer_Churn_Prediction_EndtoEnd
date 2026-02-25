import gradio
import uvicorn
from fastapi import FastAPI

app = FastAPI(
    title="Telco Churn Model",
    description="Building a model that predicts whether a person is churn or not",
    version="0.0.1",
)

# check root health - Required for AWS
@app.get("/")
def read_root():
    return {"status": "ok"}


@app.get("/predict")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}
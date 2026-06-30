"""FastAPI application for the SHL Assessment Recommender."""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

import agent

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")


class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: Optional[list[Recommendation]]
    end_of_conversation: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if len(request.messages) > 16:
        raise HTTPException(status_code=400, detail="Too many messages (max 16 per request).")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        result = agent.chat(messages)
    except Exception as exc:
        logger.error("Agent error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal agent error. Please try again.")

    recs = None
    if result.get("recommendations"):
        recs = [Recommendation(**r) for r in result["recommendations"]]

    return ChatResponse(
        reply=result["reply"],
        recommendations=recs,
        end_of_conversation=result["end_of_conversation"],
    )

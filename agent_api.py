from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Generator
import ollama

app = FastAPI(
    title="Multi-Agent Hints API",
    description="An API using Ollama (LLaMA 2) to generate or stream agent collaboration hints.",
    version="2.0"
)

class AgentHint(BaseModel):
    agent_name: str
    hint: str

@app.get("/")
async def root():
    return {
        "message": "🎉 Welcome to the Multi-Agent Hints API!",
        "docs": "http://127.0.0.1:8000/docs",
        "instructions": "Use /ask-hints to send a question and stream the response from llama2"
    }

@app.get("/ask-hints")
async def ask_hints(question: str = Query(..., description="Your question to LLaMA 2")):
    """
    Accepts a user question and streams the LLaMA 2 response using Ollama.
    """

    def stream_response() -> Generator[str, None, None]:
        try:
            for chunk in ollama.chat(
                model="llama2",
                messages=[{"role": "user", "content": question}],
                stream=True
            ):
                content = chunk.get("message", {}).get("content", "")
                yield content
        except Exception as e:
            yield f"\n\n[ERROR] {str(e)}"

    return StreamingResponse(stream_response(), media_type="text/plain")

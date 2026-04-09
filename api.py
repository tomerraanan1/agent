import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import agent as graph, extract_text

app = FastAPI(title="LangGraph Agent API")

HISTORY_DIR = "histories"
os.makedirs(HISTORY_DIR, exist_ok=True)


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    user_id: str
    response: str


def get_history(user_id: str) -> list:
    path = os.path.join(HISTORY_DIR, f"{user_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_history(user_id: str, history: list):
    path = os.path.join(HISTORY_DIR, f"{user_id}.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        history = get_history(req.user_id)
        history.append({"role": "user", "content": req.message})

        result = await graph.ainvoke({"messages": history})
        last = result["messages"][-1]
        text = extract_text(last.content)

        history.append({"role": "assistant", "content": text})
        save_history(req.user_id, history)

        return ChatResponse(user_id=req.user_id, response=text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history/{user_id}")
def clear_history(user_id: str):
    path = os.path.join(HISTORY_DIR, f"{user_id}.json")
    if os.path.exists(path):
        os.remove(path)
    return {"message": f"History cleared for {user_id}"}


@app.get("/health")
def health():
    return {"status": "ok"}

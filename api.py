import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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

        result = await graph.ainvoke({"messages": history, "steps": 0})
        last = result["messages"][-1]
        text = extract_text(last.content)

        history.append({"role": "assistant", "content": text})
        save_history(req.user_id, history)

        return ChatResponse(user_id=req.user_id, response=text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Stream agent execution as Server-Sent Events (SSE).

    Each event is a JSON object with a `type` field:
      {"type": "tool_call",    "tool": "<name>", "input": {...}}
      {"type": "tool_result",  "tool": "<name>", "output": "..."}
      {"type": "response",     "content": "..."}
      {"type": "done",         "response": "..."}
      {"type": "error",        "detail": "..."}
    """
    async def event_stream():
        history = get_history(req.user_id)
        history.append({"role": "user", "content": req.message})
        final_text = ""

        try:
            async for chunk in graph.astream({"messages": history, "steps": 0}):
                for node_name, state_update in chunk.items():
                    messages = state_update.get("messages", [])
                    if not messages:
                        continue

                    if node_name == "tools":
                        # Each message is a tool result
                        for msg in messages:
                            tool_name = getattr(msg, "name", "unknown")
                            content = extract_text(msg.content)
                            yield f"data: {json.dumps({'type': 'tool_result', 'tool': tool_name, 'output': content})}\n\n"

                    elif node_name == "llm":
                        last = messages[-1]
                        tool_calls = getattr(last, "tool_calls", [])
                        if tool_calls:
                            for tc in tool_calls:
                                yield f"data: {json.dumps({'type': 'tool_call', 'tool': tc['name'], 'input': tc['args']})}\n\n"
                        else:
                            text = extract_text(last.content)
                            if text:
                                final_text = text
                                yield f"data: {json.dumps({'type': 'response', 'content': text})}\n\n"

            history.append({"role": "assistant", "content": final_text})
            save_history(req.user_id, history)
            yield f"data: {json.dumps({'type': 'done', 'response': final_text})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/history/{user_id}")
def clear_history(user_id: str):
    path = os.path.join(HISTORY_DIR, f"{user_id}.json")
    if os.path.exists(path):
        os.remove(path)
    return {"message": f"History cleared for {user_id}"}


@app.get("/health")
def health():
    return {"status": "ok"}

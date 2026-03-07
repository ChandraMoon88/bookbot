from fastapi import FastAPI, Request
import base64
import os

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok", "service": "AI processor"}

@app.post("/process")
async def process_message(request: Request):
    try:
        data = await request.json()
        sender_id = data.get("sender_id", "unknown")
        message = data.get("message", "")
        msg_type = data.get("type", "text")
        
        print(f"Received: sender={sender_id}, type={msg_type}, msg={message}")
        
        return {
            "text": f"Echo: {message}",
            "audio_b64": None,
            "lang": "en"
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "text": "Error occurred",
            "audio_b64": None,
            "lang": "en"
        }
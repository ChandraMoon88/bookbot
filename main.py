from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse
import httpx
import os

app = FastAPI()

VERIFY_TOKEN = "mybot123"
PAGE_ACCESS_TOKEN = "EAAWVRMLpTZAUBQyFZAULnQT6IYHPuAEZBLKTYTrNwfUrPLSTavpnAHVhDlP6weUQpuwi6jXUwZAVCI9O1ZBF8axTI2wIKcHwyaiThXl74qyZCovEGOKuyiCcD6BnSe6ZAstYlkZAgXzc20B0kmO4D9fKY7FQEfuGxWS9HgNQ7oasz1QZB7C9ibtWzkT2THNpcS4iR728Ih0oTlQZDZD"  # ← from Meta dashboard

WELCOME_MESSAGE = (
    "Welcome to BookBot! 🏨\n\n"
    "I'm your hotel booking assistant. I can help you with:\n"
    "• Search & book hotel rooms\n"
    "• Check availability\n"
    "• Manage your reservations\n\n"
    "Type 'book' to start a new booking or 'help' to see all options."
)

GREETING_KEYWORDS = {"hi", "hello", "hey", "hlo", "hii", "howdy", "greetings", "good morning", "good afternoon", "good evening"}


# Webhook Verification
@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        print("Webhook Verified!")
        return PlainTextResponse(content=hub_challenge, status_code=200)
    return PlainTextResponse(content="Forbidden", status_code=403)


# Receive Messages
@app.post("/webhook")
async def receive_message(request: Request):
    data = await request.json()

    if data.get("object") == "page":
        for entry in data["entry"]:
            for event in entry.get("messaging", []):
                sender_id = event["sender"]["id"]

                # Handle "Get Started" button or any postback
                if "postback" in event:
                    payload = event["postback"].get("payload", "")
                    print(f"Postback from {sender_id}: {payload}")
                    if payload == "GET_STARTED":
                        await send_message(sender_id, WELCOME_MESSAGE)

                elif "message" in event:
                    user_message = event["message"].get("text", "")
                    print(f"Message from {sender_id}: {user_message}")

                    # Show welcome message on greeting
                    if user_message.strip().lower() in GREETING_KEYWORDS:
                        await send_message(sender_id, WELCOME_MESSAGE)
                    else:
                        # Reply to user
                        await send_message(sender_id, f"You said: {user_message}")

    return {"status": "ok"}


# ✅ Send Message Function
async def send_message(recipient_id: str, message_text: str):
    url = "https://graph.facebook.com/v17.0/me/messages"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {PAGE_ACCESS_TOKEN}"
    }
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text}
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        print("Reply sent:", response.json())


# ✅ Health check
@app.get("/")
def root():
    return {"status": "Bot is running! ✅"}

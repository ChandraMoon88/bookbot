import os
import redis

# Connect to Upstash Redis
redis_client = redis.from_url(
    os.getenv("REDIS_URL"),
    decode_responses=True
)

# Save conversation state
def save_session(messenger_id, data):
    redis_client.setex(
        f"session:{messenger_id}",
        1800,  # expires in 30 minutes
        str(data)
    )

# Get conversation state
def get_session(messenger_id):
    return redis_client.get(f"session:{messenger_id}")

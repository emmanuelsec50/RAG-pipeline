import random
from locust import HttpUser, between, task

PROMPTS = [
    "What is the tuition fee for a Bachelor's degree at Embu University?",
    "How do I apply for admission to Embu University?",
    "What courses does the School of Engineering offer?",
]


class RAGChatUser(HttpUser):
    # wait_time = 0 => every user is always mid-request,
    # so with -u N you get ~N simultaneous open SSE streams (worst case).
    wait_time = between(0, 0)

    def on_start(self):
        # unique chat per user so conversations don't collide in Redis
        self.chat_id = f"loadtest-{self.environment.runner.client_id}-{id(self)}"

    @task
    def chat(self):
        with self.client.post(
            "/",
            json={"prompt": random.choice(PROMPTS), "chat_id": self.chat_id},
            catch_response=True,
            name="api/query",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:300]}")
                return
            if "[DONE]" not in resp.text:
                resp.failure("stream did not terminate with [DONE]")
            else:
                resp.success()
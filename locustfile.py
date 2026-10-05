from locust import HttpUser, task, between


class RAGUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def query_esim(self):
        self.client.post(
            "/api/v1/query",
            json={
                "query": "What is the policy for eSIM activation?",
                "top_k": 3,
            },
        )

    @task(1)
    def check_health(self):
        self.client.get("/health")
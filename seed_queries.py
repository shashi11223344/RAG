"""
Seed script: fires 35 test queries at POST /ask to populate the query_logs table
with realistic data (a mix of answerable and out-of-scope questions).

Usage:
    python seed_queries.py
"""

import time
import requests

API_BASE = "http://localhost:8000"

# 25 answerable questions + 10 out-of-scope
QUERIES = [
    # Answerable — Service & Usage
    "What is the AWS Customer Agreement?",
    "When does the AWS Customer Agreement take effect?",
    "What services are covered under the AWS Customer Agreement?",
    "Can I use third-party content on AWS?",
    "How does AWS protect the security of my content?",
    "Can I specify the AWS region where my content is stored?",
    "How much prior notice must AWS give before discontinuing a service?",
    "How much notice does AWS give before adverse changes to SLAs?",
    # Answerable — Fees & Payment
    "How does AWS calculate and bill fees?",
    "What happens if my default payment method fails?",
    "What interest rate does AWS charge on late payments?",
    "How much notice does AWS give before increasing fees?",
    "Are fees exclusive of indirect taxes?",
    # Answerable — Suspension & Termination
    "Under what conditions can AWS suspend my account?",
    "What are my responsibilities during a suspension period?",
    "How can I terminate the AWS Customer Agreement?",
    "How much advance notice must AWS give to terminate for convenience?",
    "What happens to my content after termination?",
    "How long after termination can I retrieve my content?",
    # Answerable — Liability & Indemnification
    "What is the aggregate liability cap under the agreement?",
    "What damages are excluded from AWS liability?",
    "When must I indemnify AWS?",
    "Does AWS defend against intellectual property infringement claims?",
    # Answerable — Misc
    "Can I assign the AWS Customer Agreement to another party?",
    "What governing law applies to customers in the United States?",
    # Out-of-scope (should return 'cannot find')
    "What is the price of Amazon EC2 instances?",
    "How do I set up an S3 bucket?",
    "What is the capital of France?",
    "Who is the CEO of Amazon?",
    "What programming languages does AWS Lambda support?",
    "How many data centers does AWS have worldwide?",
    "What is the difference between AWS and Azure?",
    "How do I configure a VPC?",
    "What is AWS's refund policy for Reserved Instances?",
    "Does AWS offer a free tier for new customers?",
]


def main():
    print(f"Seeding {len(QUERIES)} queries against {API_BASE}/ask …\n")
    for i, query in enumerate(QUERIES, 1):
        try:
            resp = requests.post(f"{API_BASE}/ask", json={"query": query}, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                found = "✓" if data["answer_found"] else "✗"
                print(f"[{i:02d}] {found}  ({data['response_time_ms']:.0f} ms)  {query[:70]}")
            else:
                print(f"[{i:02d}] ERROR {resp.status_code}: {resp.json().get('detail', resp.text)[:80]}")
        except requests.exceptions.ConnectionError:
            print("Cannot reach backend. Is FastAPI running?")
            break
        time.sleep(0.5)

    print("\nDone. Run GET /analytics to see the aggregated results.")


if __name__ == "__main__":
    main()

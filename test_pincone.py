# test_pinecone.py
import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("telecom-policies")

stats = index.describe_index_stats()
print("📊 Pinecone Index Stats:")
print(stats)
import json
import faiss
import numpy as np
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# -------------------
# 1. Setup
# -------------------
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

if not endpoint or not api_key:
    raise ValueError("Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables.")

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint
)

# Use absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "framework_questions.json")
INDEX_FILE = os.path.join(BASE_DIR, "framework_faiss.index")
META_FILE = os.path.join(BASE_DIR, "framework_meta.json")

# -------------------
# 2. Load Questions
# -------------------
print(f"Loading questions from {DATA_FILE}...")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Found {len(data)} questions.")

# -------------------
# 3. Generate embeddings and build FAISS index
# -------------------
print("Generating embeddings (this may take a while)...")

dim = 1536  # embedding dimension for text-embedding-ada-002
index = faiss.IndexFlatL2(dim)

# Store metadata separately
questions = []
stages = []
frameworks = []

embeddings = []

for i, item in enumerate(data):
    q = item["question"]
    
    # Progress indicator
    if (i + 1) % 50 == 0:
        print(f"  Processed {i + 1}/{len(data)} questions...")
    
    # Get embedding
    try:
        emb = client.embeddings.create(
            model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
            input=q
        ).data[0].embedding
        
        embeddings.append(emb)
        questions.append(q)
        stages.append(item.get("stage", ""))
        frameworks.append(item.get("framework", ""))
    except Exception as e:
        print(f"  Error on question {i}: {e}")
        continue

print(f"\nSuccessfully embedded {len(embeddings)} questions.")

emb_matrix = np.array(embeddings).astype("float32")
print(f"Embedding matrix shape: {emb_matrix.shape}")
print(f"FAISS index dimension: {dim}")

if emb_matrix.shape[1] != dim:
    raise ValueError(f"Embedding dimension mismatch: got {emb_matrix.shape[1]}, expected {dim}")

index.add(emb_matrix)

# Save index + metadata
faiss.write_index(index, INDEX_FILE)

meta = {"questions": questions, "stages": stages, "frameworks": frameworks}
with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(meta, f)

print(f"\n✅ FAISS index built with {len(questions)} questions")
print(f"   Index saved to: {INDEX_FILE}")
print(f"   Metadata saved to: {META_FILE}")

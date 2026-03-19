import requests
from sentence_transformers import SentenceTransformer

# 1. Initialize the embedding model
print("Loading machine learning embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Define the Endee.io API endpoint (default local port is 8080)
ENDEE_API_URL = "http://localhost:8080/api/v1"

# 3. Sample Data
documents = [
    {"id": "1", "text": "Machine learning models require large datasets to train effectively."},
    {"id": "2", "text": "Goa is located on the southwestern coast of India."},
    {"id": "3", "text": "Vector databases like Endee are engineered for ultra-fast semantic search."}
]

# 4. Generate Embeddings and Insert into Endee
print("Generating embeddings...")
for doc in documents:
    # Convert text into a dense vector representation
    vector = model.encode(doc["text"]).tolist()
    
    # Payload structured for a standard vector DB REST API
    payload = {
        "id": doc["id"],
        "vector": vector,
        "metadata": {"text": doc["text"]}
    }
    
    # POST request to insert the document 
    # requests.post(f"{ENDEE_API_URL}/collections/my_data/insert", json=payload)
    print(f"Prepared document {doc['id']} for insertion.")

# 5. Perform a Similarity Search
query = "Where is Goa?"
query_vector = model.encode(query).tolist()

search_payload = {
    "vector": query_vector,
    "top_k": 1
}

print(f"\nSearching for: '{query}'")
# response = requests.post(f"{ENDEE_API_URL}/collections/my_data/search", json=search_payload)
# if response.status_code == 200:
#     print("Top result:", response.json())
print("Search pipeline complete! (Uncomment the API calls once your Endee collection is created).")
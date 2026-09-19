# It is Use For Data Injection Logic

import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pinecone_api_key = os.getenv(key="PINECONE_KEY")
ollama = os.getenv(key="OLLAMA_URL")
embedding_model = os.getenv(key="EMBEDDING_MODEL")

"""
When you run Ollama on your AWS EC2 VM (http://vmip:11434), the server is listening, 
but the actual API endpoints are structured like this:

http://vmip:11434/v1/models → list models

http://vmip:11434/v1/embeddings → generate embeddings

http://vmip:11434/v1/chat/completions → chat completions

------------------------------

🚀 Summary
*) base_url must end with /v1 because Ollama’s API follows the same structure as OpenAI’s.
*) Without /v1, the client points to the wrong path and fails.
*) The dummy api_key is only there to satisfy the OpenAI client library.

"""
# Create OpenAi Client
ollama_client = OpenAI(
    base_url=f"{ollama}/v1",   # Ollama requires /v1
    api_key="ollama"           # dummy string, Ollama ignores it
)


# Initialize Pinecone client
pinecone = Pinecone(api_key=pinecone_api_key)

index_name = "rag-test-index"
# dimensions = 1536 # 1563 work for text-embedding-3-small with openai key
dimensions=768    # work for nomic-embed-text

# =====================================
# create index if its not at pinecone
# ======================================
if not pinecone.has_index(index_name):
    pinecone.create_index(
        name=index_name,
        dimension=dimensions,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"

        )
    )
else:
    print("Index already exists, skipping creation.")

# =====================================
# connect with index at pinecone
# ======================================
index = pinecone.Index(index_name)

# =====================================
# Read File and embeddings
# ======================================

with open("../documents/docs.txt", "r", encoding="utf-8") as f:
    documents = f.read()

    # Use For Store Embedding Data
    vectors = []

    # split docs into chunks
    chunks = documents.split("\n\n")

    # Generate Embedding And Store At Pinecone
    for index_number, chunk in enumerate(chunks):
        response = ollama_client.embeddings.create(
            model=embedding_model, # ✅ Ollama-supported embedding model
            input=chunk,
        )
        embedding = response.data[0].embedding

        """
        
        Pinecome Terminology are like below :-
        ---------------------------------------
        index = store and search embedding data
        id = just like unique id for each chunk unniqueness
        values = store embedding data as list
        metadata= for filtering data #metadata =True
        namespace=  same like package in boot 
        
        """

        # Create Pinecone Vector
        vectors.append({
            "id": f"chunk-{index_number}",
            "values": embedding,
            "metadata": {
                "text": chunk
            }
        })

        # upload vectors to pinecone
        index.upsert(vectors=vectors)

print("Document successfully stored in Pinecone.")

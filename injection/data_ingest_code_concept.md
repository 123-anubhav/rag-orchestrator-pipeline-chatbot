Ollama client:
You’re using

python
ollama_client = OpenAI(
    base_url=f"{ollama}/v1",
    api_key="ollama"
)
which is the right way to connect to Ollama’s API. Without /v1, the client would hit the wrong endpoint and fail.

Pinecone client:
Correctly initialized with

python
pinecone = Pinecone(api_key=pinecone_api_key)
Index creation:
Using dimension=dimensions (spelling fixed) and ServerlessSpec is valid.

Vector schema:
You’re now using "values": embedding instead of "value", which matches Pinecone’s expected format.

Upsert logic:
You collect all vectors in a list and then call index.upsert(vectors=vectors) once outside the loop. That’s efficient and correct.

File reading and chunking:
Splitting on "\n\n" is fine for simple text chunking. Later you might want to add smarter chunking (e.g., by sentence length or token count), but this works for now.

Print confirmation:
Prints once after upsert, so you won’t get spammy output.

🚀 What you have now
Reads your docs.txt file.

Splits into chunks.

Generates embeddings with Ollama (text-embedding-3-small).

Stores them in Pinecone with metadata.

That’s a complete data injection pipeline.
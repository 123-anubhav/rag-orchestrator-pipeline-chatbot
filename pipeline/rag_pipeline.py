import os
from importlib.metadata import metadata

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# load_dotenv(dotenv_path="../.env")   # adjust path if rag_pipeline.py is in a subfolder
load_dotenv()

pinecone_api_key = os.getenv(key="PINECONE_KEY")
ollama = os.getenv(key="OLLAMA_URL")
embedding_model = os.getenv(key="EMBEDDING_MODEL")

# print("pinecone : ", pinecone_api_key)  # if pinecone :  None comes means token expire or something

# Initialize Pinecone client
pinecone = Pinecone(api_key=pinecone_api_key)

# Create OpenAi Client
ollama_client = OpenAI(
    base_url=f"{ollama}/v1",  # Ollama requires /v1
    api_key="ollama"  # dummy string, Ollama ignores it
)

# ===============================
# Pinecone Index Configuration
# ===============================
index_name = "rag-test-index"
dimensions = 768
index = pinecone.Index(index_name)

MODEL = "llama3.2"


def ask_questions(query):
    response = ollama_client.embeddings.create(
        model=embedding_model,
        input=query
    )
    query_embedding = response.data[0].embedding

    # search in pinecone db
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    # print("results : ", results)

    # results is a QueryResponse
    # print(type(results))
    # <class 'pinecone.core.client.model.query_response.QueryResponse'>

    # results.matches is a list of ScoredVector objects
    # for match in results.matches:
        # print(match.id, match.score, match.metadata["text"])

    # Get Retrieved Documents

    """
    Pinecone’s index.query(...), the client returns a QueryResponse object. That object has a property called .matches, which is a list of ScoredVector objects. Each ScoredVector represents one of the top‑k vectors Pinecone found that are similar to your query embedding.
    """
    documents = [match.metadata["text"] for match in results.matches]
    # print("documents:", documents)

    # now Retrieval part is done augmented logics below

    # combine documents into context
    context = "\n\n".join(documents)
    print("context:", context)

    # Answer the question using only the context provided below.
    # tell the answer using below context only if not found then use your mind and answer in bullet points if required

    # create Rag Prompt
    prompt = f"""
    You are a RAG assistant.

    Answer the user's question using the provided context.

    Rules:
    1. Use the context as the primary source.
    2. Do not invent information.
    3. If the context does not contain enough information to answer the question,
       clearly say that the information is not available in the provided documents.
    4. Do not use your general knowledge to fill missing factual information.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    try:
        response = ollama_client.responses.create(
            model=MODEL,
            input=prompt
        )
        return response.output_text

    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"


# FOR LOCAL TESTING YOUR LOGICS YOU CAN BELOW TESTING CODE
if __name__ == "__main__":
    query = input("Ask A Question : ")
    answer = ask_questions(query)
    print(answer)

import os

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_KEY")
ollama = os.getenv("OLLAMA_URL")
embedding_model = os.getenv("EMBEDDING_MODEL")


if not pinecone_api_key:
    raise ValueError("PINECONE_KEY is missing in .env")

if not ollama:
    raise ValueError("OLLAMA_URL is missing in .env")

if not embedding_model:
    raise ValueError("EMBEDDING_MODEL is missing in .env")


# ============================================================
# PINECONE CONFIGURATION
# ============================================================

pinecone = Pinecone(
    api_key=pinecone_api_key
)

index_name = "rag-test-index"

index = pinecone.Index(index_name)


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

ollama_client = OpenAI(
    base_url=f"{ollama.rstrip('/')}/v1",
    api_key="ollama"
)

MODEL = "llama3.2"


# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

# Used to decide whether RAG should be used.

RETRIEVAL_THRESHOLD = 0.55


# Used to decide which retrieved documents
# should be included in the context.

CONTEXT_THRESHOLD = 0.40


# Number of documents retrieved from Pinecone.

TOP_K = 5


# ============================================================
# GENERAL LLM ANSWER
# ============================================================

def generate_general_answer(query):

    prompt = f"""
You are a helpful AI assistant.

The application's knowledge base did not contain
sufficiently relevant information to answer the question.

Answer the question using your general pretrained knowledge.

Rules:

1. Do not claim that the answer came from the knowledge base.
2. Do not invent facts.
3. If you are uncertain, clearly say so.
4. For current or rapidly changing information,
   mention that your knowledge may not be current.
5. Give a useful and clear answer.
6. Answer directly.
7. Do not discuss the internal RAG process.

Question:
{query}

Answer:
"""

    try:

        response = ollama_client.responses.create(
            model=MODEL,
            input=prompt
        )

        return response.output_text.strip()

    except Exception as e:

        return f"Error connecting to Ollama: {str(e)}"


# ============================================================
# RAG ANSWER
# ============================================================

def generate_rag_answer(query, context):

    prompt = f"""
You are an intelligent RAG assistant.

Answer the user's question using the provided
knowledge-base context.

Rules:

1. Use the provided context as the primary source.
2. Do not invent information.
3. Do not change the meaning of the context.
4. If the question contains multiple parts, answer
   each part separately.
5. Match each piece of information to the correct
   part of the question.
6. Do not put unrelated information under the wrong
   category.
7. If information is missing, clearly say that it is
   not available in the context.
8. If the context says offices exist in a city but
   does not provide a specific street address,
   do not invent an address.
9. Keep the answer concise and direct.
10. Do not mention Pinecone, embeddings, similarity
    scores, retrieval, or the internal RAG process.
11. Distinguish different company policies correctly.
    For example, notice period and leave entitlement
    are different policies.
12. Only state information that is supported by the
    provided context.

Knowledge Base Context:
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

        return response.output_text.strip()

    except Exception as e:

        return f"Error connecting to Ollama: {str(e)}"


# ============================================================
# MAIN RAG PIPELINE
# ============================================================

def ask_questions(query):

    # --------------------------------------------------------
    # STEP 1: VALIDATE QUERY
    # --------------------------------------------------------

    if not query or not query.strip():

        return {
            "answer": "Please enter a question.",
            "source": "None",
            "score": 0.0,
            "context": ""
        }

    query = query.strip()


    # --------------------------------------------------------
    # STEP 2: CREATE QUERY EMBEDDING
    # --------------------------------------------------------

    try:

        response = ollama_client.embeddings.create(
            model=embedding_model,
            input=query
        )

        query_embedding = response.data[0].embedding

    except Exception as e:

        return {
            "answer": f"Error creating query embedding: {str(e)}",
            "source": "Error",
            "score": 0.0,
            "context": ""
        }


    # --------------------------------------------------------
    # STEP 3: SEARCH PINECONE
    # --------------------------------------------------------

    try:

        results = index.query(
            vector=query_embedding,
            top_k=TOP_K,
            include_metadata=True
        )

    except Exception as e:

        return {
            "answer": f"Error querying Pinecone: {str(e)}",
            "source": "Error",
            "score": 0.0,
            "context": ""
        }


    # --------------------------------------------------------
    # STEP 4: GET MATCHES
    # --------------------------------------------------------

    matches = results.matches


    if not matches:

        print("\nNo Pinecone results found.")
        print("Using general LLM knowledge.")

        answer = generate_general_answer(query)

        return {
            "answer": answer,
            "source": "General LLM",
            "score": 0.0,
            "context": ""
        }


    # --------------------------------------------------------
    # STEP 5: DISPLAY PINECONE RESULTS
    # --------------------------------------------------------

    print("\n========== PINECONE RESULTS ==========")

    for match in matches:

        text = match.metadata.get(
            "text",
            ""
        )

        print(
            f"ID={match.id} | "
            f"Score={match.score:.4f} | "
            f"Text={text[:200]}"
        )

    print("======================================\n")


    # --------------------------------------------------------
    # STEP 6: BEST SCORE
    # --------------------------------------------------------

    best_score = matches[0].score

    print(
        f"Best Pinecone score: {best_score:.4f}"
    )


    # ========================================================
    # CASE 1: RAG
    # ========================================================

    if best_score >= RETRIEVAL_THRESHOLD:

        print(
            f"Score >= {RETRIEVAL_THRESHOLD}"
        )

        print(
            "Relevant retrieval found."
        )

        print(
            "Using RAG."
        )


        # ----------------------------------------------------
        # STEP 7: SELECT CONTEXT DOCUMENTS
        # ----------------------------------------------------

        documents = []

        selected_matches = []

        for match in matches:

            if match.score >= CONTEXT_THRESHOLD:

                text = match.metadata.get(
                    "text",
                    ""
                )

                if text:

                    documents.append(text)

                    selected_matches.append(match)


        # ----------------------------------------------------
        # STEP 8: SAFETY CHECK
        # ----------------------------------------------------

        if not documents:

            print(
                "No usable context documents found."
            )

            print(
                "Using general LLM knowledge."
            )

            answer = generate_general_answer(query)

            return {
                "answer": answer,
                "source": "General LLM",
                "score": best_score,
                "context": ""
            }


        # ----------------------------------------------------
        # STEP 9: BUILD CONTEXT
        # ----------------------------------------------------

        context = "\n\n".join(
            documents
        )


        # ----------------------------------------------------
        # STEP 10: DISPLAY SELECTED CONTEXT
        # ----------------------------------------------------

        print(
            "\n========== SELECTED RAG CONTEXT =========="
        )

        for match in selected_matches:

            print(
                f"\nScore={match.score:.4f}"
            )

            print(
                match.metadata.get(
                    "text",
                    ""
                )
            )

        print(
            "\n===========================================\n"
        )


        # ----------------------------------------------------
        # STEP 11: GENERATE RAG ANSWER
        # ----------------------------------------------------

        answer = generate_rag_answer(
            query=query,
            context=context
        )


        # ----------------------------------------------------
        # STEP 12: RETURN RESULT
        # ----------------------------------------------------

        return {
            "answer": answer,
            "source": "Knowledge Base / RAG",
            "score": best_score,
            "context": context
        }


    # ========================================================
    # CASE 2: GENERAL LLM
    # ========================================================

    else:

        print(
            f"Score < {RETRIEVAL_THRESHOLD}"
        )

        print(
            "No sufficiently relevant knowledge-base "
            "context found."
        )

        print(
            "Using general LLM knowledge."
        )


        # ----------------------------------------------------
        # STEP 7: GENERAL LLM
        # ----------------------------------------------------

        answer = generate_general_answer(
            query
        )


        # ----------------------------------------------------
        # STEP 8: RETURN RESULT
        # ----------------------------------------------------

        return {
            "answer": answer,
            "source": "General LLM",
            "score": best_score,
            "context": ""
        }


# ============================================================
# LOCAL TESTING
# ============================================================

if __name__ == "__main__":

    query = input(
        "Ask A Question : "
    ).strip()


    result = ask_questions(
        query
    )


    print(
        "\n========== FINAL ANSWER =========="
    )

    print(
        result["answer"]
    )

    print(
        "\nSource:",
        result["source"]
    )

    print(
        "Pinecone Best Score:",
        f"{result['score']:.4f}"
    )

    print(
        "=================================="
    )
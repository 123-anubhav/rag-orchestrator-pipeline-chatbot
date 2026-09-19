*** __init__.py use for create a package and you can import anywhere files if files inside folders like frontend,pipeline

RAG WITH OUR DOCS AND PROMPT = "Answer the question using only the context provided below.{CONTEXT}"

#USE CASE :
    Good for:
        company knowledge bases
        private documents
        legal documents
        internal documentation
        enterprise RAG

DISADVANTAGE : is strict to rag docs  fails for RAG + Web Search like i.e. tell dell laptop price

---
RAG + Web Search
---

#Architecture
---
        User
         ↓
        RAG
         ↓
        No relevant document
         ↓
        Web Search
         ↓
        Find current HP laptop prices
         ↓
        LLM summarizes results
---

---
Project flow (Good)
---
                    ┌──────────────────┐
                    │   User Question  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Query Processing │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Vector Retriever │
                    │    Qdrant        │
                    └────────┬─────────┘
                             ↓
                     Relevant Context?
                       /           \
                     YES            NO
                      ↓              ↓
                ┌──────────┐   ┌──────────────┐
                │ RAG LLM  │   │ Fallback     │
                └────┬─────┘   │ Web / LLM    │
                     ↓         └──────┬───────┘
                  Answer              ↓
                                  Answer

#----------------------------------------------------
---
current architecture : rag_with_llm_respond file
---
*** RAG → Pinecone → Llama 3.2 • General LLM fallback enabled

                 USER
                  │
                  ▼
             Query Embedding
                  │
                  ▼
              Pinecone
                  │
                Top 5
                  │
                  ▼
             Best Score
              /       \
        >= 0.55       < 0.55
           │              │
           ▼              ▼
          RAG        General LLM
           │
           ▼
    Context Filtering
           │
           ▼
       Ollama LLM
           │
           ▼
      Final Answer
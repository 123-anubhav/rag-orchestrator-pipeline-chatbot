# 🤖 RAG Orchestrator Pipeline Chatbot

> **An end-to-end Retrieval-Augmented Generation (RAG) application with document ingestion, vector indexing, semantic retrieval, retrieval-confidence routing, context filtering, grounded LLM generation, and general-LLM fallback.**

This project implements a complete RAG workflow for building a knowledge-base chatbot.

The application takes documents, converts their content into searchable vector representations, stores them in **Pinecone**, and uses **Ollama** for embeddings and LLM-based response generation.

At query time, the system evaluates retrieval relevance before deciding whether to answer using the knowledge base or fall back to the LLM's general knowledge.

---

# 🏗️ System Architecture

<p align="center">
  <img src="./docs/images/01-system-architecture.png" alt="System Architecture" width="100%">
</p>

![rag pipeline](./RAG%20Pipeline_%20From%20Documents%20to%20Answers.png)

---

# 🚀 What This Project Does

The application implements two major workflows.

### 1. Document Ingestion

Documents are processed and converted into embeddings before being stored in Pinecone.

<p align="center">
  <img src="./docs/images/02-document-ingestion-flow.png" alt="Document Ingestion Flow" width="100%">
</p>

### 2. Question Answering

A user's question is embedded, searched against the vector database, evaluated for relevance, and then routed to either RAG generation or general LLM generation.

<p align="center">
  <img src="./docs/images/03-question-answering-flow.png" alt="Question Answering Flow" width="100%">
</p>

---

# 📥 Document Ingestion

The `documents/` folder contains the knowledge-base content used by the application.

The `injection/` module is responsible for taking document content and making it available for semantic retrieval.

Conceptually, the ingestion pipeline is:

<p align="center">
  <img src="./docs/images/04-ingestion-conceptual-flow.png" alt="Ingestion Conceptual Flow" width="100%">
</p>

The resulting vector records contain searchable information that can later be retrieved by the RAG pipeline.

---

# 🧠 Embedding Architecture

The project uses:

<p align="center">
  <img src="./docs/images/05-embedding-model.png" alt="Embedding Model" width="100%">
</p>

The same embedding model is used to represent textual information as vectors for semantic similarity search.

### Document Embedding

<p align="center">
  <img src="./docs/images/06-document-embedding.png" alt="Document Embedding" width="100%">
</p>

### Query Embedding

When the user asks a question:

<p align="center">
  <img src="./docs/images/07-query-embedding.png" alt="Query Embedding" width="100%">
</p>

This allows the application to find conceptually similar content rather than relying only on exact keyword matching.

---

# 🌲 Pinecone Vector Database

Pinecone is used as the vector database.

Configured index:

<p align="center">
  <img src="./docs/images/08-pinecone-index.png" alt="Pinecone Index" width="100%">
</p>

The application queries Pinecone using the generated query embedding and retrieves the most relevant records.

The retrieval configuration is:

```python
TOP_K = 5
```

Therefore, the pipeline requests the top five matching records from Pinecone for a query.

---

# 🔍 Query Processing Pipeline

The core query pipeline is implemented around the `ask_questions()` flow.

The processing sequence is:

<p align="center">
  <img src="./docs/images/09-query-processing-sequence.png" alt="Query Processing Sequence" width="100%">
</p>

---

# 🎯 Retrieval Confidence Routing

One of the main features of the implementation is **retrieval-aware routing**.

The application does not blindly send every query through the RAG path.

It evaluates the best Pinecone similarity score first.

```python
RETRIEVAL_THRESHOLD = 0.55
```

The decision is:

<p align="center">
  <img src="./docs/images/10-retrieval-confidence-routing.png" alt="Retrieval Confidence Routing" width="100%">
</p>

The source code explicitly compares the best retrieved score against `RETRIEVAL_THRESHOLD` before selecting the RAG or general-LLM path.

---

# 🧩 Context Filtering

Retrieval decision and context selection are treated separately.

The pipeline uses:

```python
CONTEXT_THRESHOLD = 0.40
```

Once the RAG path is selected, retrieved matches are filtered again.

<p align="center">
  <img src="./docs/images/11-context-filtering.png" alt="Context Filtering" width="100%">
</p>

Only matches meeting the context threshold are added to the context supplied to the LLM.

This creates a two-level retrieval strategy:

<p align="center">
  <img src="./docs/images/12-two-level-retrieval.png" alt="Two-Level Retrieval" width="100%">
</p>

---

# 🧠 RAG Generation

When sufficiently relevant knowledge is found, the selected documents are combined into a context:

<p align="center">
  <img src="./docs/images/13-rag-generation.png" alt="RAG Generation" width="100%">
</p>

The RAG prompt instructs the model to use the supplied knowledge-base context as the primary source and avoid unsupported information.

The prompt also handles cases such as:

* missing information
* multi-part questions
* unrelated information
* different policies or categories
* unsupported addresses or details
* hallucinated facts

The generation layer therefore focuses on producing answers grounded in the retrieved context.

---

# 🔄 General LLM Fallback

If the best Pinecone result does not meet the retrieval threshold, the application does not force the query through the knowledge base.

Instead:

<p align="center">
  <img src="./docs/images/14-general-llm-fallback.png" alt="General LLM Fallback" width="100%">
</p>

The fallback prompt explicitly tells the model that the knowledge base did not contain sufficiently relevant information and instructs it to answer using general pretrained knowledge.

This gives the application two response modes:

<p align="center">
  <img src="./docs/images/15-response-modes.png" alt="Response Modes" width="100%">
</p>

---

# 🛡️ Hallucination-Aware Prompting

The RAG generation prompt contains explicit rules to keep responses grounded in retrieved context.

The model is instructed to:

* use the supplied context as the primary source
* not invent information
* preserve the meaning of the context
* identify missing information
* avoid mixing unrelated information
* distinguish different policies correctly
* avoid exposing internal RAG implementation details

For example, if the retrieved context says that an office exists in a city but does not provide a street address, the model is instructed not to invent an address.

---

# 🦙 Ollama Integration

Ollama provides both the embedding and LLM interaction layer.

The project uses:

<p align="center">
  <img src="./docs/images/16-ollama-models.png" alt="Ollama Models" width="100%">
</p>

The application communicates with Ollama through an OpenAI-compatible client:

<p align="center">
  <img src="./docs/images/17-ollama-integration.png" alt="Ollama Integration" width="100%">
</p>

The source configures the Ollama endpoint using the environment variable `OLLAMA_URL`.

---

# 📊 Structured Pipeline Response

The RAG pipeline returns a structured result rather than only returning a plain string.

Example:

```python
{
    "answer": "...",
    "source": "Knowledge Base / RAG",
    "score": 0.78,
    "context": "..."
}
```

The implementation returns:

<p align="center">
  <img src="./docs/images/18-structured-response.png" alt="Structured Pipeline Response" width="100%">
</p>

The `source` field identifies whether the response came from the knowledge base/RAG path or the general LLM path.

This makes the pipeline easier to integrate with a frontend application.

---

# 🖥️ Frontend

The `frontend/` folder contains the chatbot user-interface layer.

The frontend is separated from the core RAG orchestration so that:

<p align="center">
  <img src="./docs/images/19-frontend-rag-pipeline.png" alt="Frontend RAG Pipeline" width="100%">
</p>

The core retrieval and generation logic therefore remains independent of the presentation layer.

---

# 📁 Project Structure

The repository is organized into separate application responsibilities:

<p align="center">
  <img src="./docs/images/20-project-structure.png" alt="Project Structure" width="100%">
</p>

These top-level folders/files are present in the repository itself.

### Responsibility of Each Folder

| Folder       | Responsibility                                      |
| ------------ | --------------------------------------------------- |
| `documents/` | Knowledge-base source documents                     |
| `injection/` | Document ingestion and vector indexing              |
| `pipeline/`  | Query processing, retrieval, routing and generation |
| `frontend/`  | User-facing chatbot interface                       |
| `.idea/`     | IDE/project configuration                           |

---

# 🛠️ Technology Stack

| Technology            | Purpose                                |
| --------------------- | -------------------------------------- |
| **Python**            | Application and RAG orchestration      |
| **Ollama**            | Local LLM and embedding inference      |
| **Llama 3.2**         | LLM response generation                |
| **nomic-embed-text**  | Text embedding generation              |
| **Pinecone**          | Vector database and semantic search    |
| **OpenAI Python SDK** | OpenAI-compatible interface for Ollama |
| **python-dotenv**     | Environment configuration              |
| **Streamlit**         | Frontend/UI layer                      |

---

# ⚙️ Configuration

The application reads configuration from environment variables.

Example:

```env
PINECONE_KEY=your_pinecone_api_key

OLLAMA_URL=http://localhost:11434

EMBEDDING_MODEL=nomic-embed-text
```

The code validates that the required configuration exists before initializing the Pinecone and Ollama components.

> **Security:** Never commit real API keys or credentials to GitHub. Keep `.env` local and use an example configuration file for documentation.

---

# 🚀 Setup

## 1. Clone the Repository

```bash
git clone https://github.com/123-anubhav/rag-orchestrator-pipeline-chatbot.git

cd rag-orchestrator-pipeline-chatbot
```

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Environment

Create a local `.env`:

```env
PINECONE_KEY=your_pinecone_api_key
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
```

---

# 🦙 Ollama Setup

Install and start Ollama, then pull the required models:

```bash
ollama pull llama3.2
```

```bash
ollama pull nomic-embed-text
```

Verify:

```bash
ollama list
```

The application expects the Ollama endpoint configured through:

```env
OLLAMA_URL=http://localhost:11434
```

---

# 🌲 Pinecone Setup

Create/configure the Pinecone index used by the application:

<p align="center">
  <img src="./docs/images/21-pinecone-index-config.png" alt="Pinecone Index Configuration" width="100%">
</p>

The embedding configuration used during ingestion must match the embedding configuration used when querying the index.

The query pipeline connects to the configured Pinecone index and performs vector similarity search with:

```python
TOP_K = 5
```

---

# 📥 Document Indexing Workflow

Place the knowledge-base files in:

<p align="center">
  <img src="./docs/images/22-documents-directory.png" alt="Documents Directory" width="100%">
</p>

Then use the document ingestion functionality under:

<p align="center">
  <img src="./docs/images/23-injection-directory.png" alt="Injection Directory" width="100%">
</p>

The conceptual indexing workflow is:

<p align="center">
  <img src="./docs/images/24-document-indexing-workflow.png" alt="Document Indexing Workflow" width="100%">
</p>

After indexing, the RAG pipeline can query those vectors.

> The repository's top-level `injection/` directory is confirmed, but the currently accessible GitHub view does not expose the exact script filenames inside it. Therefore this README intentionally avoids inventing a filename.

---

# ▶️ Running the RAG Pipeline

The query-side pipeline performs:

<p align="center">
  <img src="./docs/images/25-running-rag-pipeline.png" alt="Running the RAG Pipeline" width="100%">
</p>

The source also contains local CLI testing through:

```python
if __name__ == "__main__":
```

and accepts:

<p align="center">
  <img src="./docs/images/26-cli-question-prompt.png" alt="CLI Question Prompt" width="100%">
</p>

before executing the pipeline and displaying the final answer, source and Pinecone score.

---

# 💬 Example

### Knowledge-base question

<p align="center">
  <img src="./docs/images/27-example-question.png" alt="Example Question" width="100%">
</p>

If the best Pinecone score is:

<p align="center">
  <img src="./docs/images/28-example-high-score.png" alt="Example High Score" width="100%">
</p>

then:

<p align="center">
  <img src="./docs/images/29-high-score-decision.png" alt="High Score Decision" width="100%">
</p>

so the RAG path is selected.

<p align="center">
  <img src="./docs/images/30-high-score-rag-path.png" alt="High Score RAG Path" width="100%">
</p>

### Question outside the knowledge base

If the best score is:

<p align="center">
  <img src="./docs/images/31-example-low-score.png" alt="Example Low Score" width="100%">
</p>

then:

<p align="center">
  <img src="./docs/images/32-low-score-decision.png" alt="Low Score Decision" width="100%">
</p>

and the system uses the general LLM path.

<p align="center">
  <img src="./docs/images/33-low-score-general-llm.png" alt="Low Score General LLM" width="100%">
</p>

---

# 📐 Retrieval Configuration

The current pipeline uses:

| Configuration         |              Value | Purpose                                        |
| --------------------- | -----------------: | ---------------------------------------------- |
| `TOP_K`               |                `5` | Number of Pinecone matches retrieved           |
| `RETRIEVAL_THRESHOLD` |             `0.55` | Determines whether RAG is selected             |
| `CONTEXT_THRESHOLD`   |             `0.40` | Determines which matches enter the RAG context |
| `MODEL`               |         `llama3.2` | Response generation model                      |
| `EMBEDDING_MODEL`     | `nomic-embed-text` | Query/document embedding model                 |
| Pinecone index        |   `rag-test-index` | Vector storage/search                          |

These values are defined directly in the supplied pipeline implementation.

---

# 🔬 Core Engineering Decisions

## 1. Separate Retrieval Decision from Context Selection

The application does not use one threshold for everything.

<p align="center">
  <img src="./docs/images/34-threshold-separation.png" alt="Threshold Separation" width="100%">
</p>

This provides more control over the quality of context supplied to the LLM.

---

## 2. Retrieval-Aware Fallback

The application does not assume that every query belongs to the knowledge base.

<p align="center">
  <img src="./docs/images/35-retrieval-aware-fallback.png" alt="Retrieval-Aware Fallback" width="100%">
</p>

---

## 3. Grounded Generation

The RAG prompt explicitly tells the LLM to use the retrieved context and avoid unsupported information.

---

## 4. Separation of Responsibilities

The repository separates:

<p align="center">
  <img src="./docs/images/36-separation-responsibilities.png" alt="Separation of Responsibilities" width="100%">
</p>

This keeps ingestion, retrieval, generation and presentation as separate application concerns.

---

## 5. Structured Output

The pipeline returns more than the answer:

<p align="center">
  <img src="./docs/images/37-structured-output.png" alt="Structured Output" width="100%">
</p>

This allows the frontend to understand how the answer was generated.

---

# 🔐 Security

The project uses environment variables for sensitive configuration:

```env
PINECONE_KEY=...
OLLAMA_URL=...
EMBEDDING_MODEL=...
```

Do not commit secrets to source control.

Recommended repository practice:

<p align="center">
  <img src="./docs/images/38-security-flow.png" alt="Security Flow" width="100%">
</p>

Use a sanitized example configuration for GitHub documentation.

---

# 📌 End-to-End Project Flow

The complete application can be viewed as two connected pipelines.

## Indexing Pipeline

<p align="center">
  <img src="./docs/images/39-end-to-end-indexing.png" alt="End-to-End Indexing" width="100%">
</p>

## Query Pipeline

<p align="center">
  <img src="./docs/images/40-end-to-end-query.png" alt="End-to-End Query" width="100%">
</p>

---

# 🎯 What This Project Demonstrates

This implementation demonstrates practical experience with:

* Retrieval-Augmented Generation
* Vector databases
* Semantic similarity search
* Embedding generation
* Document ingestion
* Vector indexing
* Top-K retrieval
* Retrieval confidence thresholds
* Context filtering
* Grounded prompt engineering
* LLM fallback strategies
* Local LLM inference with Ollama
* Pinecone integration
* Environment-based configuration
* Python-based GenAI application architecture
* Frontend integration with a RAG backend

---

# 👨‍💻 Author

**Anubhav Srivastava**

**Java Full Stack Developer | GenAI Developer**

GitHub:  
https://github.com/123-anubhav

Project:  
https://github.com/123-anubhav/rag-orchestrator-pipeline-chatbot

---

# ⭐ Project Summary

**RAG Orchestrator Pipeline Chatbot** implements a complete retrieval-aware GenAI workflow:

<p align="center">
  <img src="./docs/images/41-project-summary.png" alt="Project Summary" width="100%">
</p>

> **A complete practical RAG implementation combining document ingestion, embeddings, Pinecone vector search, retrieval-aware routing, context filtering, grounded generation, and LLM fallback using Python and Ollama.**

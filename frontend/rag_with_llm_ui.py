import streamlit as st
import sys
import os


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from pipeline.rag_with_llm_respond import ask_questions


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI RAG Assistant",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}


/* ================= HERO ================= */

.hero {
    padding: 1.8rem 2rem;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #667eea 0%,
        #764ba2 100%
    );
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 25px rgba(0,0,0,.12);
}

.hero h1 {
    margin: 0;
    font-size: 2.1rem;
}

.hero p {
    margin: .5rem 0 0;
    opacity: .9;
    font-size: 1rem;
}


/* ================= ANSWER ================= */

.answer-box {
    padding: 1.2rem;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    background: #ffffff;
    margin-top: .5rem;
    margin-bottom: .8rem;
}


/* ================= SOURCE BADGES ================= */

.rag-badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 20px;
    background: #dcfce7;
    color: #166534;
    font-size: 13px;
    font-weight: 600;
}

.general-badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 20px;
    background: #dbeafe;
    color: #1e40af;
    font-size: 13px;
    font-weight: 600;
}


/* ================= CONTEXT ================= */

.context-text {
    font-size: 14px;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

    <h1>🤖 AI RAG Assistant</h1>

    <p>
        Ask questions from your knowledge base —
        with General LLM fallback when no relevant
        document is found.
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.success("🟢 Pinecone Connected")

    st.info("🧠 LLM: Llama 3.2")

    st.info("📚 Vector DB: Pinecone")

    st.info("🔄 RAG + General LLM Fallback")

    st.divider()

    st.caption("Retrieval threshold: 0.55")

    st.caption("Context threshold: 0.40")

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# CHAT STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS CHAT
# ============================================================

for message in st.session_state.messages:

    # --------------------------------------------------------
    # USER
    # --------------------------------------------------------

    if message["role"] == "user":

        with st.chat_message("user"):

            st.markdown(
                message["content"]
            )

    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    else:

        with st.chat_message("assistant"):

            # Answer
            st.markdown("### 🤖 AI Answer")

            st.markdown(
                message["answer"]
            )

            st.divider()

            # Source
            source = message.get(
                "source",
                "Unknown"
            )

            if source == "Knowledge Base / RAG":

                st.success(
                    "📚 Source: Knowledge Base / RAG"
                )

            elif source == "General LLM":

                st.info(
                    "🧠 Source: General LLM"
                )

            else:

                st.warning(
                    f"Source: {source}"
                )


            # Score
            score = message.get(
                "score",
                0.0
            )

            st.caption(
                f"🔢 Pinecone Best Score: {score:.4f}"
            )


            # Retrieved context
            context = message.get(
                "context",
                ""
            )

            if (
                source == "Knowledge Base / RAG"
                and context
            ):

                with st.expander(
                    "📚 View Retrieved Knowledge"
                ):

                    st.markdown(
                        context
                    )


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask something..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if query:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role": "user",

        "content": query

    })


    with st.chat_message("user"):

        st.markdown(query)


    # --------------------------------------------------------
    # ASSISTANT MESSAGE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching knowledge base and thinking..."
        ):

            try:

                result = ask_questions(query)


                # ====================================================
                # RESULT FROM BACKEND
                # ====================================================

                answer = result.get(
                    "answer",
                    "No answer returned."
                )

                source = result.get(
                    "source",
                    "Unknown"
                )

                score = result.get(
                    "score",
                    0.0
                )

                context = result.get(
                    "context",
                    ""
                )


            except Exception as e:

                answer = f"❌ Error: {str(e)}"

                source = "Error"

                score = 0.0

                context = ""


        # ====================================================
        # AI ANSWER
        # ====================================================

        st.markdown("### 🤖 AI Answer")

        st.markdown(answer)


        # ====================================================
        # SOURCE
        # ====================================================

        st.divider()

        if source == "Knowledge Base / RAG":

            st.success(
                "📚 Source: Knowledge Base / RAG"
            )

        elif source == "General LLM":

            st.info(
                "🧠 Source: General LLM"
            )

        else:

            st.warning(
                f"Source: {source}"
            )


        # ====================================================
        # SCORE
        # ====================================================

        st.caption(
            f"🔢 Pinecone Best Score: {score:.4f}"
        )


        # ====================================================
        # RAG CONTEXT
        # ====================================================

        if (
            source == "Knowledge Base / RAG"
            and context
        ):

            with st.expander(
                "📚 View Retrieved Knowledge"
            ):

                st.markdown(context)


    # ========================================================
    # SAVE ASSISTANT MESSAGE
    # ========================================================

    st.session_state.messages.append({

        "role": "assistant",

        "answer": answer,

        "source": source,

        "score": score,

        "context": context

    })


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div style="
    text-align:center;
    color:#9ca3af;
    margin-top:2rem;
    font-size:.85rem;
">
    RAG → Pinecone → Llama 3.2
    • General LLM fallback enabled
</div>
""", unsafe_allow_html=True)
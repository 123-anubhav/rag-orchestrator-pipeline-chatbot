import os
import streamlit as st

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from  pipeline.rag_pipeline import ask_questions

st.title("RAG Pipeline Demo")

query = st.text_input("Enter your question:")

# Add a button
if st.button("Ask"):
    if query:
        answer= ask_questions(query)
        st.write("### Answer")
        st.write(answer)

        """        st.write("### Retrieved Context")
        for doc in answer:
        st.write(doc)
        """
    else:
        st.warning("Please enter a question before clicking Ask.")

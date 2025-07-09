import streamlit as st
from qdrant_client import QdrantClient

from retrieval import rag_pipeline

from core.config import config

qdrant_client = QdrantClient(
    url=f"http://{config.QDRANT_URL}:6333"
)


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hello! How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        output = rag_pipeline(prompt, qdrant_client)
        # output = run_llm(client, st.session_state.messages, st.session_state.temperature, 
        #                  st.session_state.max_tokens)
        st.write(output['answer'].answer)
    st.session_state.messages.append({"role": "assistant", "content": output})

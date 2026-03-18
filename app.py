import streamlit as st
from retrieval import RAGPipeline

st.set_page_config(page_title="AI Assistant", page_icon="🤖")

st.title("🤖 AI Research Assistant")

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

rag_pipeline = load_pipeline()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
query = st.chat_input("Ask something...")

if query:
    # User message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_pipeline.run_rag(query)

            st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

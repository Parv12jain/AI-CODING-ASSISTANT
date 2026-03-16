import streamlit as st
from retrieval import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize RAG
rag = RAGPipeline()

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("### Retrieval Settings")
    k = st.slider("Number of documents to retrieve", 1, 10, 5)

    st.markdown("---")
    st.markdown("### About")
    st.write(
        "This AI assistant answers questions using your research papers "
        "through Retrieval-Augmented Generation (RAG)."
    )

# Main Title
st.title("🤖 AI Research Assistant")
st.markdown("Ask questions about your research papers and get AI-powered answers.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
query = st.chat_input("Ask something about your research papers...")

if query:

    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag.run_rag(query, k=k)

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
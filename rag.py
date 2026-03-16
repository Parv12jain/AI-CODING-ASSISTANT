import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

def load_documents(data_path="Research_paper"):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    if len(documents) == 0:
        print("No documents found")
    print(f"Loaded {len(documents)} documents.")

    return documents

# SPLIT DOCUMENTS
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(chunks)} chunks from {len(documents)} documents.")
    return chunks

# CREATE EMBEDDINGS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def create_vector_store(chunks, persist_directory="db/Chroma_db"):
    print("Creating embeddings and storing in ChromaDB...")

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )

    print("Vector store created successfully")

    return vector_store

if __name__ == "__main__":

    print("Starting RAG pipeline...")

    documents = load_documents()

    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks)

    print("RAG pipeline completed")


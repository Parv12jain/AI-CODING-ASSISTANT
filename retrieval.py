from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()


class RAGPipeline:
    def __init__(self, persist_directory="db/Chroma_db"):
        self.persist_directory = persist_directory

        # Embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load vector store
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )

        # API key check
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in .env")

        # Gemini LLM
        self.llm = ChatMistralAI(
            model="mistral-small",
            temperature=0.3
        )

    # ✅ OUTSIDE __init__
    def retrieve(self, query, k):
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        return retriever.invoke(query)

    def generate_response(self, query, retrieved_docs):
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""
You are an AI research assistant.

Use ONLY the information from the provided context to answer the question.

Guidelines:
- Give a clear explanation in 3–5 sentences.
- Do not add information that is not in the context.
- If the answer is not in the context, say: "The context does not contain the answer."

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke(prompt)

        return getattr(response, "content", str(response))

    def run_rag(self, query, k=5):
        retrieved_docs = self.retrieve(query, k)

        response = self.generate_response(query, retrieved_docs)

        print("\nSources:")
        for doc in retrieved_docs:
            print(doc.metadata.get("source"))

        return response


if __name__ == "__main__":
    rag_pipeline = RAGPipeline()

    query = "What is BERT?"
    response = rag_pipeline.run_rag(query)

    print("\nAnswer:\n")
    print(response)

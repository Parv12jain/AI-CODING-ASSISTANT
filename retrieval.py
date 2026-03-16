from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

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
            persist_directory=persist_directory,
            embedding_function=self.embedding_model
        )

        # Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3
        )

    def retrieve(self, query, k=3):
        return self.vector_store.similarity_search(query, k=k)

    def generate_response(self, query, retrieved_docs):
        """Generate response using retrieved documents"""

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

        return response.content

    def run_rag(self, query, k=5):
        """Run the complete RAG pipeline"""
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
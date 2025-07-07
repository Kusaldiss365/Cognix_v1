from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from utils.openai_config import get_openai_llm

class ContextAgent:
    def __init__(self, material_pdf_path, persist_directory):
        self.material_pdf_path = material_pdf_path
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.llm = get_openai_llm()
        self.embedding = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")

    def ingest_and_index(self):
        loader = PyPDFLoader(self.material_pdf_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(pages)

        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )

    def get_vectorstore(self):
        if not self.vectorstore:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding
            )
        # print("Vector store output: \n",self.vectorstore)
        return self.vectorstore

    def retrieve_context(self, query, top_k=3):
        vectorstore = self.get_vectorstore()
        return vectorstore.similarity_search(query, k=top_k)

    def generate_answer(self, question, context_docs):
        # Combine context documents into a single string
        context_text = "\n\n".join(doc.page_content for doc in context_docs)

        # Prepare a prompt for the LLM that includes the question and context
        prompt = (
            f"You are a helpful assistant. Use the following context to answer the question as accurately as possible.\n\n"
            f"{context_text}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        # Invoke the LLM to get the answer
        answer = self.llm.invoke(prompt)

        return answer

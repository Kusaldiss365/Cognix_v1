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

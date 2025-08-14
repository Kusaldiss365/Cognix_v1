from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from utils.openai_config import get_openai_llm
import re


class ContextAgent:
    def __init__(self, material_pdf_path, persist_directory):
        self.material_pdf_path = material_pdf_path
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.llm = get_openai_llm()
        self.embedding = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")

    def _guess_page_title(self, page_text: str) -> str:
        """
        Heuristic: first strong-looking line (numbered heading, ALL CAPS line,
        or a short Title Case line). Falls back to first non-empty line.
        """
        lines = [l.strip() for l in page_text.splitlines() if l.strip()]
        if not lines:
            return "Untitled"
        # numbered heading like "3.1 Activity Lifecycle"
        for l in lines[:10]:
            if re.match(r"^\d+(\.\d+)*\s+[A-Za-z].{2,}", l):
                return l[:120]
        # ALL CAPS short line
        for l in lines[:10]:
            if len(l) <= 80 and re.match(r"^[A-Z0-9 ,\-:()]+$", l) and any(c.isalpha() for c in l):
                return l[:120]
        # Title Case short line
        for l in lines[:10]:
            if len(l) <= 100 and sum(w[:1].isupper() for w in l.split()) >= max(2, len(l.split()) // 2):
                return l[:120]
        # fallback: first non-empty
        return lines[0][:120]

    def ingest_and_index(self):
        loader = PyPDFLoader(self.material_pdf_path)
        pages = loader.load()  # each page has metadata={"source": "...", "page": <0-index>}

        # Pre-compute a title per page
        page_titles = {}
        for p in pages:
            p_idx = p.metadata.get("page", None)
            page_titles[p_idx] = self._guess_page_title(p.page_content)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(pages)

        # Enrich chunk metadata with page and page_title (preserve any existing metadata)
        for d in docs:
            p_idx = d.metadata.get("page", None)
            d.metadata["page"] = p_idx
            d.metadata["page_title"] = page_titles.get(p_idx, "Untitled")

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
        return self.vectorstore

    def retrieve_context(self, query, top_k=3):
        vectorstore = self.get_vectorstore()
        hits = vectorstore.similarity_search(query, k=top_k)
        # Optional: quick debug print to verify correctness
        # for h in hits:
        #     print(f"hit: p.{(h.metadata.get('page') or 0)+1} | {h.metadata.get('page_title')}")
        return hits

    def format_hits_with_citations(self, hits):
        """
        Formats context for prompts/UI with explicit page + title labels.
        Useful to feed into Evaluation/Reflection prompts so the model stops guessing.
        """
        blocks = []
        for h in hits:
            p = h.metadata.get("page")
            t = h.metadata.get("page_title", "Untitled")
            label = f"[p.{(p or 0) + 1} | {t}]"
            blocks.append(f"{label}\n{h.page_content}")
        return "\n\n".join(blocks)

import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class ResumeRAG:
    def __init__(self, persist_dir: str = "chroma_resumes"):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.persist_dir = persist_dir
        self.vector_store = None

    def load_resumes(self, resume_dir: str) -> List[Document]:
        """Load all PDF and text resumes from directory"""
        docs = []
        print(f"Loading resumes from: {resume_dir}")
        if not os.path.exists(resume_dir):
            print(f"Directory does not exist: {resume_dir}")
            return docs
        for file in os.listdir(resume_dir):
            path = os.path.join(resume_dir, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
                resume_docs = loader.load()
                for doc in resume_docs:
                    doc.metadata["source"] = file
                    doc.metadata["type"] = "resume"
                docs.extend(resume_docs)
                print(f"Loaded PDF: {file} ({len(resume_docs)} pages)")
            elif file.endswith(".txt"):
                with open(path, 'r') as f:
                    content = f.read()
                    docs.append(Document(page_content=content, metadata={"source": file, "type": "resume"}))
                    print(f"Loaded TXT: {file} ({len(content)} chars)")
        print(f"Total documents loaded: {len(docs)}")
        return docs

    def load_jd(self, jd_path: str) -> Document:
        """Load job description from PDF or text file"""
        if jd_path.endswith(".pdf"):
            loader = PyPDFLoader(jd_path)
            docs = loader.load()
            return docs[0] if docs else Document(page_content="")
        else:
            with open(jd_path, 'r') as f:
                return Document(page_content=f.read(), metadata={"type": "jd"})

    def create_vector_store(self, docs: List[Document]):
        """Create and persist vector store"""
        if os.path.exists(self.persist_dir):
            import shutil
            shutil.rmtree(self.persist_dir)

        chunks = self.text_splitter.split_documents(docs)
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=self.persist_dir
        )
        return self.vector_store

    def retrieve_relevant(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve top-k relevant resume chunks"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        return self.vector_store.similarity_search(query, k=k)

    def load_and_index(self, resume_dir: str):
        """Load resumes from directory and create vector store"""
        docs = self.load_resumes(resume_dir)
        if docs:
            self.create_vector_store(docs)
        return self

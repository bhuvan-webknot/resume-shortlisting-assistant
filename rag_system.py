import os
import logging
from typing import List, Optional
from collections import Counter
import math

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


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
        self._all_chunks: List[Document] = []
        self._cross_encoder = None

    def load_resumes(self, resume_dir: str) -> List[Document]:
        docs = []
        logger.info(f"Loading resumes from: {resume_dir}")
        if not os.path.exists(resume_dir):
            logger.warning(f"Directory does not exist: {resume_dir}")
            return docs
        for file in sorted(os.listdir(resume_dir)):
            path = os.path.join(resume_dir, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
                resume_docs = loader.load()
                for doc in resume_docs:
                    doc.metadata["source"] = file
                    doc.metadata["type"] = "resume"
                docs.extend(resume_docs)
                logger.info(f"Loaded PDF: {file} ({len(resume_docs)} pages)")
            elif file.endswith(".txt"):
                with open(path, 'r') as f:
                    content = f.read()
                    docs.append(Document(page_content=content, metadata={"source": file, "type": "resume"}))
                    logger.info(f"Loaded TXT: {file} ({len(content)} chars)")
        logger.info(f"Total documents loaded: {len(docs)}")
        return docs

    def create_vector_store(self, docs: List[Document]):
        chunks = self.text_splitter.split_documents(docs)
        self._all_chunks = chunks
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
        )
        logger.info(f"Created in-memory vector store with {len(chunks)} chunks")
        return self.vector_store

    def _compute_keyword_scores(self, query: str, docs: List[Document]) -> List[float]:
        query_terms = set(query.lower().split())
        if not query_terms:
            return [0.0] * len(docs)

        doc_term_counts = []
        for d in docs:
            terms = d.page_content.lower().split()
            doc_term_counts.append(Counter(terms))

        idf = {}
        n_docs = len(docs)
        for term in query_terms:
            df = sum(1 for c in doc_term_counts if term in c)
            idf[term] = math.log((n_docs + 1) / (df + 1)) + 1

        avg_dl = sum(len(d.page_content.split()) for d in docs) / max(len(docs), 1)

        scores = []
        for i, d in enumerate(docs):
            score = 0.0
            doc_len = len(d.page_content.split())
            for term in query_terms:
                tf = doc_term_counts[i].get(term, 0)
                bm25_tf = tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * doc_len / max(avg_dl, 1)))
                score += bm25_tf * idf.get(term, 1)
            scores.append(score)

        max_score = max(scores) if scores else 1.0
        return [s / max_score if max_score > 0 else 0.0 for s in scores]

    def _load_cross_encoder(self):
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                logger.info("Loaded cross-encoder reranker")
            except Exception as e:
                logger.warning(f"Could not load cross-encoder: {e}")
        return self._cross_encoder

    def retrieve_relevant(self, query: str, k: int = 5, use_hybrid: bool = True,
                          rerank: bool = False) -> List[Document]:
        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        embedding_k = k * 3 if (use_hybrid or rerank) else k
        embedding_results = self.vector_store.similarity_search_with_relevance_scores(query, k=embedding_k)
        emb_docs = [r[0] for r in embedding_results]
        raw_scores = [r[1] if isinstance(r[1], (int, float)) else 0.0 for r in embedding_results]

        emb_scores = []
        for s in raw_scores:
            clamped = max(0.0, min(1.0, s))
            emb_scores.append(clamped)

        if use_hybrid and self._all_chunks:
            keyword_scores = self._compute_keyword_scores(query, emb_docs)
            alpha = 0.7
            combined = [
                (d, alpha * emb_scores[i] + (1 - alpha) * keyword_scores[i])
                for i, d in enumerate(emb_docs)
            ]
            combined.sort(key=lambda x: x[1], reverse=True)
            emb_docs = [c[0] for c in combined[:k]]
            emb_scores = [c[1] for c in combined[:k]]
        else:
            emb_docs = emb_docs[:k]
            emb_scores = emb_scores[:k]

        if rerank and len(emb_docs) > 1:
            ce = self._load_cross_encoder()
            if ce:
                try:
                    pairs = [(query, d.page_content) for d in emb_docs]
                    rerank_scores = ce.predict(pairs)
                    reranked = sorted(zip(emb_docs, rerank_scores), key=lambda x: x[1], reverse=True)
                    emb_docs = [r[0] for r in reranked]
                    emb_scores = [float(r[1]) for r in reranked]
                    logger.info(f"Reranked {len(emb_docs)} results")
                except Exception as e:
                    logger.warning(f"Reranking failed: {e}")

        for i, d in enumerate(emb_docs):
            d.metadata["score"] = emb_scores[i] if i < len(emb_scores) else 0.0

        return emb_docs[:k]

    def load_and_index(self, resume_dir: str):
        docs = self.load_resumes(resume_dir)
        if docs:
            self.create_vector_store(docs)
        return self

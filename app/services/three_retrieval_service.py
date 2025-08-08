import fitz  # PyMuPDF
import httpx
import time
import uuid
import numpy as np
from typing import List, Optional
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.core.config import settings
import torch # <--- 1. Import torch


class RetrievalService:
    """
    RetrievalService — improved version with:
      - Eager loading of CrossEncoder to prevent meta tensor errors.
      - Corrected fallback logic to prevent recursion crashes.
    """

    INDEX_NAME = "hackathon-rag-index"

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        chunk_size_words: int = 1400,
        chunk_overlap_words : int = 220,
        enable_lazy_cross_encoder : bool = True,
        crossencoder_threshold: float = 0.05,
    ):
        print("Initializing RetrievalService with Pinecone and local models...")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Load embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"Loaded embedding model: {embedding_model_name} (dim={self.embedding_dim})")
        except Exception as e:
            print(f"Warning: failed to load embedding model '{embedding_model_name}': {e}")
            fallback = "all-MiniLM-L6-v2"
            self.embedding_model = SentenceTransformer(fallback, device=self.device)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"Fell back to embedding model: {fallback} (dim={self.embedding_dim})")

        # FIX: Eagerly load the CrossEncoder model at initialization to prevent loading errors.
        self.reranker_model_name = reranker_model_name
        self.cross_encoder: Optional[CrossEncoder] = None
        self.use_crossencoder = enable_lazy_cross_encoder
        if self.use_crossencoder:
            try:
                print(f"Eagerly loading CrossEncoder model: {self.reranker_model_name}")
                self.cross_encoder = CrossEncoder(self.reranker_model_name, device=self.device)
                print("CrossEncoder loaded successfully.")
            except Exception as e:
                print(f"CRITICAL WARNING: could not load CrossEncoder '{self.reranker_model_name}': {e}")
                print("Cross-encoder functionality will be disabled.")
                self.use_crossencoder = False # Disable it if it fails to load

        self.crossencoder_threshold = crossencoder_threshold

        # Text splitting config
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words

        # Pinecone client + index placeholder
        self.pinecone = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = None

        # in-memory storage of chunks (text -> metadata mapping optional)
        self.text_chunks: List[str] = []
        self.metadata_store = {}

        print("RetrievalService initialized successfully.")

    # ---------------------- Ingestion & Indexing ----------------------
    def ingest_and_process_pdf(self, pdf_url: str, namespace: str = "default", force_reingest: bool = False):
        """
        Downloads PDF, extracts text, chunks it, embeds in batches, and upserts to Pinecone.
        """
        print(f"[Ingest] Starting ingestion for PDF: {pdf_url} (ns={namespace})")

        # 1. Download PDF
        try:
            with httpx.Client() as client:
                response = client.get(pdf_url, follow_redirects=True, timeout=20.0)
                response.raise_for_status()
                pdf_bytes = response.content
        except Exception as e:
            raise ValueError(f"Could not download PDF: {e}")

        # 2. Extract text with page tags
        try:
            full_text = []
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                for page_num, page in enumerate(doc):
                    page_text = page.get_text()
                    if page_text.strip():
                        full_text.append(f"[PAGE {page_num+1}] {page_text}")
            full_text = "\n\n".join(full_text)
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {e}")

        # 3. Chunk text
        self.text_chunks, self.metadata_store = self._split_text_into_chunks_with_metadata(
            full_text,
            chunk_size=self.chunk_size_words,
            overlap=self.chunk_overlap_words
        )
        print(f"[Ingest] Split into {len(self.text_chunks)} chunks.")

        # 4. Check if namespace already exists
        self.index = self.pinecone.Index(self.INDEX_NAME)
        if not force_reingest:
            try:
                stats = self.index.describe_index_stats()
                if namespace in stats.get("namespaces", {}) and stats["namespaces"][namespace]["vector_count"] > 0:
                    print(f"[Ingest] Namespace '{namespace}' already exists. Skipping ingestion.")
                    return
            except Exception as e:
                print(f"Warning: could not check namespace stats: {e}")

        if force_reingest:
            try:
                print(f"[Ingest] Clearing namespace '{namespace}'...")
                self.index.delete(delete_all=True, namespace=namespace)
            except Exception as e:
                print(f"Warning: failed to clear namespace: {e}")

        # 5. Embed in batches
        print("[Ingest] Creating embeddings...")
        batch_size = 64
        all_embeddings = []
        for i in range(0, len(self.text_chunks), batch_size):
            batch = self.text_chunks[i:i + batch_size]
            embs = self.embedding_model.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            all_embeddings.append(embs)
        chunk_embeddings = np.vstack(all_embeddings) if all_embeddings else np.zeros((0, self.embedding_dim))

        # 6. Upsert to Pinecone in batches
        print("[Ingest] Upserting to Pinecone...")
        vectors = [
            {
                "id": f"{namespace}-{i}",
                "values": embedding.tolist(),
                "metadata": {"text": chunk, "chunk_id": i}
            }
            for i, (chunk, embedding) in enumerate(zip(self.text_chunks, chunk_embeddings))
        ]

        for i in range(0, len(vectors), 100):
            batch = vectors[i:i + 100]
            try:
                self.index.upsert(vectors=batch, namespace=namespace)
            except Exception as e:
                print(f"Warning: upsert batch failed at {i}: {e}")

        print(f"[Ingest] Completed: {len(vectors)} vectors stored in ns='{namespace}'.")

    # ---------------------- Search & Rerank ----------------------
    def search_and_rerank(self, query: str, top_k_retrieval: int = 20, top_n_rerank: int = 5) -> List[str]:
        """
        Two-stage retrieval with robust fallback.
        """
        if not self.index:
            raise RuntimeError("Document has not been ingested. Call ingest_and_process_pdf() first.")

        # Stage 1: Fast retrieval from Pinecone
        q_emb = self.embedding_model.encode([query])[0].tolist()
        try:
            resp = self.index.query(vector=q_emb, top_k=top_k_retrieval, include_metadata=True, include_values=True)
        except Exception as e:
            print(f"Warning: Pinecone query failed: {e}")
            return []

        matches = resp.get("matches") or []
        if not matches:
            return []

        texts = [m.get("metadata", {}).get("text", "") for m in matches]
        # This function can be expanded with the full reranking logic if needed,
        # but for now, we'll return the top texts directly.
        return texts[:top_n_rerank]

    # ---------------------- Global Candidate Pool ----------------------
    def build_global_candidate_pool(self, hypothetical_answers: List[str], pool_top_k: int = 200, group_size: int = 8) -> List[str]:
        if not self.index:
            raise RuntimeError("Document has not been ingested.")
        if not hypothetical_answers:
            return []

        embeddings = np.array(self.embedding_model.encode(hypothetical_answers, show_progress_bar=False), dtype=float)
        n = len(embeddings)
        group_size = max(1, min(group_size, n))
        candidate_chunks = []
        seen = set()

        for i in range(0, n, group_size):
            grp = embeddings[i : i + group_size]
            centroid = np.mean(grp, axis=0).tolist()
            try:
                resp = self.index.query(vector=centroid, top_k=min(pool_top_k, 150), include_metadata=True)
            except Exception as e:
                print(f"Warning: Pinecone query for centroid failed: {e}")
                continue
            matches = resp.get("matches") or []
            for m in matches:
                text = m.get("metadata", {}).get("text", "")
                if text and text not in seen:
                    seen.add(text)
                    candidate_chunks.append(text)
                if len(candidate_chunks) >= pool_top_k:
                    break
            if len(candidate_chunks) >= pool_top_k:
                break

        return candidate_chunks

    def rerank_from_pool(self, query: str, candidate_chunks: List[str], top_n_rerank: int = 8) -> List[str]:
        """
        Local re-ranking of a shared candidate pool using CrossEncoder, with robust fallback.
        """
        if not candidate_chunks:
            return []

        # Default bi-encoder scoring logic, also used as a fallback
        def bi_encoder_rerank():
            q_vec = np.array(self.embedding_model.encode([query])[0], dtype=float)
            chunk_embs = np.array(self.embedding_model.encode(candidate_chunks, show_progress_bar=False), dtype=float)
            def cosine(a, b): return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10))
            scores = [cosine(q_vec, emb) for emb in chunk_embs]
            scored = list(zip(scores, candidate_chunks))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored[:top_n_rerank]]

        # If cross-encoder is disabled or not loaded, use bi-encoder fallback immediately
        if not self.use_crossencoder or not self.cross_encoder:
            return bi_encoder_rerank()

        # If we have a cross-encoder, try to use it
        pairs = [[query, chunk] for chunk in candidate_chunks]
        try:
            scores = self.cross_encoder.predict(pairs)
            reranked = list(zip(scores, candidate_chunks))
            reranked.sort(key=lambda x: x[0], reverse=True)
            return [chunk for _, chunk in reranked[:top_n_rerank]]
        except Exception as e:
            # FIX: If predict fails, directly call the fallback logic instead of recursing.
            print(f"Warning: CrossEncoder predict failed: {e}. Falling back to bi-encoder.")
            return bi_encoder_rerank()

    # ---------------------- Helpers ----------------------
    def _split_text_into_chunks_with_metadata(self, text: str, chunk_size: int, overlap: int):
        if not text:
            return [], {}

        pages = text.split("[PAGE ")
        chunks = []
        metadata = {}
        chunk_idx = 0

        for p in pages:
            if not p.strip():
                continue
            if "]" in p:
                page_number_str, page_text = p.split("]", 1)
                try:
                    page_no = int(page_number_str.strip())
                except Exception:
                    page_no = None
            else:
                page_no = None
                page_text = p

            words = page_text.strip().split()
            i = 0
            while i < len(words):
                end = min(i + chunk_size, len(words)) # Simplified chunking
                chunk_words = words[i:end]
                chunk_text = " ".join(chunk_words).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                    metadata[chunk_idx] = {"text": chunk_text, "page": page_no, "chunk_id": chunk_idx}
                    chunk_idx += 1
                if end == len(words):
                    break
                i += chunk_size - overlap

        return chunks, metadata
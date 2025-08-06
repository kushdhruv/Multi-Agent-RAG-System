import fitz  # PyMuPDF
import httpx
import time
from pinecone import Pinecone, ServerlessSpec
# Import both classes directly from the top-level package
from semantic_text_splitter import TextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List
from app.core.config import settings

class RetrievalService:
    """
    A service class responsible for document ingestion, processing, and retrieval
    using Pinecone as the vector database.
    """
    
    INDEX_NAME = "hackathon-rag-index"
    EMBEDDING_DIMENSION = 384 # Based on the 'all-MiniLM-L6-v2' model

    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2', reranker_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initializes the RetrievalService, loading ML models and the Pinecone client.
        """
        print("Initializing RetrievalService with Pinecone...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.reranker_model = CrossEncoder(reranker_model_name)

        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=512,
        chunk_overlap=76  # Overlap in tokens (15% of 512)
    )
        
        # Initialize Pinecone client
        self.pinecone = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = None
        self.text_chunks: List[str] = []
        print("RetrievalService initialized successfully.")

    def ingest_and_process_pdf(self, pdf_url: str):
        """
        Downloads a PDF, extracts text, chunks it, and upserts the embeddings
        into a new Pinecone index.
        """
        print(f"Ingesting PDF from: {pdf_url}")
        # 1. Download and Extract Text (same as before)
        try:
            with httpx.Client() as client:
                response = client.get(pdf_url, follow_redirects=True, timeout=30.0)
                response.raise_for_status()
                pdf_bytes = response.content
        except httpx.RequestError as e:
            raise ValueError(f"Could not download PDF from URL: {pdf_url}")

        full_text = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                full_text += page.get_text()
        
        self.text_chunks = self.text_splitter.split_text(full_text)
        print(f"Split text into {len(self.text_chunks)} chunks.")

        # 2. Create Vector Embeddings
        print("Creating embeddings for text chunks...")
        chunk_embeddings = self.embedding_model.encode(
            self.text_chunks, 
            show_progress_bar=True
        )

        # 3. Setup Pinecone Index
        print("Setting up Pinecone index...")
        # For a hackathon, it's robust to delete and recreate the index on each run
        if self.INDEX_NAME in self.pinecone.list_indexes().names():
            print(f"Deleting existing index: {self.INDEX_NAME}")
            self.pinecone.delete_index(self.INDEX_NAME)
        
        print(f"Creating new index: {self.INDEX_NAME}")
        self.pinecone.create_index(
            name=self.INDEX_NAME,
            dimension=self.EMBEDDING_DIMENSION,
            metric="cosine", # Cosine similarity is standard for semantic search
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # Wait for the index to be ready
        while not self.pinecone.describe_index(self.INDEX_NAME).status['ready']:
            time.sleep(1)

        self.index = self.pinecone.Index(self.INDEX_NAME)
        print("Pinecone index is ready.")

        # 4. Upsert vectors into Pinecone
        print("Upserting vectors to Pinecone...")
        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(self.text_chunks, chunk_embeddings)):
            vectors_to_upsert.append({
                "id": str(i),
                "values": embedding.tolist(),
                "metadata": {"text": chunk} # Store the original text as metadata
            })
        
        # Upsert in batches for efficiency
        self.index.upsert(vectors=vectors_to_upsert, batch_size=100)
        print(f"Successfully upserted {len(vectors_to_upsert)} vectors to Pinecone.")


    def search_and_rerank(self, query: str, top_k_retrieval: int = 20, top_n_rerank: int = 5) -> List[str]:
        """
        Performs a two-stage search using Pinecone for retrieval and a CrossEncoder for reranking.
        """
        if not self.index:
            raise RuntimeError("Document has not been ingested. Call ingest_and_process_pdf() first.")

        # Stage 1: Fast Retrieval with Pinecone
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        query_response = self.index.query(
            vector=query_embedding,
            top_k=top_k_retrieval,
            include_metadata=True
        )
        
        # --- FIX STARTS HERE ---
        # Check if there are any matches before proceeding
        if not query_response['matches']:
            print("Warning: No relevant chunks found in Pinecone for the query.")
            return [] # Return an empty list to prevent errors
        # --- FIX ENDS HERE ---

        retrieved_chunks = [match['metadata']['text'] for match in query_response['matches']]

        # Stage 2: Accurate Reranking with CrossEncoder (same as before)
        rerank_pairs = [[query, chunk] for chunk in retrieved_chunks]
        scores = self.reranker_model.predict(rerank_pairs)
        
        scored_chunks = list(zip(scores, retrieved_chunks))
        scored_chunks.sort(key=lambda x: x, reverse=True)
        
        reranked_chunks = [chunk for score, chunk in scored_chunks[:top_n_rerank]]
        
        return reranked_chunks
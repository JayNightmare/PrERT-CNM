"""
Purpose: Ingest and segment unstructured data (text or PDF) for the PrERT engine, establishing a Context Memory Bank.
Goal: Provide robust text segmentation that feeds a streaming or batched memory bank, ensuring the downstream Transformer never loses global context across chunk boundaries.
Scalability & Innovation: Standard chunking destroys narrative causality. This module prepares a 'Context Memory Bank'—an episodic storage mechanism that allows the sequence encoder to retrieve preceding entity states. We reject naive sliding windows in favor of semantic boundary detection.
"""
import io
import re
import uuid
import chromadb
import fitz
import os
from dotenv import load_dotenv

load_dotenv()

class ContextMemoryBank:
    def __init__(self, collection_name: str = None):
        if collection_name is None:
            collection_name = os.getenv("CHROMA_COLLECTION_NAME")
            if not collection_name:
                raise ValueError("Missing ChromaDB collection name. Please set CHROMADB_COLLECTION_NAME in .env")
            
        api_key = os.getenv("CHROMA_API_KEY")
        tenant = os.getenv("CHROMA_TENANT")
        database = os.getenv("CHROMA_DATABASE")
        
        if not api_key or not tenant or not database:
            raise ValueError(f"Missing ChromaDB credentials in .env. API_KEY: {bool(api_key)}, TENANT: {bool(tenant)}, DATABASE: {bool(database)}")
            
        self.client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def store(self, segment: str, metadata: dict) -> None:
        doc_id = str(uuid.uuid4())
        self.collection.add(
            documents=[segment],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def retrieve_context(self, session_id: str = None) -> list:
        if session_id:
            results = self.collection.get(where={"session_id": session_id})
        else:
            results = self.collection.get()
            
        context_list = []
        if results and results.get("documents"):
            for doc, meta, doc_id in zip(results["documents"], results["metadatas"], results["ids"]):
                context_list.append({
                    "id": doc_id,
                    "segment": doc,
                    "metadata": meta
                })
        context_list.sort(key=lambda x: x["metadata"].get("chunk_idx", 0))
        return context_list

class DocumentIngestor:
    def __init__(self):
        self.memory_bank = ContextMemoryBank()

    def _semantic_chunking(self, text: str) -> list:
        raw_chunks = re.split(r'(?=\n(?:Article|Section|Clause)\s+\d)', text, flags=re.IGNORECASE)
        
        refined_chunks = []
        for chunk in raw_chunks:
            paragraphs = re.split(r'\n\s*\n', chunk)
            current_merged = ""
            for p in paragraphs:
                p = p.strip()
                if not p:
                    continue
                if len(current_merged) + len(p) < 500:
                    current_merged += " " + p if current_merged else p
                else:
                    if current_merged:
                        refined_chunks.append(current_merged.strip())
                    current_merged = p
            if current_merged:
                refined_chunks.append(current_merged.strip())
                
        return [c for c in refined_chunks if c]

    def process_text(self, text: str, session_id: str) -> ContextMemoryBank:
        segments = self._semantic_chunking(text)
        for idx, seg in enumerate(segments):
            self.memory_bank.store(seg, {"chunk_idx": idx, "source": "text", "session_id": session_id})
        return self.memory_bank

    def process_pdf(self, pdf_bytes: bytes, session_id: str) -> ContextMemoryBank:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            full_text = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                full_text.append(page.get_text("text"))
                
        combined_text = "\n".join(full_text)
        segments = self._semantic_chunking(combined_text)
        for idx, seg in enumerate(segments):
            self.memory_bank.store(seg, {"chunk_idx": idx, "source": "pdf", "session_id": session_id})
        
        return self.memory_bank

    def process_document(self, content: bytes | str, is_pdf: bool, session_id: str) -> ContextMemoryBank:
        if is_pdf:
            return self.process_pdf(content, session_id)
        else:
            return self.process_text(str(content), session_id)

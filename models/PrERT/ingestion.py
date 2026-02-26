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

    @staticmethod
    def _is_explicit_policy_boundary(paragraph: str) -> bool:
        p = (paragraph or "").strip()
        if not p:
            return False

        header_pattern = r"^(policy|privacy policy|data policy|notice|terms|section)\s*(\d+|[ivxlcdm]+)?\s*[:\-]"
        transition_pattern = r"^(in a separate policy|for another service|for a different service|another policy)\b"

        return bool(re.match(header_pattern, p, flags=re.IGNORECASE) or re.match(transition_pattern, p, flags=re.IGNORECASE))

    @staticmethod
    def _keyword_set(text: str) -> set[str]:
        stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "your", "you", "are", "our", "all",
            "will", "into", "upon", "while", "have", "has", "had", "their", "them", "they", "not",
            "only", "also", "may", "can", "must", "shall", "should", "about", "under", "over"
        }
        tokens = re.findall(r"\b[a-z]{4,}\b", (text or "").lower())
        return {token for token in tokens if token not in stopwords}

    @staticmethod
    def _are_paragraphs_connected(previous_paragraph: str, current_paragraph: str) -> bool:
        if not previous_paragraph or not current_paragraph:
            return True

        if DocumentIngestor._is_explicit_policy_boundary(current_paragraph):
            return False

        continuation_starts = (
            "this", "these", "such", "it", "they", "however", "additionally", "furthermore",
            "therefore", "moreover", "in addition", "also"
        )
        current_lower = current_paragraph.strip().lower()
        if current_lower.startswith(continuation_starts):
            return True

        prev_keywords = DocumentIngestor._keyword_set(previous_paragraph)
        current_keywords = DocumentIngestor._keyword_set(current_paragraph)
        if not prev_keywords or not current_keywords:
            return True

        overlap = prev_keywords.intersection(current_keywords)
        return len(overlap) >= 2

    @staticmethod
    def _split_into_policy_blocks(text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            return []

        policy_blocks = []
        current_block = [paragraphs[0]]

        for paragraph in paragraphs[1:]:
            previous = current_block[-1]
            connected = DocumentIngestor._are_paragraphs_connected(previous, paragraph)

            if connected:
                current_block.append(paragraph)
            else:
                policy_blocks.append("\n\n".join(current_block))
                current_block = [paragraph]

        if current_block:
            policy_blocks.append("\n\n".join(current_block))

        return policy_blocks

    def _semantic_chunking(self, text: str) -> list[dict]:
        policy_blocks = self._split_into_policy_blocks(text)
        if not policy_blocks:
            return []

        segmented = []
        global_chunk_idx = 0

        for policy_idx, block in enumerate(policy_blocks):
            raw_chunks = re.split(r'(?=\n(?:Article|Section|Clause|Policy)\s+\d)', block, flags=re.IGNORECASE)

            policy_chunk_idx = 0
            for chunk in raw_chunks:
                paragraphs = re.split(r'\n\s*\n', chunk)
                current_merged = ""

                for p in paragraphs:
                    p = p.strip()
                    if not p:
                        continue

                    if len(current_merged) + len(p) < 700:
                        current_merged += " " + p if current_merged else p
                    else:
                        if current_merged:
                            segmented.append({
                                "segment": current_merged.strip(),
                                "chunk_idx": global_chunk_idx,
                                "policy_id": policy_idx,
                                "policy_chunk_idx": policy_chunk_idx,
                            })
                            global_chunk_idx += 1
                            policy_chunk_idx += 1
                        current_merged = p

                if current_merged:
                    segmented.append({
                        "segment": current_merged.strip(),
                        "chunk_idx": global_chunk_idx,
                        "policy_id": policy_idx,
                        "policy_chunk_idx": policy_chunk_idx,
                    })
                    global_chunk_idx += 1
                    policy_chunk_idx += 1

        return segmented

    def process_text(self, text: str, session_id: str) -> ContextMemoryBank:
        segments = self._semantic_chunking(text)
        for seg in segments:
            self.memory_bank.store(seg["segment"], {
                "chunk_idx": seg["chunk_idx"],
                "policy_id": seg["policy_id"],
                "policy_chunk_idx": seg["policy_chunk_idx"],
                "source": "text",
                "session_id": session_id
            })
        return self.memory_bank

    def process_pdf(self, pdf_bytes: bytes, session_id: str) -> ContextMemoryBank:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            full_text = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                full_text.append(page.get_text("text"))
                
        combined_text = "\n".join(full_text)
        segments = self._semantic_chunking(combined_text)
        for seg in segments:
            self.memory_bank.store(seg["segment"], {
                "chunk_idx": seg["chunk_idx"],
                "policy_id": seg["policy_id"],
                "policy_chunk_idx": seg["policy_chunk_idx"],
                "source": "pdf",
                "session_id": session_id
            })
        
        return self.memory_bank

    def process_document(self, content: bytes | str, is_pdf: bool, session_id: str) -> ContextMemoryBank:
        if is_pdf:
            return self.process_pdf(content, session_id)
        else:
            return self.process_text(str(content), session_id)

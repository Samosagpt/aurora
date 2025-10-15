
"""
Generalized RAG (Retrieval-Augmented Generation) Handler for Aurora
Supports JSON database storage with vector search capabilities
"""
import json
import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class RAGDatabase:
    """JSON-based RAG database with vector search simulation"""
    
    def __init__(self, db_path: str = "rag_db.json"):
        """Initialize RAG database
        
        Args:
            db_path: Path to JSON database file
        """
        self.db_path = Path(db_path)
        self.data = {
            "metadata": {
                "created": None,
                "updated": None,
                "version": "1.0",
                "total_documents": 0
            },
            "documents": []
        }
        self.load_database()
    
    def load_database(self) -> bool:
        """Load database from JSON file"""
        try:
            if self.db_path.exists():
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                logger.info(f"✅ Loaded RAG database: {self.db_path} ({self.data['metadata']['total_documents']} documents)")
                return True
            else:
                logger.info(f"📝 Creating new RAG database: {self.db_path}")
                self.data["metadata"]["created"] = datetime.now().isoformat()
                self.save_database()
                return False
        except Exception as e:
            logger.error(f"❌ Error loading RAG database: {e}")
            return False
    
    def save_database(self) -> bool:
        """Save database to JSON file"""
        try:
            self.data["metadata"]["updated"] = datetime.now().isoformat()
            self.data["metadata"]["total_documents"] = len(self.data["documents"])
            
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved RAG database: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving RAG database: {e}")
            return False
    
    def add_document(self, content: str, doc_id: Optional[str] = None, 
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a document to the database
        
        Args:
            content: Document content (text)
            doc_id: Optional document ID (auto-generated if not provided)
            metadata: Optional metadata dictionary
        
        Returns:
            Document ID
        """
        try:
            # Generate document ID if not provided
            if not doc_id:
                doc_id = f"doc_{int(time.time() * 1000)}_{len(self.data['documents'])}"
            
            # Create document object
            document = {
                "id": doc_id,
                "content": content,
                "created": datetime.now().isoformat(),
                "metadata": metadata or {},
                "chunks": self._chunk_text(content)
            }
            
            # Check if document already exists (update instead of add)
            existing_idx = self._find_document_index(doc_id)
            if existing_idx is not None:
                self.data["documents"][existing_idx] = document
                logger.info(f"📝 Updated document: {doc_id}")
            else:
                self.data["documents"].append(document)
                logger.info(f"➕ Added document: {doc_id}")
            
            self.save_database()
            return doc_id
        except Exception as e:
            logger.error(f"❌ Error adding document: {e}")
            return ""
    
    def add_documents_batch(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add multiple documents in batch
        
        Args:
            documents: List of document dictionaries with 'content', optional 'id' and 'metadata'
        
        Returns:
            List of document IDs
        """
        doc_ids = []
        for doc in documents:
            doc_id = self.add_document(
                content=doc.get('content', ''),
                doc_id=doc.get('id'),
                metadata=doc.get('metadata')
            )
            if doc_id:
                doc_ids.append(doc_id)
        
        return doc_ids
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """Split text into chunks for better retrieval
        
        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks
        
        Returns:
            List of chunk dictionaries
        """
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        chunk_idx = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append({
                        "index": chunk_idx,
                        "text": current_chunk.strip(),
                        "length": len(current_chunk.strip())
                    })
                    chunk_idx += 1
                
                # Start new chunk with overlap
                if overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + sentence + " "
                else:
                    current_chunk = sentence + " "
        
        # Add last chunk
        if current_chunk:
            chunks.append({
                "index": chunk_idx,
                "text": current_chunk.strip(),
                "length": len(current_chunk.strip())
            })
        
        return chunks
    
    def search(self, query: str, top_k: int = 5, min_score: float = 0.1) -> List[Dict[str, Any]]:
        """Search for relevant documents using keyword matching
        
        Args:
            query: Search query
            top_k: Number of top results to return
            min_score: Minimum relevance score threshold
        
        Returns:
            List of relevant documents with scores
        """
        try:
            # Normalize query
            query_lower = query.lower()
            query_terms = set(re.findall(r'\w+', query_lower))
            
            results = []
            
            # Search through all documents and chunks
            for doc in self.data["documents"]:
                for chunk in doc["chunks"]:
                    chunk_text_lower = chunk["text"].lower()
                    chunk_terms = set(re.findall(r'\w+', chunk_text_lower))
                    
                    # Calculate simple relevance score (term overlap)
                    common_terms = query_terms & chunk_terms
                    if not common_terms:
                        continue
                    
                    score = len(common_terms) / max(len(query_terms), 1)
                    
                    # Boost score if query appears as phrase
                    if query_lower in chunk_text_lower:
                        score += 0.5
                    
                    # Check minimum score
                    if score >= min_score:
                        results.append({
                            "doc_id": doc["id"],
                            "chunk_index": chunk["index"],
                            "content": chunk["text"],
                            "score": score,
                            "metadata": doc.get("metadata", {})
                        })
            
            # Sort by score (descending)
            results.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"🔍 Search query: '{query}' - Found {len(results)} results")
            return results[:top_k]
        
        except Exception as e:
            logger.error(f"❌ Error during search: {e}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID
        
        Args:
            doc_id: Document ID
        
        Returns:
            Document dictionary or None
        """
        idx = self._find_document_index(doc_id)
        if idx is not None:
            return self.data["documents"][idx]
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID
        
        Args:
            doc_id: Document ID
        
        Returns:
            True if deleted, False otherwise
        """
        try:
            idx = self._find_document_index(doc_id)
            if idx is not None:
                del self.data["documents"][idx]
                self.save_database()
                logger.info(f"🗑️ Deleted document: {doc_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error deleting document: {e}")
            return False
    
    def clear_database(self) -> bool:
        """Clear all documents from database"""
        try:
            self.data["documents"] = []
            self.save_database()
            logger.info("🗑️ Cleared RAG database")
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing database: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        total_docs = len(self.data["documents"])
        total_chunks = sum(len(doc.get("chunks", [])) for doc in self.data["documents"])
        total_size = sum(len(doc.get("content", "")) for doc in self.data["documents"])
        
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "total_characters": total_size,
            "created": self.data["metadata"].get("created"),
            "updated": self.data["metadata"].get("updated")
        }
    
    def _find_document_index(self, doc_id: str) -> Optional[int]:
        """Find document index by ID"""
        for idx, doc in enumerate(self.data["documents"]):
            if doc["id"] == doc_id:
                return idx
        return None


class RAGHandler:
    """High-level RAG handler for Aurora with Ollama integration"""
    
    def __init__(self, db_path: str = "rag_db.json", ollama_model: str = "llama3.2"):
        """Initialize RAG handler
        
        Args:
            db_path: Path to JSON database file
            ollama_model: Ollama model for generation
        """
        self.db = RAGDatabase(db_path)
        self.ollama_model = ollama_model
        logger.info(f"🤖 RAG Handler initialized with model: {ollama_model}")
    
    def add_knowledge(self, content: str, source: str = "manual", 
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add knowledge to RAG database
        
        Args:
            content: Knowledge content
            source: Source of knowledge (e.g., "manual", "file", "web")
            metadata: Additional metadata
        
        Returns:
            Document ID
        """
        meta = metadata or {}
        meta["source"] = source
        meta["added_time"] = datetime.now().isoformat()
        
        return self.db.add_document(content, metadata=meta)
    
    def query(self, question: str, context_window: int = 3, 
             return_sources: bool = True) -> Dict[str, Any]:
        """Query RAG system with a question
        
        Args:
            question: User question
            context_window: Number of context chunks to retrieve
            return_sources: Whether to return source documents
        
        Returns:
            Dictionary with answer and optional sources
        """
        try:
            # Search for relevant context
            search_results = self.db.search(question, top_k=context_window)
            
            if not search_results:
                return {
                    "answer": "I don't have enough information in my knowledge base to answer that question.",
                    "sources": [],
                    "context_used": False
                }
            
            # Build context from search results
            context_parts = []
            sources = []
            
            for idx, result in enumerate(search_results):
                context_parts.append(f"[Context {idx+1}]\n{result['content']}")
                if return_sources:
                    sources.append({
                        "doc_id": result["doc_id"],
                        "score": result["score"],
                        "metadata": result["metadata"]
                    })
            
            context = "\n\n".join(context_parts)
            
            # Generate answer using Ollama (if available)
            try:
                import ollama
                
                prompt = f"""Answer the following question naturally and conversationally using the provided context. 

Context Information:
{context}

User Question: {question}

IMPORTANT Instructions:
1. Answer naturally as if you know this information directly - DO NOT use phrases like "according to the context", "based on the provided information", "the context states", etc.
2. Be conversational and confident in your response
3. If the context has the answer, state it directly as fact
4. Only mention if information is missing or unclear when truly necessary
5. Be concise but complete
6. Use a friendly, helpful tone

Answer the question now:"""
                
                response = ollama.chat(
                    model=self.ollama_model,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                answer = response.get("message", {}).get("content", "").strip()
                
            except Exception as e:
                logger.warning(f"⚠️ Ollama generation failed: {e}")
                # Fallback to simple context return
                answer = f"Based on the available information:\n\n{context}"
            
            return {
                "answer": answer,
                "sources": sources if return_sources else [],
                "context_used": True,
                "num_contexts": len(search_results)
            }
        
        except Exception as e:
            logger.error(f"❌ Error during RAG query: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "context_used": False
            }
    
    def import_from_file(self, filepath: str, chunk_size: int = 1000) -> List[str]:
        """Import knowledge from a text file
        
        Args:
            filepath: Path to text file
            chunk_size: Maximum characters per document chunk
        
        Returns:
            List of document IDs
        """
        try:
            path = Path(filepath)
            if not path.exists():
                logger.error(f"❌ File not found: {filepath}")
                return []
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks if content is large
            doc_ids = []
            if len(content) <= chunk_size:
                doc_id = self.add_knowledge(
                    content,
                    source=f"file:{path.name}",
                    metadata={"filename": path.name, "size": len(content)}
                )
                doc_ids.append(doc_id)
            else:
                # Split into multiple documents
                chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                for idx, chunk in enumerate(chunks):
                    doc_id = self.add_knowledge(
                        chunk,
                        source=f"file:{path.name}",
                        metadata={
                            "filename": path.name,
                            "chunk_index": idx,
                            "total_chunks": len(chunks)
                        }
                    )
                    doc_ids.append(doc_id)
            
            logger.info(f"📂 Imported {len(doc_ids)} documents from {path.name}")
            return doc_ids
        
        except Exception as e:
            logger.error(f"❌ Error importing from file: {e}")
            return []
    
    def export_to_file(self, filepath: str) -> bool:
        """Export entire database to a file
        
        Args:
            filepath: Path to output file
        
        Returns:
            True if successful
        """
        try:
            path = Path(filepath)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.db.data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Exported RAG database to {filepath}")
            return True
        except Exception as e:
            logger.error(f"❌ Error exporting database: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return self.db.get_stats()


# Global RAG handler instance
_rag_handler = None

def get_rag_handler(db_path: str = "rag_db.json", ollama_model: str = "llama3.2") -> RAGHandler:
    """Get or create global RAG handler instance"""
    global _rag_handler
    if _rag_handler is None:
        _rag_handler = RAGHandler(db_path, ollama_model)
    return _rag_handler


# Convenience functions
def add_knowledge(content: str, source: str = "manual", metadata: Optional[Dict[str, Any]] = None) -> str:
    """Add knowledge to default RAG database"""
    handler = get_rag_handler()
    return handler.add_knowledge(content, source, metadata)


def query_rag(question: str, context_window: int = 3) -> str:
    """Query default RAG database and return answer"""
    handler = get_rag_handler()
    result = handler.query(question, context_window)
    return result["answer"]


def import_knowledge_file(filepath: str) -> List[str]:
    """Import knowledge from file to default RAG database"""
    handler = get_rag_handler()
    return handler.import_from_file(filepath)


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 60)
    print("Aurora RAG System - Test Mode")
    print("=" * 60)
    
    # Initialize RAG handler
    rag = RAGHandler(db_path="test_rag.json")
    
    # Add some sample knowledge
    print("\n📝 Adding sample knowledge...")
    rag.add_knowledge(
        "Aurora is an advanced AI assistant developed to help users with various tasks. "
        "It features voice recognition, text-to-speech, and multimodal capabilities.",
        source="manual",
        metadata={"category": "about"}
    )
    
    rag.add_knowledge(
        "Aurora supports multiple models through Ollama integration, including llama3.2, "
        "codellama, mistral, and more. Users can switch between models easily.",
        source="manual",
        metadata={"category": "features"}
    )
    
    rag.add_knowledge(
        "The RAG (Retrieval-Augmented Generation) system allows Aurora to access and "
        "utilize a knowledge base stored in JSON format for better responses.",
        source="manual",
        metadata={"category": "features"}
    )
    
    # Display stats
    stats = rag.get_database_stats()
    print(f"\n📊 Database Stats:")
    print(f"  - Documents: {stats['total_documents']}")
    print(f"  - Chunks: {stats['total_chunks']}")
    print(f"  - Characters: {stats['total_characters']}")
    
    # Test query
    print("\n🔍 Testing query...")
    question = "What is Aurora and what can it do?"
    result = rag.query(question)
    
    print(f"\n❓ Question: {question}")
    print(f"\n💡 Answer:\n{result['answer']}")
    print(f"\n📚 Used {result['num_contexts']} context(s)")
    
    if result['sources']:
        print(f"\n📖 Sources:")
        for idx, source in enumerate(result['sources']):
            print(f"  {idx+1}. Document: {source['doc_id']} (score: {source['score']:.2f})")
    
    print("\n✅ RAG system test completed!")

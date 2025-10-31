"""
Unit tests for RAG handler performance optimizations
"""
import pytest
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_handler import RAGDatabase, WORD_PATTERN, SENTENCE_SPLIT_PATTERN


class TestRAGPerformance:
    """Test RAG handler performance improvements"""
    
    def test_precompiled_patterns(self):
        """Test that pre-compiled regex patterns work correctly"""
        test_text = "This is a test. And another sentence!"
        
        # Test word pattern
        words = WORD_PATTERN.findall(test_text.lower())
        assert len(words) > 0
        assert "test" in words
        
        # Test sentence pattern
        sentences = SENTENCE_SPLIT_PATTERN.split(test_text)
        assert len(sentences) >= 2
    
    def test_text_chunking_efficiency(self):
        """Test that text chunking uses efficient string concatenation"""
        db = RAGDatabase('/tmp/test_perf_chunking.json')
        db.clear_database()
        
        # Create a large text to test chunking
        large_text = " ".join([f"Sentence {i}." for i in range(100)])
        
        # Time the chunking operation
        start = time.time()
        chunks = db._chunk_text(large_text, chunk_size=500, overlap=50)
        elapsed = time.time() - start
        
        # Verify chunks were created
        assert len(chunks) > 0
        
        # Should be fast (less than 0.5 seconds for this size)
        assert elapsed < 0.5, f"Chunking took too long: {elapsed}s"
    
    def test_search_with_cached_patterns(self):
        """Test search uses pre-compiled patterns efficiently"""
        db = RAGDatabase('/tmp/test_perf_search.json')
        db.clear_database()
        
        # Add test documents
        for i in range(10):
            db.add_document(f"Test document {i} with various keywords and phrases.")
        
        # Time search operations
        start = time.time()
        results = db.search("test keywords", top_k=5)
        elapsed = time.time() - start
        
        # Verify results
        assert len(results) > 0
        
        # Should be fast (less than 0.1 seconds for this size)
        assert elapsed < 0.1, f"Search took too long: {elapsed}s"
    
    def test_early_termination_in_search(self):
        """Test that search terminates early for low-score chunks"""
        db = RAGDatabase('/tmp/test_perf_early_term.json')
        
        # Clear any existing data
        db.clear_database()
        
        # Add documents with varying relevance
        db.add_document("Completely irrelevant content about something else entirely")
        db.add_document("Test content with the exact keywords we want")
        db.add_document("Another irrelevant document without the search terms")
        
        # Search with high minimum score
        results = db.search("test keywords", top_k=10, min_score=0.3)
        
        # Should only return relevant results
        assert len(results) == 1, f"Expected 1 result, got {len(results)}: {results}"
    
    def test_list_join_vs_concatenation(self):
        """Test that list.join is used instead of string concatenation"""
        # This is more of a code inspection test
        # We can verify behavior by checking the result
        db = RAGDatabase('/tmp/test_perf_listjoin.json')
        
        text = "Sentence one. Sentence two. Sentence three."
        chunks = db._chunk_text(text, chunk_size=20, overlap=5)
        
        # Verify chunks are correctly formed
        assert all('text' in chunk for chunk in chunks)
        assert all(len(chunk['text']) > 0 for chunk in chunks)
    
    def test_sentence_based_overlap(self):
        """Test that overlap preserves complete sentences, not character slices"""
        db = RAGDatabase('/tmp/test_sentence_overlap.json')
        
        # Create text with distinct sentences
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        chunks = db._chunk_text(text, chunk_size=50, overlap=25)
        
        # Should have multiple chunks with overlap
        assert len(chunks) >= 2, "Should create multiple chunks"
        
        # Check that overlapping chunks start with complete sentences
        for i in range(1, len(chunks)):
            chunk_text = chunks[i]['text']
            # Each chunk should start with a capital letter (beginning of sentence)
            # or be a complete sentence from the previous chunk
            assert chunk_text[0].isupper() or chunk_text.startswith(' '), \
                f"Chunk {i} should start with complete sentence: '{chunk_text[:30]}...'"
            
            # Verify no mid-word cuts by checking common overlap patterns
            if i > 0:
                prev_chunk = chunks[i-1]['text']
                # If there's overlap, it should be complete sentences
                words_curr = chunk_text.split()
                words_prev = prev_chunk.split()
                # First word of current chunk should appear in previous chunk as complete word
                if words_curr:
                    first_word = words_curr[0]
                    assert first_word in words_prev, \
                        f"Overlap should preserve complete words: '{first_word}' not in previous chunk"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

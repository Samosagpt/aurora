"""
Test script for Aurora RAG System
Demonstrates basic RAG functionality
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_handler import RAGHandler, get_rag_handler

def main():
    print("=" * 70)
    print("Aurora RAG System - Interactive Test")
    print("=" * 70)
    
    # Initialize RAG handler
    print("\n🚀 Initializing RAG handler...")
    rag = RAGHandler(db_path="rag_db.json", ollama_model="llama3.2")
    
    # Check if database is empty
    stats = rag.get_database_stats()
    print(f"\n📊 Current database stats:")
    print(f"  - Documents: {stats['total_documents']}")
    print(f"  - Chunks: {stats['total_chunks']}")
    print(f"  - Characters: {stats['total_characters']:,}")
    
    # If empty, offer to load sample data
    if stats['total_documents'] == 0:
        print("\n📭 Database is empty. Would you like to load sample knowledge?")
        response = input("Load sample_knowledge.txt? (y/n): ").strip().lower()
        
        if response == 'y':
            sample_file = Path(__file__).parent / "sample_knowledge.txt"
            if sample_file.exists():
                print(f"\n📂 Loading {sample_file}...")
                doc_ids = rag.import_from_file(str(sample_file))
                print(f"✅ Loaded {len(doc_ids)} document(s)")
                
                # Refresh stats
                stats = rag.get_database_stats()
                print(f"\n📊 Updated database stats:")
                print(f"  - Documents: {stats['total_documents']}")
                print(f"  - Chunks: {stats['total_chunks']}")
                print(f"  - Characters: {stats['total_characters']:,}")
            else:
                print(f"❌ Sample file not found: {sample_file}")
    
    # Interactive query loop
    print("\n" + "=" * 70)
    print("💬 Interactive Query Mode")
    print("=" * 70)
    print("\nAsk questions about Aurora or type 'exit' to quit.")
    print("Examples:")
    print("  - What is Aurora?")
    print("  - How do I use image generation?")
    print("  - What are the system requirements?")
    print("  - How does the RAG system work?")
    print()
    
    while True:
        try:
            # Get user input
            question = input("\n❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Special commands
            if question.lower() == 'stats':
                stats = rag.get_database_stats()
                print(f"\n📊 Database Statistics:")
                print(f"  - Documents: {stats['total_documents']}")
                print(f"  - Chunks: {stats['total_chunks']}")
                print(f"  - Characters: {stats['total_characters']:,}")
                if stats.get('created'):
                    print(f"  - Created: {stats['created']}")
                if stats.get('updated'):
                    print(f"  - Updated: {stats['updated']}")
                continue
            
            if question.lower() == 'help':
                print("\n📖 Available commands:")
                print("  - stats: Show database statistics")
                print("  - help: Show this help message")
                print("  - exit/quit/q: Exit the program")
                print("\nOr ask any question about Aurora!")
                continue
            
            # Query RAG system
            print("\n🔍 Searching knowledge base...")
            result = rag.query(question, context_window=3)
            
            # Display answer
            print("\n" + "=" * 70)
            print("💡 Answer:")
            print("=" * 70)
            print(result['answer'])
            print()
            
            # Display sources
            if result.get('sources'):
                print("📚 Sources used:")
                for idx, source in enumerate(result['sources']):
                    print(f"  {idx+1}. Document: {source['doc_id'][:50]}... (relevance: {source['score']:.2f})")
                print()
            
            # Display stats
            if result.get('context_used'):
                print(f"ℹ️  Retrieved {result['num_contexts']} relevant context(s)")
            else:
                print("⚠️  No relevant context found")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

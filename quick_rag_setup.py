"""
Quick test to add sample knowledge and verify RAG works
"""

from rag_handler import get_rag_handler

def quick_setup():
    print("="*80)
    print("QUICK RAG SETUP TEST")
    print("="*80)
    
    rag = get_rag_handler()
    
    # Check current stats
    stats = rag.get_database_stats()
    print(f"\n📊 Current Database:")
    print(f"   Documents: {stats['total_documents']}")
    print(f"   Chunks: {stats['total_chunks']}")
    
    # Add some test knowledge
    print("\n➕ Adding test knowledge...")
    
    test_knowledge = [
        {
            "content": """AURORA is an advanced AI assistant created by the Aurora project contributors. 
            The name stands for Agentic Unified multi-model Reasoning Orchestrator for Rapid One-shot Assistance.
            AURORA combines multiple AI capabilities including chat, image generation, video generation, voice interaction, and web search.""",
            "source": "project_info",
            "category": "about_aurora"
        },
        {
            "content": """The AURORA development team consists of core contributors:
            1. Aurora project contributors - see CONTRIBUTORS.md or the project GitHub for details""",
            "source": "team_info",
            "category": "team"
        },
        {
            "content": """AURORA's RAG (Retrieval-Augmented Generation) system uses a JSON-based knowledge database. 
            It can store documents, search for relevant content, and enhance AI responses with contextual information.
            The RAG system gives priority to knowledge base information over general AI knowledge.""",
            "source": "features",
            "category": "rag_system"
        },
        {
            "content": """To use AURORA effectively:
            1. Chat with AI using the main chat interface
            2. Generate images in the Image Generation page
            3. Create videos in the Video Generation page
            4. Add knowledge to the RAG database in Knowledge Base page
            5. Use voice input for hands-free interaction""",
            "source": "usage_guide",
            "category": "how_to_use"
        }
    ]
    
    for knowledge in test_knowledge:
        try:
            doc_id = rag.add_knowledge(
                content=knowledge["content"],
                source=knowledge["source"],
                metadata={"category": knowledge["category"]}
            )
            print(f"   ✅ Added: {knowledge['category']} (ID: {doc_id})")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Check updated stats
    stats = rag.get_database_stats()
    print(f"\n📊 Updated Database:")
    print(f"   Documents: {stats['total_documents']}")
    print(f"   Chunks: {stats['total_chunks']}")
    
    # Test some queries
    print("\n🔍 Testing Queries...")
    print("="*80)
    
    test_queries = [
        "What is AURORA?",
        "Who are the AURORA developers?",
        "How does the RAG system work?",
        "How do I use AURORA?"
    ]
    
    for query in test_queries:
        print(f"\n❓ Question: {query}")
        print("-"*80)
        
        try:
            # Search for context
            results = rag.db.search(query, top_k=2)
            
            if results:
                print(f"✅ Found {len(results)} relevant result(s)")
                print(f"   Best match: {results[0]['metadata'].get('category', 'N/A')} (score: {results[0]['score']:.2f})")
                print(f"   Content preview: {results[0]['content'][:100]}...")
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*80)
    print("✅ SETUP COMPLETE!")
    print("="*80)
    print("\n💡 Tips:")
    print("   1. Run 'streamlit run streamlit_app.py' to start the app")
    print("   2. The 'Use RAG' checkbox in chat is enabled by default")
    print("   3. Ask questions about AURORA to test RAG")
    print("   4. Look for the '📚 Answer from Knowledge Base' badge")
    print("\n")

if __name__ == "__main__":
    quick_setup()

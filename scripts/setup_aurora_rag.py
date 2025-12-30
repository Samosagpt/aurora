"""
Initialize AURORA knowledge base with project information
This script adds AURORA system information to the RAG database
"""

import json

from aurora_system import get_aurora_system
from rag_handler import get_rag_handler


def initialize_aurora_knowledge():
    """Add AURORA system information to the RAG knowledge base"""

    print("Initializing AURORA Knowledge Base...")
    print("=" * 80)

    # Get AURORA system configuration
    aurora = get_aurora_system()

    # Get RAG handler
    rag = get_rag_handler()

    # Prepare knowledge documents
    documents = []

    # 1. Project Overview
    project = aurora.config.get("project", {})
    documents.append(
        {
            "content": f"""# AURORA Project Overview

{project.get('name', 'AURORA')} stands for {project.get('full_form', '')}.

Version: {project.get('version', '1.0')}

{project.get('attribution', '')}

AURORA is an advanced AI assistant that combines multiple AI capabilities for one-shot rapid assistance across diverse tasks.""",
            "metadata": {
                "source": "aurora_config.json",
                "category": "project_overview",
                "type": "official",
            },
        }
    )

    # 2. Capabilities
    capabilities = aurora.get_capabilities()
    capabilities_text = f"""# AURORA Capabilities

## Modalities
AURORA supports the following modalities:
{chr(10).join(f"- {mod}" for mod in capabilities.get('modalities', []))}

## Skills
AURORA can perform:
{chr(10).join(f"- {skill.replace('_', ' ').title()}" for skill in capabilities.get('skills', []))}

These capabilities make AURORA a versatile AI assistant for various tasks."""

    documents.append(
        {
            "content": capabilities_text,
            "metadata": {
                "source": "aurora_config.json",
                "category": "capabilities",
                "type": "official",
            },
        }
    )

    # 3. Team Information
    team_members = aurora.get_team_info()
    for member in team_members:
        member_text = f"""# Team Member: {member['name']}

## Roles
{chr(10).join(f"- {role.replace('_', ' ').title()}" for role in member['roles'])}

## Responsibilities
{chr(10).join(f"- {resp}" for resp in member['responsibilities'])}

## Skills
{chr(10).join(f"- {skill.replace('_', ' ').title()}" for skill in member['skills'])}

{member['name']} is a core member of the AURORA development team."""

        documents.append(
            {
                "content": member_text,
                "metadata": {
                    "source": "aurora_config.json",
                    "category": "team",
                    "member_id": member["id"],
                    "type": "official",
                },
            }
        )

    # 4. Operational Policies
    policies = aurora.get_policies()
    policies_text = f"""# AURORA Operational Policies

## Privacy
{policies.get('privacy', '')}

## Safety
{policies.get('safety', '')}

## Factuality
{policies.get('factuality', '')}

These policies guide AURORA's operations and interactions."""

    documents.append(
        {
            "content": policies_text,
            "metadata": {
                "source": "aurora_config.json",
                "category": "policies",
                "type": "official",
            },
        }
    )

    # 5. System Prompt Summary
    system_prompt = aurora.get_system_prompt()
    documents.append(
        {
            "content": f"""# AURORA System Prompt

{system_prompt}

This prompt defines AURORA's identity, tone, output format, and operational guidelines.""",
            "metadata": {
                "source": "aurora_config.json",
                "category": "system_prompt",
                "type": "official",
            },
        }
    )

    # Add documents to RAG
    print(f"\nAdding {len(documents)} documents to RAG knowledge base...")

    added_ids = []
    for i, doc in enumerate(documents, 1):
        try:
            doc_id = rag.add_knowledge(
                content=doc["content"],
                source=doc["metadata"].get("source", "aurora_config"),
                metadata=doc["metadata"],
            )
            added_ids.append(doc_id)
            print(f"  [{i}/{len(documents)}] Added: {doc['metadata']['category']} (ID: {doc_id})")
        except Exception as e:
            print(f"  [{i}/{len(documents)}] Error: {e}")

    print("\n" + "=" * 80)
    print("✅ AURORA Knowledge Base Initialized Successfully!")
    print(f"Total documents added: {len(added_ids)}")
    print("=" * 80)

    # Test query
    print("\nTesting RAG with sample queries...")
    print("-" * 80)

    test_queries = [
        "What is AURORA?",
        "Who created AURORA?",
        "What are AURORA's capabilities?",
        "Tell me about the AURORA team",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            answer = rag.query(query, context_window=2)
            print(f"Answer: {answer[:150]}..." if len(answer) > 150 else f"Answer: {answer}")
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 80)
    print("Initialization Complete!")


if __name__ == "__main__":
    initialize_aurora_knowledge()

# 🌌 AURORA Documentation

## 🔍 Overview

**AURORA** stands for **Agentic Unified multi-model Reasoning Orchestrator for Rapid One-shot Assistance**.  
It is an advanced AI assistant designed to combine multiple intelligent capabilities—conversation, generation, reasoning, and automation—under one unified system.

Developed collaboratively by the **Aurora project contributors**, it provides a flexible platform for interacting with various AI models and tools in real time.

---

## ✨ Key Features

1. **🧠 Intelligent Chat** – Natural and context-aware conversations powered by Ollama models.  
2. **🎨 Image Generation** – Create stunning AI-generated images using **Stable Diffusion**.  
3. **🎞️ Video Generation** – Generate short videos directly from text descriptions.  
4. **📚 Retrieval-Augmented Generation (RAG)** – Integrate custom documents and ask questions using your own knowledge base.  
5. **🎤 Voice Interaction** – Speak naturally using built-in speech-to-text and text-to-speech features.  
6. **🌐 Web Integration** – Search Wikipedia, Google, and YouTube, and fetch live weather or news updates.  
7. **🖥️ Desktop Automation** – Control desktop tasks and workflows with AI agents.

---

## 📖 Using the RAG System

The **Retrieval-Augmented Generation (RAG)** system enhances AURORA’s responses with information from your personal knowledge base.

### 🪜 Steps to Use RAG
1. **Enable RAG** — Check the “Use RAG” option in the sidebar.  
2. **Add Documents** — Upload or add files in the *RAG Knowledge Base* page.  
3. **Ask Questions** — Use the chat as usual; AURORA automatically searches your documents.  
4. **View Results** — Responses enriched with RAG context show a **📚** indicator.  

💡 *Tip: Keep your knowledge base updated for more accurate answers.*

---

## 🖥️ System Requirements

| Requirement | Details |
|--------------|----------|
| **Python** | 3.8 or higher |
| **Chat Engine** | Ollama (download from [ollama.ai](https://ollama.ai)) |
| **Memory (RAM)** | Minimum 8 GB, recommended 16 GB |
| **GPU** | Strongly recommended for image and video generation |
| **Microphone** | Optional, for voice input |

---

## 👥 Team

Developed and maintained by **Aurora Project Contributors**.  
For the complete list of contributors, see [`CONTRIBUTORS.md`](../CONTRIBUTORS.md) or visit the [Aurora GitHub Repository](https://github.com/<username>/aurora).


---

## 🧩 Troubleshooting Guide

### 💬 Chat Not Working?
- Ensure **Ollama** is installed and running with:
  ```bash
  ollama serve
  ```


### 📚 RAG Not Responding?
Check that your knowledge base contains uploaded documents.

### 🖼️ Images or Videos Not Generating?
Confirm GPU availability and verify your Stable Diffusion installation.

## ℹ️ Additional Help
For further assistance, open the About page within the application or consult the documentation in the /docs directory.

This document was converted from AURORA_DOCS.txt to Markdown format to enhance readability, structure, and accessibility as part of issue #5.

---

✅ **What this version improves:**
- Uses proper Markdown headings & emojis for clarity.
- Adds spacing and formatting for readability.
- Converts plain lists into clear structured sections.
- Adds links & tables for a polished, professional look.
- Meets the issue requirement: “Convert and enhance AURORA_DOCS.txt to proper Markdown format.”

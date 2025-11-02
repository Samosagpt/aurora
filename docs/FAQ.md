# ❓ Frequently Asked Questions (FAQ)

This document answers common questions about installing, using, and contributing to **AURORA**.

---

## 💡 General

### **Q1. What is AURORA?**
AURORA stands for **Agentic Unified multi-model Reasoning Orchestrator for Rapid One-shot Assistance**.  
It is an advanced open-source AI assistant that combines chat, image, video, and automation features into one unified system.

### **Q2. Who maintains AURORA?**
AURORA is developed and maintained by the **Aurora Project Contributors**.  
You can find the full list in the [`CONTRIBUTORS.md`](../CONTRIBUTORS.md) file or on the [official GitHub repository](https://github.com/Tejaji-0/aurora).

---

## ⚙️ Installation & Setup

### **Q3. How do I install AURORA?**
Please follow the detailed steps in the [`docs/INSTALLATION.md`](INSTALLATION.md) guide.

### **Q4. I’m getting “command not found: ollama” — what should I do?**
Install **Ollama** from [ollama.ai](https://ollama.ai) and ensure it’s running by executing:
```bash
ollama serve
Then restart AURORA.
```
### **Q5. Do I need a GPU to run AURORA?**
A GPU is recommended for image and video generation, but AURORA can run on CPU for chat and RAG-based tasks.

## 💬 Using AURORA

### **Q6. How does RAG (Retrieval-Augmented Generation) work?**
RAG enhances responses by searching your knowledge base for relevant documents.
You can enable it using the “Use RAG” checkbox in the sidebar.
For detailed instructions, see docs/AURORA_DOCS.md

## **Q7. Why isn’t RAG giving any results?**
- Ensure that you’ve uploaded documents to your knowledge base.
- Restart the app and try again.
- Check console logs for missing dependencies.

## 🧑‍💻 Contributing
### **Q8. How can I contribute to AURORA?**
Follow the contribution steps in CONTRIBUTING.md

### **Q9. How do I report bugs or suggest features?**

- Open an issue on the GitHub Issues page
- Please include steps to reproduce the problem and screenshots if applicable.

## 🧩 Troubleshooting
### **Q10. My chat window doesn’t respond.**

- Ensure Ollama is running (ollama serve)
- Check internet connectivity
- Restart the application

### **Q11. Images are not generating.**

**Make sure:**
- You have a GPU or the correct Stable Diffusion model installed
- All dependencies are properly configured

## 📞 Support

If your question isn’t answered here, please:
Check the About section in the app, or
Reach out through the Aurora GitHub Discussions https://github.com/Tejaji-0/aurora/discussions


**Made with ❤️ by the Aurora project contributors**

*AURORA - Your intelligent companion for the age of AI*


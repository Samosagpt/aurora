# Configuration Guide

## Overview
This document explains how to configure AURORA for optimal performance and feature access.

## Configuration Files
- **`.env`** — Contains environment variables (API keys, secrets)
- **`config.yaml`** — Holds runtime and model settings

## Key Parameters

| Parameter | Description | Example |
|------------|--------------|----------|
| `MODEL_NAME` | Default language model | `llama3:latest` |
| `IMAGE_MODEL` | Image generation model | `stable-diffusion-v1-5` |
| `ENABLE_RAG` | Enables Retrieval-Augmented Generation | `true` |
| `KNOWLEDGE_BASE_PATH` | Directory for RAG docs | `./data/rag_docs/` |
| `PORT` | App running port | `8080` |

## API Keys
If using web integrations (Google, Weather, etc.), add your keys inside `.env`:
```bash
GOOGLE_API_KEY=your_key
OPENWEATHER_API_KEY=your_key
```

## 🪵 Logging

You can modify the log level in your `config.yaml` file:

```yaml
logging:
  level: INFO
```

 ## Advanced Options
- Enable GPU acceleration
- Configure custom embeddings for RAG
- Use Docker for isolated setup (see INSTALLATION.md)

---

### 🔗  `docs/API.md`

```markdown
# API Documentation

## Overview
This document describes AURORA’s main API endpoints and expected responses.

### Base URL
http://localhost:8080/api


---

### 🧠 Chat API
**Endpoint:** `/api/chat`  
**Method:** `POST`

**Request:**
```json
{
  "message": "Hello AURORA!"
}
```
**Response:**
```json
{
  "reply": "Hello! How can I assist you today?"
}
```

### 🖼️ Image Generation API
**Endpoint:** /api/generate/image
**Method:**  POST

**Request:**
```json
{
  "prompt": "A futuristic city skyline at sunset"
}
```
**Response**
```json 
{
  "image_url": "/outputs/generated_image.png"
}
```

### 🧩 RAG Query API
 **Endpoint:**/api/rag/query
 **Method:**POST

**Request:**
```json
{
  "question": "What is Aurora?",
  "use_rag": true
}
```
**Response**
```json
{
  "answer": "Aurora is an AI assistant that uses RAG for contextual responses."
}
```

### 🌦️ External Integrations

**Endpoints for:**
- `/api/weather`
- `/api/news`
- `/api/search`

📘 **Note:**  
Refer to the respective module documentation for detailed parameters and usage examples.

### 🧰 Error Handling

**Standard error response:**
```json
{
  "error": "Invalid request",
  "details": "Missing required field: prompt"
}
```
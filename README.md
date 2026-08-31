## 🤖 AI - BizVision RAG

#### Multimodal RAG for Business Documents

BizVision-RAG is a **Multimodal Retrieval-Augmented Generation (RAG)** system designed to understand and answer questions from business documents containing both **text and images**.

The system processes PDF and DOCX documents, extracts and chunks textual information, extracts images from PDFs, generates embeddings for both modalities, retrieves relevant information, and uses a vision-capable Large Language Model (LLM) to generate the final answer.

---
business documents often contain important information in:

- Text
- Charts
- Graphs
- Tables
- Images
- Visual reports

BizVision-RAG extends traditional RAG into a **Multimodal RAG architecture** so that the system can retrieve and reason over both textual and visual information.


---
#### Project Setup

1. Clone the Repository
```bash
git clone https://github.com/Kumar-with-code/BizVision-Rag.git
cd BizVision-Rag
```

2. Install Dependencies

This project uses uv.

```bash
uv sync
```

3. Environment Variables

```bash
GROQ_API_KEY=your_groq_api_key
```
4. Run the Project from the project root:

```bash
uv run python -m app.core.rag_pipeline
```
---
# Architecture
```text
                         User
                          │
                          ▼
                 ┌─────────────────┐
                 │   PDF / DOCX    │
                 └────────┬────────┘
                          │
                 Document Processing
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
            TEXT                   IMAGES
              │                       │
              ▼                       ▼
          Chunking                PyMuPDF
              │                       │
              ▼                       ▼
      MiniLM Embeddings             PIL
              │                       │
              ▼                       ▼
         Text FAISS                 CLIP
              │                       │
              │                       ▼
              │                  Image FAISS
              │                       │
              └───────────┬───────────┘
                          │
                     User Query
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
       Text Retrieval          CLIP Text Retrieval
              │                       │
              ▼                       ▼
        Text Context               Images
              │                       │
              └───────────┬───────────┘
                          │
                          ▼
                 Multimodal Context
                          │
                          ▼
                 Qwen Vision LLM
                          │
                          ▼
                     Final Answer
# Medical Chatbot System Analysis

This document provides a comprehensive technical overview and analysis of the `MedicalChatbot-with-LLMs-LangChain-Pinecone-Flask-AWS` project.

## 1. System Overview

The project is an AI-powered medical assistant capable of answering health-related questions. It relies on a Retrieval-Augmented Generation (RAG) approach to synthesize answers grounded in trusted medical text (supplied as PDF documents). 

**Core Technology Stack:**
*   **Web Framework:** Flask (Python)
*   **LLM Provider:** Google Generative AI (`gemini-2.5-flash`)
*   **Orchestration:** LangChain
*   **Vector Database:** Pinecone (Serverless)
*   **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
*   **Frontend:** HTML, CSS, JavaScript (jQuery + Bootstrap)
*   **Deployment:** Dockerized for AWS

## 2. Architecture & Data Flow

The project is broadly split into an **Ingestion Pipeline** (offline processing) and a **Chat Pipeline** (online real-time interactions).

### A. Data Ingestion & Indexing (`store_index.py`)
1.  **Loading:** The system loads raw medical information from PDF files placed in the `data/` directory using LangChain's `DirectoryLoader` and `PyPDFLoader`.
2.  **Preprocessing:** To reduce vector database overhead, the system filters the metadata to keep only the minimal requisite info (`source` paths) via `filter_to_minimal_docs`.
3.  **Chunking:** The parsed documents are split into smaller segments using a `RecursiveCharacterTextSplitter` (chunk size: 500 characters, overlap: 20 characters).
4.  **Embedding & Storage:** The chunks are converted into 384-dimensional dense vectors using the HuggingFace `all-MiniLM-L6-v2` model and pushed directly to a Pinecone vector index named `medical-chatbot`.

### B. Conversational Chat Engine (`app.py`)
1.  **User Request:** Users enter their continuous messages via the browser chat UI, which sends an AJAX `POST` request to the `/get` endpoint.
2.  **Context Retrieval:** The `app.py` script utilizes the `PineconeVectorStore` retriever to establish a similarity search on the incoming query, pulling the top 3 (`k=3`) most relevant document chunks.
3.  **Prompt Assembly:** LangChain’s `ChatPromptTemplate` constructs a comprehensive prompt. It injects:
    *   The core system instructions (from `src/prompt.py`) commanding the AI to be concise and accurate.
    *   The context retrieved from Pinecone.
    *   Past conversations retained by `ConversationBufferMemory`.
4.  **LLM Invocation:** The prompt is sent to Google's Generative AI (`gemini-2.5-flash`) to orchestrate a natural conversational reply based exclusively on the given parameters.
5.  **Response Handling:** The memory is updated with the current user query and AI response pair, and the output text is returned to the frontend.

## 3. Directory Structure

```text
├── .env                  # Environment secrets (e.g., PINECONE_API_KEY, GOOGLE_API_KEY)
├── Dockerfile            # Container configuration (Python 3.10 slim buster)
├── app.py                # Main Flask application and Chat Engine logic
├── requirements.txt      # Project Python dependencies
├── setup.py              # Packaging configuration as a Python module
├── store_index.py        # Independent script to read PDFs and populate Pinecone
├── README.md             # Project documentation
├── data/                 # Directory reserved for source medical PDF documents
├── src/                  # Core processing modules
│   ├── helper.py         # Utility functions: extraction, chunking, and embeddings
│   └── prompt.py         # System rules and prompt template variables
├── static/               # Static assets
│   └── style.css         # UI stylesheet (custom chat theme)
└── templates/
    └── chat.html         # Frontend view utilizing Bootstrap and jQuery
```

## 4. Key Implementation Details

1.  **Memory Implementation:** The bot does not process queries in isolation; it uses LangChain's `ConversationBufferMemory`. This enables realistic conversational flow, context awareness, and follow-up question capabilities.
2.  **Frontend Interactivity:** The frontend uses jQuery's asynchronous (`$.ajax`) mechanisms. When sending a message, a loading/processing experience lets users query instantly without full page reloads. Responses are parsed directly into DOM elements mimicking standard SMS interfaces.
3.  **HuggingFace Embeddings on Device Runtime:** Instead of relying on external APIs for extracting dense vectors, it uses the local lightweight Hugging Face model (`all-MiniLM-L6-v2`) via `sentence-transformers`, reducing costs and increasing processing latency for embedding queries.

## 5. Deployment Information

*   The project specifies `0.0.0.0` as the host mapping in `app.py`, allowing external traffic into the Flask app.
*   The `Dockerfile` prepares a minimal environment using `python:3.10-slim-buster` handling the full install of `requirements.txt`, making it deployment-ready for AWS services such as ECS, Fargate, or standalone EC2 setups.

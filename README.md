# Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS  
A production-ready **Medical AI Chatbot** powered by **Google Gemini**, **LangChain**, **Pinecone Vector DB**, **Flask**, and fully deployed using **Docker + AWS (ECR + EC2) + GitHub Actions CI/CD**.

This chatbot uses **RAG (Retrieval-Augmented Generation)** to provide accurate, context-aware medical information.  
It retrieves medical knowledge from Pinecone and generates final answers using **Gemini 2.5 Pro**.

---
!/screenshots/chatbot-ui-1.png

#  Features  
-  **LLM-powered medical chatbot** using Google Gemini  
-  **RAG pipeline** with Pinecone similarity search  
-  Vector embeddings via Sentence Transformers  
-  **Conversation memory** for contextual chat  
-  Flask-based web interface  
-  Dockerized application  
-  Fully automated **CI/CD to AWS EC2** using GitHub Actions  
-  Fast, scalable, production-grade architecture  

---

#  Tech Stack  
- **Python 3.10**  
- **Flask**  
- **LangChain**  
- **Google Gemini (via langchain-google-genai)**  
- **Sentence Transformers**  
- **Pinecone Vector Database**  
- **Docker**  
- **AWS ECR + EC2 + GitHub Actions**  

---

#  How the Chatbot Works (RAG Pipeline)
- User enters a medical question
- Query is sent to Pinecone to retrieve top-k similar medical chunks
- Retrieved chunks + user query are passed to Gemini
- Gemini generates a medically accurate answer
- Conversation memory stores chat history for contextual dialogue
  
---

# Author
Satyanarayan Mohapatro
Final-year B.Tech CSE  
AI/ML | LLMs | LangChain | Cloud

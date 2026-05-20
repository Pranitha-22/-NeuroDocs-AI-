# 🧠📄 NeuroDocs AI — Multimodal Document Intelligence Platform

An advanced AI-powered document intelligence system that enables users to upload PDFs and interact with them using natural language and voice. Built with FastAPI, FAISS, Sentence Transformers, OCR, and Retrieval-Augmented Generation (RAG) to deliver citation-based answers with image evidence and speech capabilities.

---

## 🧠 Key Highlights

- 📄 Intelligent PDF Question Answering
- 🔍 Semantic Search (FAISS + Sentence Transformers)
- 🧠 Extractive Retrieval-Augmented Generation (RAG)
- 📌 Citation-Based Answers with Source References
- 🖼️ OCR and Automatic Image Extraction
- 🎤 Speech-to-Text for Voice Queries
- 🔊 Text-to-Speech for Audio Responses
- ⚡ Sub-Second Retrieval Performance
- 📥 PDF Export of Responses

---

## 🚀 What Makes This Project Unique

Unlike traditional PDF readers and keyword search tools, this project:

- Understands **natural language questions**
- Performs **semantic search using transformer embeddings**
- Supports **scanned PDFs with OCR**
- Extracts and displays **embedded images**
- Provides **citation-based, explainable answers**
- Enables **voice-based interaction**
- Delivers **near real-time responses**

---

## 🧠 System Architecture

User Uploads PDF  
→ Text Extraction (PyMuPDF)  
→ OCR for Scanned Pages (Tesseract)  
→ Document Chunking (LangChain)  
→ Embedding Generation (Sentence Transformers)  
→ FAISS Vector Index  
→ Semantic Search  
→ Relevant Chunk Retrieval  
→ Extractive RAG Answer Generation  
→ Citation + Image Evidence  
→ Speech-to-Text / Text-to-Speech  
→ Frontend Display

---

## 📊 Core Features

- PDF text extraction and parsing
- OCR for scanned and image-based documents
- Semantic vector search
- Citation-based question answering
- Image extraction and preview
- Voice input (Speech-to-Text)
- Audio output (Text-to-Speech)
- Persistent document indexing
- PDF report generation

---

## 🛠 Tech Stack

### Backend
- FastAPI
- Python
- Uvicorn

### NLP & AI
- Sentence Transformers
- Retrieval-Augmented Generation (RAG)
- FAISS
- LangChain

### Document Processing
- PyMuPDF
- Pillow
- Tesseract OCR
- ReportLab

### Speech AI
- SpeechRecognition
- gTTS
- Web Speech API

### Data Processing
- Pandas
- NumPy

### Frontend
- HTML
- CSS
- JavaScript

---

## ▶️ Run Locally

```bash
git clone https://github.com/<your-username>/NeuroDocs-AI.git
cd NeuroDocs-AI

python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
uvicorn main:app --reload

Open in your browser
http://127.0.0.1:8000

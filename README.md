# 🦜🔗 RAG YouTube Chatbot Using LangChain

A powerful Retrieval-Augmented Generation (RAG) application that allows users to "chat" with any YouTube video. By simply providing a video URL, the system processes the transcript and lets you ask questions, receiving accurate answers based specifically on the video's content.

### Video Demo - https://drive.google.com/file/d/1ZfX0vzb034Nd81FO5uVMzRQTGGsw1OSr/view?usp=sharing

## 📖 Overview

This project leverages LangChain to orchestrate a RAG pipeline. It extracts transcripts from YouTube videos, chunks the text, creates vector embeddings, and stores them in a FAISS vector database. When a user asks a question, the system retrieves the most relevant context from the video and uses a Large Language Model (LLM) to generate a precise answer.

## ✨ Key Features

- **🎥 YouTube Transcript Loading:** Automatically fetches and processes captions from YouTube videos.
- **🧠 Intelligent Text Splitting:** Uses recursive character splitting to maintain context across chunks.
- **🔍 Vector Search:** Implements FAISS (Facebook AI Similarity Search) for high-speed, local similarity search.
- **🤖 LLM Integration:** Supports integration with top-tier LLMs via OpenRouter or Google Gemini.
- **💬 Interactive UI:** Built with Streamlit for a seamless, chat-like experience.
- **💾 Session Memory:** Maintains conversation history within the session.

## 🛠️ Tech Stack

- **Language:** Python
- **Orchestration:** LangChain
- **Frontend:** Streamlit
- **Vector Store:** FAISS
- **Embeddings:** HuggingFace / OpenAI / Google Generative AI
- **Utilities:** youtube-transcript-api, python-dotenv

## 🚀 Getting Started

Follow these instructions to set up the project on your local machine.

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/RaunakSachdeva2004/RAG-Youtube-Chatbot-Using-Langchain.git
cd RAG-Youtube-Chatbot-Using-Langchain
```


## 2. Create a Virtual Environment (Recommended)

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

## 3. macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 4. Set Up Environment Variables
Create a .env file in the root directory of the project. You will need an API key for the LLM service you intend to use.

### .env file content

### If using OpenRouter
OPENROUTER_API_KEY=your_openrouter_api_key_here

### If using Google Gemini
GOOGLE_API_KEY=your_google_api_key_here



#### To use the Chatbot:

- Paste a valid YouTube video URL into the sidebar input field.
- Click the "Process Video" button.
- Wait for the system to download the transcript and build the vector index.
- Once processing is complete, type your question in the chat input box and hit Enter!

  ## 🧩 How It Works
- Ingestion: The app uses YoutubeLoader to get the transcript of the video.

- Splitting: The transcript is divided into smaller chunks using RecursiveCharacterTextSplitter to fit within the LLM's context window.

- Embedding: These chunks are converted into numerical vectors (embeddings).

- Storage: The vectors are stored locally in a FAISS index.

- Retrieval: When you ask a question, the system finds the vectors most similar to your query.

- Generation: The relevant text chunks + your question are sent to the LLM, which generates a natural language response.




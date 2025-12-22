import streamlit as st
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Load environment variables (Make sure .env is in the same folder)
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="RAG YouTube Chatbot", layout="wide")
st.title("🤖 Chat with YouTube Videos")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Settings")
    video_url = st.text_input("Enter YouTube Video URL/ID:")
    process_button = st.button("Process Video")

    st.markdown("---")
    st.markdown("Built by [Raunak Sachdeva](https://github.com/raunaksachdeva2004) | Powered by Groq & Ollama")

# --- Logic to Process Video ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if process_button and video_url:
    with st.spinner("Fetching transcript and processing..."):
        try:
            # 1. Extract Video ID (simple check)
            if "v=" in video_url:
                video_id = video_url.split("v=")[1].split("&")[0]
            else:
                video_id = video_url

            # 2. Fetch Transcript
            transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
            transcript_text = " ".join([t.text for t in transcript_list])

            # 3. Split Text
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.create_documents([transcript_text])

            # 4. Create Embeddings & Vector Store
            # Note: Using your Ollama model. Ensure Ollama is running!
            embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
            vector_store = FAISS.from_documents(chunks, embeddings)
            
            # Save to session state so we don't re-process on every chat
            st.session_state.vector_store = vector_store
            st.success("Video processed successfully! You can now ask questions.")

        except Exception as e:
            st.error(f"Error processing video: {e}")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
user_input = st.chat_input("Ask something about the video...")

if user_input:
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Generate Response
    if st.session_state.vector_store is not None:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve relevant context
                retriever = st.session_state.vector_store.as_retriever()
                docs = retriever.invoke(user_input)
                context_text = "\n\n".join([doc.page_content for doc in docs])

                # Prepare Prompt
                template = """
                You are a helpful assistant. Answer the question based ONLY on the following context:
                {context}
                
                Question: {question}
                """
                prompt = PromptTemplate(template=template, input_variables=["context", "question"])
                
                # Initialize LLM (Using your Groq setup)
                llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)
                
                # Generate Answer
                chain = prompt | llm
                response = chain.invoke({"context": context_text, "question": user_input})
                
                st.markdown(response.content)
                
                # Save assistant response
                st.session_state.messages.append({"role": "assistant", "content": response.content})
    else:
        st.error("Please process a video first!")
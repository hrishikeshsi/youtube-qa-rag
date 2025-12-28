import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="YouTube Q&A with RAG",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF0000;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎬 YouTube Video Q&A with RAG</h1>', unsafe_allow_html=True)
st.markdown("---")

def extract_video_id(url: str) -> str | None:
    """Extract video ID from various YouTube URL formats."""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/shorts\/([^&\n?#]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def load_youtube_transcript(video_url: str) -> list:
    """Load transcript from YouTube video."""
    try:
        loader = YoutubeLoader.from_youtube_url(
            video_url,
            add_video_info=True,
            language=["en", "hi"],
            translation="en"
        )
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error loading transcript: {str(e)}")
        return []

def create_vector_store(documents: list) -> FAISS:
    """Create FAISS vector store from documents."""
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Create FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def create_qa_chain(vector_store: FAISS):
    """Create the QA chain with Gemini."""
    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )
    
    # Custom prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful AI assistant that answers questions based on YouTube video content.
        Use the following context from the video transcript to answer the question.
        If you cannot find the answer in the context, say "I couldn't find specific information about this in the video."
        
        Context from video:
        {context}
        
        Question: {input}
        
        Provide a clear, concise, and helpful answer:"""
    )
    
    # Create document chain and retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    qa_chain = create_retrieval_chain(retriever, document_chain)
    
    return qa_chain

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "video_info" not in st.session_state:
    st.session_state.video_info = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input (optional - can use .env)
    api_key = st.text_input(
        "Google API Key (optional if in .env)",
        type="password",
        help="Enter your Google API key or set it in .env file"
    )
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.markdown("---")
    st.header("📹 Video Input")
    
    # YouTube URL input
    youtube_url = st.text_input(
        "Enter YouTube URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    # Process video button
    if st.button("🔄 Process Video", type="primary", use_container_width=True):
        if not youtube_url:
            st.error("Please enter a YouTube URL")
        elif not os.getenv("GOOGLE_API_KEY"):
            st.error("Please provide a Google API Key")
        else:
            video_id = extract_video_id(youtube_url)
            if not video_id:
                st.error("Invalid YouTube URL")
            else:
                with st.spinner("Loading video transcript..."):
                    documents = load_youtube_transcript(youtube_url)
                    
                    if documents:
                        st.session_state.video_info = {
                            "title": documents[0].metadata.get("title", "Unknown"),
                            "author": documents[0].metadata.get("author", "Unknown"),
                            "length": documents[0].metadata.get("length", 0)
                        }
                        
                        with st.spinner("Creating embeddings and vector store..."):
                            st.session_state.vector_store = create_vector_store(documents)
                            st.session_state.qa_chain = create_qa_chain(st.session_state.vector_store)
                            st.session_state.chat_history = []
                            st.success("✅ Video processed successfully!")
                    else:
                        st.error("Could not load video transcript")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Video info display
    if st.session_state.video_info:
        st.subheader("📺 Current Video")
        st.info(f"""
        **Title:** {st.session_state.video_info['title']}  
        **Author:** {st.session_state.video_info['author']}  
        **Duration:** {st.session_state.video_info['length'] // 60} minutes
        """)
    
    # Chat interface
    st.subheader("💬 Ask Questions")
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
    
    # Question input
    user_question = st.chat_input("Ask a question about the video...")
    
    if user_question:
        if not st.session_state.qa_chain:
            st.error("Please process a YouTube video first!")
        else:
            with st.chat_message("user"):
                st.write(user_question)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.qa_chain.invoke({"input": user_question})
                    answer = result["answer"]
                    st.write(answer)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((user_question, answer))

with col2:
    # Instructions
    st.subheader("📖 How to Use")
    st.markdown("""
    1. **Enter your Google API Key** in the sidebar (or set in `.env` file)
    2. **Paste a YouTube URL** in the sidebar
    3. **Click "Process Video"** to load the transcript
    4. **Ask questions** about the video content!
    
    ---
    
    **Supported URLs:**
    - `youtube.com/watch?v=...`
    - `youtu.be/...`
    - `youtube.com/shorts/...`
    
    ---
    
    **Tips:**
    - Ask specific questions for better answers
    - The AI uses the video transcript for context
    - Works best with videos that have transcripts/captions
    """)
    
    # Example questions
    if st.session_state.video_info:
        st.subheader("💡 Example Questions")
        st.markdown("""
        - What is this video about?
        - What are the main topics covered?
        - Can you summarize the key points?
        - What did they say about [specific topic]?
        """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Built with LangChain, FAISS, Google Gemini & Streamlit</p>",
    unsafe_allow_html=True
)

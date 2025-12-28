import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
import os
import re


load_dotenv()


st.set_page_config(
    page_title="YouTube Q&A RAG - With Evaluation",
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
        margin-bottom: 1rem;
    }
    .chunk-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background-color: #e8f4ea;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)



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
    """Load transcript from YouTube video using youtube-transcript-api directly."""
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("Could not extract video ID from URL")
            return []

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
     
            try:
                transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
            except:

                try:
                    transcript = transcript_list.find_transcript(['hi', 'es', 'fr', 'de'])
                    transcript = transcript.translate('en')
                except:
         
                    transcript = transcript_list.find_generated_transcript(['en'])
            
            transcript_data = transcript.fetch()
            
        except Exception as e:
 
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi', 'en-US'])
        

        full_text = " ".join([entry['text'] for entry in transcript_data])
        

        document = Document(
            page_content=full_text,
            metadata={
                "source": video_url,
                "video_id": video_id,
                "title": f"YouTube Video: {video_id}",
                "author": "Unknown",
                "length": sum([entry.get('duration', 0) for entry in transcript_data])
            }
        )
        
        return [document]
        
    except Exception as e:
        st.error(f"Error loading transcript: {str(e)}")
        st.info("💡 Tip: Make sure the video has captions/subtitles enabled.")
        return []

def create_chunks(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    CHUNKING: Split documents into smaller, overlapping chunks.
    
    Why chunking matters:
    - LLMs have context limits
    - Smaller chunks = more precise retrieval
    - Overlap ensures context isn't lost at boundaries
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks: list) -> FAISS:
    """
    EMBEDDINGS + VECTOR STORE: Convert text to vectors and store in FAISS.
    
    Process:
    1. Each chunk → Google Embedding Model → 768-dim vector
    2. Vectors stored in FAISS index for fast similarity search
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def retrieve_relevant_chunks(vector_store: FAISS, query: str, k: int = 4) -> list:
    """
    VECTOR SEARCH: Find most similar chunks to the query.
    
    Uses cosine similarity to find k nearest neighbors.
    Returns chunks with similarity scores.
    """
    docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
    return docs_with_scores

def generate_answer(llm, context: str, question: str) -> str:
    """
    GROUNDED GENERATION: Generate answer based ONLY on retrieved context.
    
    The prompt explicitly instructs the model to:
    - Only use provided context
    - Admit when information isn't available
    - Avoid hallucination
    """
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful AI assistant. Answer the question based ONLY on the provided context.
        
IMPORTANT RULES:
1. ONLY use information from the context below
2. If the answer is not in the context, say "This information is not available in the video transcript."
3. Do NOT make up or assume any information
4. Quote relevant parts when possible
5. Be concise and accurate

CONTEXT FROM VIDEO TRANSCRIPT:
{context}

QUESTION: {question}

ANSWER (based only on the context above):"""
    )
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer


def calculate_relevance_score(query: str, chunk_text: str, similarity_score: float) -> dict:
    """
    Calculate relevance metrics for a retrieved chunk.
    
    FAISS returns L2 distance (lower = more similar)
    We convert to a 0-100 relevance score.
    """
  
    relevance_score = max(0, 100 - (similarity_score * 10))
    

    query_words = set(query.lower().split())
    chunk_words = set(chunk_text.lower().split())
    keyword_overlap = len(query_words & chunk_words) / max(len(query_words), 1)
    
    return {
        "relevance_score": round(relevance_score, 2),
        "l2_distance": round(similarity_score, 4),
        "keyword_overlap": round(keyword_overlap * 100, 2),
        "chunk_length": len(chunk_text)
    }

def evaluate_answer(answer: str, context: str) -> dict:
    """
    Evaluate answer quality for hallucination detection.
    
    Checks:
    - Is answer grounded in context?
    - Answer length and completeness
    - Explicit uncertainty markers
    """
    uncertainty_phrases = [
        "not available", "not mentioned", "doesn't mention",
        "no information", "cannot find", "not in the video",
        "not specified", "unclear from"
    ]
    has_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)
    
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    grounding_ratio = len(answer_words & context_words) / max(len(answer_words), 1)
    
    return {
        "answer_length": len(answer),
        "word_count": len(answer.split()),
        "grounding_ratio": round(grounding_ratio * 100, 2),
        "has_uncertainty_marker": has_uncertainty,
        "potential_hallucination_risk": "Low" if grounding_ratio > 0.3 or has_uncertainty else "Medium"
    }



EVALUATION_QUESTIONS = [
    "What is the main topic of this video?",
    "What are the key points discussed?",
    "Who is the speaker or presenter?",
    "What problems or challenges are mentioned?",
    "What solutions or recommendations are provided?",
    "Are there any specific examples given?",
    "What is the conclusion or final message?",
    "What tools or technologies are mentioned?",
    "What is the target audience for this content?",
    "Are there any statistics or data mentioned?"
]



if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "video_info" not in st.session_state:
    st.session_state.video_info = None
if "raw_transcript" not in st.session_state:
    st.session_state.raw_transcript = ""
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = []


with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input(
        "Google API Key",
        type="password",
        help="Get from: https://aistudio.google.com/app/apikey"
    )
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.markdown("---")
    

    st.header("🔧 RAG Parameters")
    chunk_size = st.slider("Chunk Size", 200, 2000, 1000, 100,
                          help="Size of each text chunk in characters")
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50,
                             help="Overlap between consecutive chunks")
    top_k = st.slider("Top-K Retrieval", 1, 10, 4,
                     help="Number of chunks to retrieve")
    
    st.markdown("---")
    st.header("📹 Video Input")
    
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://youtube.com/watch?v=..."
    )
    
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
                with st.spinner("Step 1/3: Loading transcript..."):
                    documents = load_youtube_transcript(youtube_url)
                
                if documents:
                    st.session_state.raw_transcript = documents[0].page_content
                    st.session_state.video_info = {
                        "title": documents[0].metadata.get("title", "Unknown"),
                        "author": documents[0].metadata.get("author", "Unknown"),
                        "length": documents[0].metadata.get("length", 0)
                    }
                    
                    with st.spinner("Step 2/3: Chunking transcript..."):
                        st.session_state.chunks = create_chunks(documents, chunk_size, chunk_overlap)
                    
                    with st.spinner("Step 3/3: Creating embeddings & vector store..."):
                        st.session_state.vector_store = create_vector_store(st.session_state.chunks)
                    
                    st.session_state.evaluation_results = []
                    st.success(f"✅ Processed! Created {len(st.session_state.chunks)} chunks")
                else:
                    st.error("Could not load transcript")
   
    if st.session_state.chunks:
        st.markdown("---")
        st.header("📊 Processing Stats")
        st.metric("Total Chunks", len(st.session_state.chunks))
        st.metric("Avg Chunk Size", f"{sum(len(c.page_content) for c in st.session_state.chunks) // len(st.session_state.chunks)} chars")
        st.metric("Transcript Length", f"{len(st.session_state.raw_transcript):,} chars")


st.markdown('<h1 class="main-header">🎬 YouTube Q&A RAG System</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>With Evaluation & Transparency Features</p>", unsafe_allow_html=True)
st.markdown("---")


tab1, tab2, tab3, tab4 = st.tabs(["💬 Q&A Interface", "🔍 RAG Pipeline Visualization", "📝 Evaluation Suite", "📚 Technical Details"])


with tab1:
    if st.session_state.video_info:
        st.info(f"**📺 Video:** {st.session_state.video_info['title']} by {st.session_state.video_info['author']}")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Ask a Question")
        user_question = st.text_input("Your question:", placeholder="What is this video about?")
        
        if st.button("🔍 Get Answer", type="primary") and user_question:
            if not st.session_state.vector_store:
                st.error("Please process a video first!")
            else:
                with st.spinner("Processing..."):
                 
                    retrieved = retrieve_relevant_chunks(
                        st.session_state.vector_store, 
                        user_question, 
                        k=top_k
                    )
                    
                    context = "\n\n---\n\n".join([doc.page_content for doc, score in retrieved])
                    
      
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash",
                        google_api_key=os.getenv("GOOGLE_API_KEY"),
                        temperature=0.2
                    )
                    answer = generate_answer(llm, context, user_question)
                    
                  
                    st.markdown("### 🤖 Answer")
                    st.success(answer)
                    
                   
                    eval_metrics = evaluate_answer(answer, context)
                    
                    st.markdown("### 📊 Answer Metrics")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Word Count", eval_metrics["word_count"])
                    m2.metric("Grounding %", f"{eval_metrics['grounding_ratio']}%")
                    m3.metric("Hallucination Risk", eval_metrics["potential_hallucination_risk"])
    
    with col2:
        st.subheader("🎯 Retrieved Context")
        st.caption("TRANSPARENCY: These are the exact chunks used to generate the answer")
        
        if user_question and st.session_state.vector_store:
            retrieved = retrieve_relevant_chunks(st.session_state.vector_store, user_question, k=top_k)
            
            for i, (doc, score) in enumerate(retrieved):
                relevance = calculate_relevance_score(user_question, doc.page_content, score)
                
                with st.expander(f"📄 Chunk {i+1} | Relevance: {relevance['relevance_score']}%", expanded=(i==0)):
                    st.markdown(f"**L2 Distance:** {relevance['l2_distance']}")
                    st.markdown(f"**Keyword Overlap:** {relevance['keyword_overlap']}%")
                    st.markdown("**Content:**")
                    st.markdown(f"```\n{doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}\n```")


with tab2:
    st.subheader("🔍 RAG Pipeline - Step by Step")
    
    st.markdown("""
    This visualization shows each step of the RAG (Retrieval Augmented Generation) pipeline:
    """)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("### 1️⃣ LOAD")
        st.markdown("**YouTube Transcript**")
        st.markdown("Extract text from video captions/subtitles")
        if st.session_state.raw_transcript:
            st.success(f"✅ {len(st.session_state.raw_transcript):,} chars")
    
    with col2:
        st.markdown("### 2️⃣ CHUNK")
        st.markdown("**Text Splitting**")
        st.markdown(f"Split into overlapping chunks ({chunk_size} chars, {chunk_overlap} overlap)")
        if st.session_state.chunks:
            st.success(f"✅ {len(st.session_state.chunks)} chunks")
    
    with col3:
        st.markdown("### 3️⃣ EMBED")
        st.markdown("**Vectorization**")
        st.markdown("Convert text to 768-dim vectors using Google Embeddings")
        if st.session_state.vector_store:
            st.success("✅ Embedded")
    
    with col4:
        st.markdown("### 4️⃣ STORE")
        st.markdown("**FAISS Index**")
        st.markdown("Store vectors for fast similarity search")
        if st.session_state.vector_store:
            st.success("✅ Indexed")
    
    with col5:
        st.markdown("### 5️⃣ RETRIEVE")
        st.markdown("**Vector Search**")
        st.markdown(f"Find top-{top_k} most similar chunks to query")
        st.info("Ready")
    
    st.markdown("---")
    

    if st.session_state.chunks:
        st.subheader("📦 Sample Chunks (Chunking Demonstration)")
        
        for i, chunk in enumerate(st.session_state.chunks[:3]):
            with st.expander(f"Chunk {i+1} of {len(st.session_state.chunks)} | {len(chunk.page_content)} characters"):
                st.text(chunk.page_content)
        
        if len(st.session_state.chunks) > 3:
            st.caption(f"... and {len(st.session_state.chunks) - 3} more chunks")

with tab3:
    st.subheader("📝 RAG Evaluation Suite")
    
    st.markdown("""
    This section allows systematic evaluation of the RAG system's performance.
    
    **What we're testing:**
    - ✅ Retrieved chunk relevance
    - ✅ Answer completeness  
    - ✅ Hallucination detection
    - ✅ Grounded generation
    """)
    
    if not st.session_state.vector_store:
        st.warning("⚠️ Please process a video first to run evaluations")
    else:
        st.markdown("### 🎯 Evaluation Questions")
       
        selected_questions = st.multiselect(
            "Select questions to evaluate:",
            EVALUATION_QUESTIONS,
            default=EVALUATION_QUESTIONS[:3]
        )
       
        custom_q = st.text_input("Or add a custom question:")
        if custom_q:
            selected_questions.append(custom_q)
        
        if st.button("🚀 Run Evaluation", type="primary"):
            st.session_state.evaluation_results = []
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.2
            )
            
            progress = st.progress(0)
            
            for idx, question in enumerate(selected_questions):
                with st.spinner(f"Evaluating: {question[:50]}..."):
                 
                    retrieved = retrieve_relevant_chunks(st.session_state.vector_store, question, k=top_k)
                    context = "\n\n".join([doc.page_content for doc, score in retrieved])
                    
                    
                    answer = generate_answer(llm, context, question)
                    
                    
                    chunk_relevances = [
                        calculate_relevance_score(question, doc.page_content, score)
                        for doc, score in retrieved
                    ]
                    answer_eval = evaluate_answer(answer, context)
                    
                    st.session_state.evaluation_results.append({
                        "question": question,
                        "answer": answer,
                        "retrieved_chunks": [(doc.page_content[:200], score) for doc, score in retrieved],
                        "chunk_relevances": chunk_relevances,
                        "answer_evaluation": answer_eval,
                        "avg_relevance": sum(r["relevance_score"] for r in chunk_relevances) / len(chunk_relevances)
                    })
                
                progress.progress((idx + 1) / len(selected_questions))
            
            st.success("✅ Evaluation complete!")
        
       
        if st.session_state.evaluation_results:
            st.markdown("---")
            st.markdown("### 📊 Evaluation Results")
            
           
            avg_relevance = sum(r["avg_relevance"] for r in st.session_state.evaluation_results) / len(st.session_state.evaluation_results)
            avg_grounding = sum(r["answer_evaluation"]["grounding_ratio"] for r in st.session_state.evaluation_results) / len(st.session_state.evaluation_results)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Chunk Relevance", f"{avg_relevance:.1f}%")
            m2.metric("Avg Answer Grounding", f"{avg_grounding:.1f}%")
            m3.metric("Questions Evaluated", len(st.session_state.evaluation_results))
            
            st.markdown("---")
            
        
            for i, result in enumerate(st.session_state.evaluation_results):
                with st.expander(f"Q{i+1}: {result['question'][:60]}...", expanded=(i==0)):
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**🤖 Generated Answer:**")
                        st.info(result["answer"])
                        
                        st.markdown("**📄 Retrieved Chunks:**")
                        for j, (chunk, score) in enumerate(result["retrieved_chunks"]):
                            rel = result["chunk_relevances"][j]
                            st.markdown(f"*Chunk {j+1}* (Relevance: {rel['relevance_score']}%, Distance: {rel['l2_distance']})")
                            st.text(chunk + "...")
                    
                    with col2:
                        st.markdown("**📊 Metrics:**")
                        st.metric("Avg Chunk Relevance", f"{result['avg_relevance']:.1f}%")
                        st.metric("Answer Grounding", f"{result['answer_evaluation']['grounding_ratio']}%")
                        st.metric("Hallucination Risk", result['answer_evaluation']['potential_hallucination_risk'])
                        st.metric("Word Count", result['answer_evaluation']['word_count'])


with tab4:
    st.subheader("📚 Technical Implementation Details")
    
    st.markdown("""
    ## RAG Fundamentals Demonstrated
    
    ### 1️⃣ Chunking
    **What:** Breaking large documents into smaller, overlapping pieces.
    
    **Why it matters:**
    - LLMs have limited context windows
    - Smaller chunks enable more precise retrieval
    - Overlap prevents losing context at boundaries
    
    **Our implementation:**
    ```python
    RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Characters per chunk
        chunk_overlap=200,    # Overlap between chunks
        separators=["\\n\\n", "\\n", ". ", " ", ""]
    )
    ```
    
    ---
    
    ### 2️⃣ Embeddings
    **What:** Converting text into dense numerical vectors.
    
    **Why it matters:**
    - Captures semantic meaning, not just keywords
    - Enables similarity comparison between texts
    - Foundation for vector search
    
    **Our implementation:**
    ```python
    GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"  # 768-dimensional vectors
    )
    ```
    
    ---
    
    ### 3️⃣ Vector Search (FAISS)
    **What:** Finding most similar chunks to a query using vector similarity.
    
    **Why it matters:**
    - Fast approximate nearest neighbor search
    - Scales to millions of vectors
    - Returns relevant context for generation
    
    **Our implementation:**
    ```python
    # Index creation
    FAISS.from_documents(chunks, embeddings)
    
    # Retrieval with scores
    vector_store.similarity_search_with_score(query, k=4)
    ```
    
    ---
    
    ### 4️⃣ Grounded Generation
    **What:** Generating answers based ONLY on retrieved context.
    
    **Why it matters:**
    - Prevents hallucination
    - Answers are traceable to source
    - User can verify accuracy
    
    **Our implementation:**
    ```python
    prompt = \"\"\"Answer based ONLY on the provided context.
    If the answer is not in the context, say "This information 
    is not available in the video transcript."
    
    CONTEXT: {context}
    QUESTION: {question}\"\"\"
    ```
    
    ---
    
    ### 5️⃣ Transparency
    **What:** Showing users exactly what context was used.
    
    **Why it matters:**
    - Builds trust in the system
    - Enables verification of answers
    - Helps identify retrieval failures
    
    **Our implementation:**
    - Display retrieved chunks with relevance scores
    - Show L2 distance and keyword overlap
    - Highlight potential hallucination risks
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## Evaluation Metrics Explained
    
    | Metric | Description | Good Value |
    |--------|-------------|------------|
    | **Relevance Score** | How similar retrieved chunk is to query (0-100) | > 70% |
    | **L2 Distance** | Raw FAISS distance (lower = more similar) | < 1.0 |
    | **Keyword Overlap** | % of query words found in chunk | > 30% |
    | **Grounding Ratio** | % of answer words found in context | > 30% |
    | **Hallucination Risk** | Likelihood of made-up information | Low |
    """)

st.markdown("---")
st.markdown("""
<p style='text-align: center; color: gray;'>
    <strong>YouTube Q&A RAG System</strong><br>
    Built with LangChain | FAISS | Google Gemini 2.0 Flash | Streamlit<br>
    <em>Demonstrating: Chunking • Embeddings • Vector Search • Grounded Generation • Transparency</em>
</p>
""", unsafe_allow_html=True)

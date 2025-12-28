# 🎬 YouTube Q&A RAG Application

A powerful Retrieval Augmented Generation (RAG) application that lets you ask questions about any YouTube video using AI. Built with LangChain, FAISS, Google Gemini 2.0 Flash, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red.svg)

## ✨ Features

- 📹 **YouTube Transcript Loading** - Automatically extracts transcripts from YouTube videos
- 🧠 **Smart Chunking** - Splits transcripts into optimal chunks for retrieval
- 🔍 **FAISS Vector Search** - Fast and efficient similarity search
- 🤖 **Google Gemini 2.0 Flash** - State-of-the-art LLM for generating answers
- 💬 **Interactive Chat Interface** - Ask questions in a conversational manner
- 📊 **Video Metadata Display** - Shows video title, author, and duration

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  YouTube URL    │────▶│  Transcript      │────▶│  Text Splitter  │
│                 │     │  Loader          │     │  (Chunking)     │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  User Question  │────▶│  FAISS Vector    │◀────│  Google         │
│                 │     │  Store           │     │  Embeddings     │
└────────┬────────┘     └────────┬─────────┘     └─────────────────┘
         │                       │
         │                       ▼
         │              ┌──────────────────┐
         └─────────────▶│  Gemini 2.0      │────▶  Answer
                        │  Flash LLM       │
                        └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Google API Key (for Gemini and Embeddings)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd RAG
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your Google API Key
   # Get your key from: https://aistudio.google.com/app/apikey
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 🔑 Getting Your Google API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and paste it in your `.env` file

## 📖 How to Use

1. **Enter your Google API Key** in the sidebar (or set it in the `.env` file)
2. **Paste a YouTube URL** - Supported formats:
   - `https://www.youtube.com/watch?v=VIDEO_ID`
   - `https://youtu.be/VIDEO_ID`
   - `https://www.youtube.com/shorts/VIDEO_ID`
3. **Click "Process Video"** to load and process the transcript
4. **Ask questions** about the video content in the chat interface

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| LLM | Google Gemini 2.0 Flash | Generate answers |
| Embeddings | Google Embedding Model | Vectorize text |
| Vector Store | FAISS | Store and retrieve embeddings |
| Framework | LangChain | RAG pipeline orchestration |
| UI | Streamlit | Web interface |
| Transcript | youtube-transcript-api | Extract video transcripts |

## 📁 Project Structure

```
RAG/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── .env                # Your environment variables (gitignored)
├── .gitignore          # Git ignore file
└── README.md           # This file
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Your Google API key for Gemini and Embeddings | Yes |

### Customization Options

You can modify these parameters in `app.py`:

- **Chunk Size**: Default 1000 characters
- **Chunk Overlap**: Default 200 characters
- **Number of Retrieved Chunks**: Default 4 (k=4)
- **Temperature**: Default 0.3 for focused responses

## 🐛 Troubleshooting

### Common Issues

1. **"Could not load video transcript"**
   - Ensure the video has captions/subtitles enabled
   - Try a different video

2. **"Invalid API Key"**
   - Verify your Google API key is correct
   - Ensure the Gemini API is enabled in your Google Cloud project

3. **"Rate limit exceeded"**
   - Wait a few minutes and try again
   - Consider using a paid API tier

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the amazing RAG framework
- [Google AI](https://ai.google.dev/) for Gemini and Embeddings
- [Streamlit](https://streamlit.io/) for the easy-to-use UI framework
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search

---

<p align="center">Built with ❤️ using LangChain, FAISS, Google Gemini & Streamlit</p>

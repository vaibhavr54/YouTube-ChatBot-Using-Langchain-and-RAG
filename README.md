# 🧠 YouTube RAG Chatbot (LLaMA 3.3 + FAISS + LangChain)

### 🚀 Chat with YouTube Videos Using AI-Powered Retrieval-Augmented Generation (RAG)

A **Streamlit-based conversational AI** application that enables users to interact with **YouTube video transcripts** using **LLaMA 3.3 (via OpenRouter)** and **FAISS vector search** for context-aware Q&A and summarization.

---

## 🧩 Features

- 🎬 **Fetch YouTube transcripts automatically** (official + auto captions)
- ⚙️ **Embed and index** video content using FAISS for semantic search
- 🧠 **Context-aware Q&A** powered by LLaMA 3.3 via OpenRouter
- 📜 **Detailed video summarization**
- 💾 **Download full transcript** as `.txt`
- 💬 **Memory-aware conversations** (conversation context retained)
- 🖥️ **Modern two-column Streamlit UI** (responsive & user-friendly)

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend/UI** | [Streamlit](https://streamlit.io/) |
| **LLM Backend** | [LLaMA 3.3 (8B Instruct)](https://openrouter.ai/) via OpenRouter |
| **Vector Store** | [FAISS](https://faiss.ai/) |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Framework** | [LangChain](https://www.langchain.com/) |
| **YouTube Integration** | `youtube-transcript-api`, `yt_dlp` |
| **Language** | Python 3.9+ |

---

## 🧑‍💻 Setup Instructions

### 1️⃣ Clone this Repository
```bash
git clone https://github.com/vaibhavr54/YouTube-ChatBot-Using-Langchain-and-RAG.git
cd YouTube-ChatBot-Using-Langchain-and-RAG
````

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate      # On Windows
# or
source venv/bin/activate   # On macOS/Linux
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App

```bash
streamlit run app.py
```

Then open the local URL displayed in your terminal (usually [http://localhost:8501](http://localhost:8501)).

---

## 🔑 OpenRouter API Key Setup

This project uses **OpenRouter** to access the LLaMA 3.3 model.

1. Visit [https://openrouter.ai](https://openrouter.ai)
2. Log in and go to **Account → API Keys**
3. Generate a new API key
4. Enter it in the **Streamlit app input field** when prompted

> 🔒 Your key is never stored — only used during your session.

---

## ⚙️ Environment Variables (Optional)

If you prefer, you can set your API key via an environment variable:

```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

or on Windows PowerShell:

```bash
setx OPENROUTER_API_KEY "your_api_key_here"
```

Then modify `app.py` to read from it if not entered manually.

---

## 🧠 Usage

1. Paste any **YouTube video link**
2. The app automatically **fetches and parses** the transcript
3. Click **“Download Transcript”** to save it locally
4. Use the **chat interface** to:

   * Ask contextual questions
   * Summarize the full video
   * Explore concepts discussed in the content

---

## 🖼️ Preview


---

## 🧪 Example Queries

* “Summarize the video in 5 bullet points.”
* “What are the key takeaways from this video?”
* “Explain the concept discussed at 5:30 mark.”
* “Who is the speaker and what is the topic about?”

---

## 🧱 Project Structure

```
📁 YouTube-ChatBot-Using-Langchain-and-RAG
├── app.py                     # Main Streamlit app
├── requirements.txt           # Dependencies
├── faiss_youtube_index/       # Vector index (auto-generated)
├── .gitignore
└── README.md
```

---

## 🤝 Contributing

Pull requests are welcome!
If you'd like to improve the UI, optimize retrieval logic, or add support for more models — open a PR or issue.

---

## 🌟 Acknowledgements

* [LangChain](https://www.langchain.com/) for the framework
* [FAISS](https://faiss.ai/) for vector similarity search
* [OpenRouter](https://openrouter.ai/) for providing access to advanced LLMs
* [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/) for transcript fetching

---

**Made with ❤️ by [Vaibhav Rakshe](https://github.com/vaibhavr54)**

> “Let your videos talk back intelligently.” 🎬💬

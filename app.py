import streamlit as st
import os, re, json, requests, xml.etree.ElementTree as ET
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="🧠 YouTube RAG Chatbot", layout="wide")

st.title("🧠 YouTube RAG Chatbot (LLaMA 3.3 + FAISS + LangChain)")
st.caption(
    "A Streamlit-based conversational AI that integrates **LLaMA 3.3 via OpenRouter** "
    "with **FAISS-based vector retrieval** to perform context-aware Q&A and summarization "
    "on **YouTube video transcripts**."
)

# Create two-column layout
left_col, right_col = st.columns([1.1, 1.3])


# -------------------- LEFT COLUMN: Setup & Transcript --------------------
with left_col:
    st.subheader("⚙️ Configuration & Transcript")

    # --- API Key Input ---
    api_key = st.text_input("🔑 Enter your OpenRouter API Key:", type="password")
    if not api_key:
        st.warning("Please enter your OpenRouter API key to continue.")
        st.stop()

    # --- YouTube URL Input ---
    video_url = st.text_input("🎥 Enter YouTube Video URL:")

    def extract_video_id(url):
        pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
        match = re.search(pattern, url)
        return match.group(1) if match else None

    video_id = extract_video_id(video_url)

    if video_id:
        video_embed = f"https://www.youtube.com/embed/{video_id}"
        st.markdown(
            f'<iframe width="100%" height="300" src="{video_embed}" frameborder="0" allowfullscreen></iframe>',
            unsafe_allow_html=True,
        )
    else:
        if video_url:
            st.error("⚠️ Invalid YouTube URL. Please enter a valid one.")
        st.stop()

    # -------------------- CACHED FUNCTIONS --------------------

    @st.cache_data(show_spinner=True)
    def fetch_transcript(video_id, video_url):
        """Fetches transcript using YouTubeTranscriptAPI or yt_dlp as fallback."""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            st.success("✅ Official transcript fetched successfully!")
            return " ".join([t["text"] for t in transcript])
        except Exception:
            try:
                transcript_obj = YouTubeTranscriptApi.list_transcripts(video_id)
                available_transcript = transcript_obj.find_transcript(["en", "en-US"])
                transcript = available_transcript.fetch()
                st.success("✅ Transcript found via alternative fetch!")
                return " ".join([t["text"] for t in transcript])
            except Exception:
                try:
                    st.warning("⚠️ Official transcript not found, trying auto captions via yt_dlp...")

                    ydl_opts = {
                        "skip_download": True,
                        "quiet": True,
                        "writeautomaticsub": True,
                        "subtitleslangs": ["en"],
                        "writesubtitles": True,
                    }

                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(video_url, download=False)
                        subtitles = info.get("subtitles") or info.get("automatic_captions")

                        if not subtitles or "en" not in subtitles:
                            raise ValueError("No English captions found via yt_dlp.")

                        caption_url = subtitles["en"][0].get("url")
                        if not caption_url:
                            raise ValueError("Caption URL missing from yt_dlp output.")

                        r = requests.get(caption_url)
                        r.raise_for_status()
                        raw_text = r.text.strip()

                        # Handle .json3 captions
                        if raw_text.startswith("{") and '"events":' in raw_text:
                            data = json.loads(raw_text)
                            segments = []
                            for event in data.get("events", []):
                                for seg in event.get("segs", []) if "segs" in event else []:
                                    if "utf8" in seg:
                                        text = seg["utf8"].strip()
                                        if text and not text.startswith("\n"):
                                            segments.append(text)
                            full_text = " ".join(segments)

                        # Handle XML captions
                        elif raw_text.startswith("<"):
                            xml_root = ET.fromstring(raw_text)
                            transcript_texts = [
                                "".join(el.itertext()).replace("\n", " ").strip()
                                for el in xml_root.findall("text")
                            ]
                            full_text = " ".join(transcript_texts)

                        # Handle VTT or plain text
                        else:
                            lines = [
                                line.strip() for line in raw_text.splitlines()
                                if line.strip() and not re.match(r"^\d{1,2}:\d{2}", line)
                            ]
                            full_text = " ".join(lines)

                        st.info("✅ Auto captions successfully downloaded and parsed via yt_dlp.")
                        return full_text

                except Exception as e:
                    st.error(f"⚠️ Could not fetch transcript automatically: {e}")
                    st.stop()

    @st.cache_resource(show_spinner=True)
    def build_vectorstore(full_text, api_key, video_id):
        """Builds or loads a cached FAISS vectorstore for the given video."""
        store_path = f"faiss_store_{video_id}"

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(full_text)

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=api_key
        )

        if os.path.exists(store_path):
            st.info("📂 Loading cached FAISS index from disk...")
            vectorstore = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        else:
            st.info("🧠 Building new FAISS vectorstore...")
            vectorstore = FAISS.from_texts(texts, embeddings)
            vectorstore.save_local(store_path)
            st.success("✅ FAISS index saved for future use.")

        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
        return retriever


    # -------------------- EXECUTION --------------------
    full_text = fetch_transcript(video_id, video_url)
    if not full_text.strip():
        st.error("Transcript appears empty. Please try another video.")
        st.stop()

    st.success("✅ Transcript processed successfully!")

    # --- Download Option ---
    st.download_button(
        label="📄 Download Full Transcript (.txt)",
        data=full_text,
        file_name=f"youtube_transcript_{video_id}.txt",
        mime="text/plain",
    )

    st.divider()

    # --- Embedding & Vectorization ---
    st.subheader("🔍 Embedding & Vectorization")
    retriever = build_vectorstore(full_text, api_key, video_id)
    st.info("✅ FAISS index ready for retrieval.")


# -------------------- RIGHT COLUMN: Chat Interface --------------------
with right_col:
    st.subheader("💬 Chat with Your Video")

    llm = ChatOpenAI(
        model="meta-llama/llama-3.3-8b-instruct:free",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        temperature=0.3
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    query = st.text_input("🧠 Ask a question about the video:", placeholder="e.g. What is the main topic?")
    col1, col2 = st.columns(2)
    with col1:
        summarize = st.button("Summarize Full Video")
    with col2:
        ask = st.button("Ask")

    if summarize:
        with st.spinner("Generating detailed video summary..."):
            try:
                response = qa_chain.invoke({"question": "Summarize the full video in detail."})
                st.success("🧩 Summary:")
                st.write(response["answer"])
            except Exception as e:
                st.error(f"Error during summarization: {e}")

    elif ask:
        if not query.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    response = qa_chain.invoke({"question": query})
                    st.success("💡 Response:")
                    st.write(response["answer"])
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    
import streamlit as st
import os
import tempfile
import json
from dotenv import load_dotenv
from gtts import gTTS
from groq import Groq
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
import magic
from langchain.prompts import PromptTemplate
# -------------------------------
# Config
# -------------------------------
HISTORY_FILE = "podcast_history.json"
USER_CREDENTIALS = {"karthik": "mypassword123", "alice": "alicepass"}

# -------------------------------
# Load chat/podcast history
# -------------------------------
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):  # legacy support
                return {"default_user": data}
            return data
    return {}

def save_history(all_history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_history, f, ensure_ascii=False, indent=4)

# -------------------------------
# Session state init
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "podcasts" not in st.session_state:
    st.session_state.podcasts = []
if "mes" not in st.session_state:
    st.session_state.mes = []
if "session_podcasts" not in st.session_state:
    st.session_state.session_podcasts = []

# -------------------------------
# Environment & LLM setup
# -------------------------------
load_dotenv()
GROQ_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_KEY:
    st.error("⚠️ GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    groq_api_key=GROQ_KEY,
    streaming=True
)
client = Groq(api_key=GROQ_KEY)

api_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wiki)
api_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_arxiv)
search = DuckDuckGoSearchRun(name="search")
tools = [wiki, arxiv, search]

# -------------------------------
# Login Page
# -------------------------------
def show_login():
    st.title("🔐 PodMate Login")
    with st.form("login_form"):
        username_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if username_input in USER_CREDENTIALS and USER_CREDENTIALS[username_input] == password_input:
                st.session_state.logged_in = True
                st.session_state.username = username_input

                # Load all user history
                all_history = load_history()
                st.session_state.podcasts = all_history.get(username_input, [])
                st.session_state.session_podcasts = []  # start clean for this login
                st.session_state.mes = [{"role": "ai", "content": "Hello! I’m your AI research assistant 🤖."}]

                st.success(f"Welcome, {username_input}!")

            else:
                st.error("❌ Invalid username or password")

# -------------------------------
# Main App
# -------------------------------
def show_main_app():
    username = st.session_state.username
    st.sidebar.success(f"Logged in as {username}")
    st.set_page_config(page_title="PodMate 🎧", page_icon="🎙️", layout="wide")
    st.title("🎧 AI-Powered Podcast Generator & Research Assistant")

    tab1, tab2, tab3 = st.tabs(["🎙️ Podcast Generator", "🤖 Research Assistant", "🗂️ Podcast History"])

    # ---------------- TAB 1: Podcast Generator ----------------
    # ---------------- TAB 1: Podcast Generator ----------------
    # ---------------- TAB 1: Podcast Generator ----------------
    with tab1:
        st.header("Generate Podcasts from Notes or Textbooks")

        uploaded_file = st.file_uploader("📄 Upload a PDF or TXT file", type=["pdf", "txt"])

        if uploaded_file:
            mime = magic.from_buffer(uploaded_file.read(2048), mime=True)
            uploaded_file.seek(0)
            if not (mime in ["application/pdf", "text/plain"]):
                st.error("❌ Invalid file type. Please upload a valid PDF or TXT file.")
                st.stop()

        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read())
                file_path = tmp.name
            st.success("✅ File uploaded successfully!")

            # File size check (limit: 10 MB)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > 10:
                st.error("❌ File too large! Please upload a file smaller than 10 MB.")
                st.stop()

            if st.button("🚀 Generate Podcast"):
                with st.spinner("🧠 Summarizing and generating your podcast..."):
                    # Load text
                    if uploaded_file.name.endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        text_content = " ".join([d.page_content for d in docs])
                    else:
                        with open(file_path, "r", encoding="utf-8") as f:
                            text_content = f.read()
                        docs = [{"page_content": text_content}]

                    # Split text into chunks
                    # Split text into chunks
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

# For PyPDFLoader, docs is a list of Document objects
                    chunks = splitter.create_documents([d.page_content for d in docs])

                    # Decide summarization strategy
                    word_count = len(text_content.split())
                    #st.info(f"📊 Document contains approximately **{word_count:,} words**.")

                    if word_count < 5000:
                        #st.write("🧩 Using **standard summarization** (short document).")
                        summarize_chain = load_summarize_chain(llm, chain_type="stuff", verbose=True)
                    else:
                        #st.write("⚙️ Using **Map-Reduce summarization** (large document).")
                        summarize_chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)

                    # Run summarization
                    summary = summarize_chain.run(chunks)

                    st.subheader("🧾 Summary:")
                    st.write(summary)

                    # Convert summary to audio
                    def generate_podcast_audio(text, title="podcast", voice="Fritz-PlayAI"):
                        model = "playai-tts"
                        response = client.audio.speech.create(
                            model=model,
                            voice=voice,
                            input=text,
                            response_format="wav"
                        )
                        out_path = f"{title}.wav"
                        response.write_to_file(out_path)
                        return out_path

                    try:
                        audio_path = generate_podcast_audio(summary, uploaded_file.name, voice="Celeste-PlayAI")
                    except Exception:
                        st.warning("⚠️ Groq TTS limit reached — using gTTS fallback.")
                        tts = gTTS(summary)
                        audio_path = f"{uploaded_file.name}_fallback.mp3"
                        tts.save(audio_path)

                    st.audio(audio_path)
                    st.success("🎧 Podcast generated successfully!")

                    # Save in current session only
                    new_entry = {"title": uploaded_file.name, "summary": summary, "audio": audio_path}
                    st.session_state.session_podcasts.append(new_entry)

        # Display only current session’s podcasts
        if st.session_state.session_podcasts:
            st.subheader("📚 Podcasts Generated This Session")
            for i, pod in enumerate(st.session_state.session_podcasts):
                with st.expander(f"🎧 {i+1}. {pod['title']}"):
                    st.write(pod["summary"])
                    st.audio(pod["audio"])


    # ---------------- TAB 2: Research Assistant ----------------
    with tab2:
        st.header("🤖 Research Assistant")
        st.markdown("**🧩 Active Tools:** Wikipedia, ArXiv, Web Search")

        chat_container = st.container()
        for msg in st.session_state.mes:
            st.chat_message(msg["role"]).write(msg["content"])

        st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
        prompt = st.chat_input("Ask me anything about your topic...")

        if prompt:
            st.session_state.mes.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            agent = initialize_agent(
                llm=llm,
                tools=tools,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True
            )

            with st.spinner("🔍 Researching across tools... please wait"):
                response = agent.invoke({"input": prompt})

            output_text = response.get("output", str(response))
            st.chat_message("ai").write(output_text)
            st.session_state.mes.append({"role": "ai", "content": output_text})

    # ---------------- TAB 3: Podcast History ----------------
    with tab3:
        st.header("🗂️ Podcast History")

        # Load history from JSON
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                all_history = json.load(f)
            # If using per-user history
            username = st.session_state.get("username", "default_user")
            user_history = all_history.get(username, [])
        else:
            user_history = []

        if user_history:
            for i, pod in enumerate(user_history[::-1]):  # Show newest first
                with st.expander(f"🎧 {len(user_history)-i}. {pod['title']}"):
                    st.write("**Summary:**")
                    st.write(pod["summary"])
                    st.audio(pod["audio"])
        else:
            st.info("ℹ️ No podcast history yet. Generate one from the 'Podcast Generator' tab!")
    # ---------------- Logout ----------------
    # Logout
    if st.sidebar.button("Logout"):
        username = st.session_state.username
        all_history = load_history()
        if username not in all_history:
            all_history[username] = []
        all_history[username].extend(st.session_state.session_podcasts)  # only session podcasts
        save_history(all_history)

        # clear all
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.podcasts = []
        st.session_state.session_podcasts = []
        st.session_state.mes = []

        st.rerun()

# -------------------------------
# Conditional Rendering
# -------------------------------
if st.session_state.logged_in:
    show_main_app()
else:
    show_login()

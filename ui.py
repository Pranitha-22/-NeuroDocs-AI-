import html
import io
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import fitz
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.image_extraction import extract_images_from_pdf
from app.ingestion.pdf_loader import extract_pages_from_pdf
from app.rag.chunking import chunk_pages
from app.rag.vector_store import create_faiss_index
from app.rag.qa_pipeline import generate_rag_result

st.set_page_config(
    page_title="NeuroDocs AI",
    page_icon="brain",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
#MainMenu, footer, header { visibility: hidden; }
.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0f172a 48%, #172033 100%);
    color: #f8fafc;
}
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1220px; }
[data-testid="stSidebar"] {
    background: #050b16;
    border-right: 1px solid rgba(255,255,255,0.08);
}
.hero-title {
    font-size: 2.45rem;
    font-weight: 900;
    line-height: 1.05;
    margin: 0 0 0.2rem;
    background: linear-gradient(90deg, #38bdf8, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle { color: #cbd5e1; font-size: 0.98rem; margin-bottom: 1rem; }
.metric-card, .answer-box, .source-card, .tool-card {
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 8px;
    box-shadow: 0 14px 34px rgba(0,0,0,0.26);
}
.metric-card { padding: 0.85rem; text-align: center; min-height: 88px; }
.metric-value { font-size: 1.5rem; font-weight: 850; color: #38bdf8; }
.metric-label { color: #cbd5e1; font-size: 0.88rem; }
.answer-box { padding: 1.15rem; border-color: rgba(45, 212, 191, 0.45); }
.answer-title { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:0.8rem; }
.answer-title h2 { margin:0; font-size:1.35rem; }
.confidence { color:#cbd5e1; font-size:0.88rem; }
.answer-text { font-size:1rem; line-height:1.72; color:#f8fafc; }
.source-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:0.85rem; margin-top:0.9rem; }
.source-card { padding:0.9rem; }
.source-card strong { color:#7dd3fc; }
.source-card p { color:#cbd5e1; line-height:1.55; font-size:0.9rem; margin:0.45rem 0 0; }
.section-title { font-size:1.18rem; font-weight:800; margin:1rem 0 0.55rem; }
.tool-card { padding:0.9rem; margin-top:0.85rem; }
.small-muted { color:#94a3b8; font-size:0.86rem; }
.stButton > button, .stDownloadButton > button { border-radius:8px; font-weight:700; }
div[data-testid="stTextInput"] input { border-radius:8px; }
div[data-testid="stForm"] { border-radius:8px; border-color: rgba(148, 163, 184, 0.28); }
.stSlider { padding-top: 0; }
</style>
""",
    unsafe_allow_html=True,
)

DEFAULTS = {
    "authenticated": False,
    "processed": False,
    "current_filename": None,
    "chunks": [],
    "index": None,
    "image_folder": None,
    "image_count": 0,
    "page_count": 0,
    "processing_time": 0.0,
    "last_question": "",
    "last_result": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def require_login():
    password = os.getenv("NEURODOCS_APP_PASSWORD")
    if not password:
        st.session_state.authenticated = True
        return

    if st.session_state.authenticated:
        return

    st.markdown('<div class="hero-title">NeuroDocs AI</div>', unsafe_allow_html=True)
    st.caption("Authentication is enabled for this local app.")
    entered = st.text_input("Password", type="password")
    if st.button("Unlock", use_container_width=True):
        if entered == password:
            st.session_state.authenticated = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()


def process_pdf(uploaded_file):
    start = time.time()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_pdf_path = tmp.name

    try:
        pages = extract_pages_from_pdf(temp_pdf_path)
        chunks = chunk_pages(pages)
        index = create_faiss_index(chunks)

        image_folder = PROJECT_ROOT / "data" / "extracted_images"
        if image_folder.exists():
            shutil.rmtree(image_folder)
        image_folder.mkdir(parents=True, exist_ok=True)

        image_count = extract_images_from_pdf(temp_pdf_path, str(image_folder))

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.image_folder = str(image_folder)
        st.session_state.image_count = image_count
        st.session_state.page_count = len(pages)
        st.session_state.processing_time = round(time.time() - start, 2)
        st.session_state.processed = True
        st.session_state.current_filename = uploaded_file.name
        st.session_state.last_question = ""
        st.session_state.last_result = None
    finally:
        try:
            os.remove(temp_pdf_path)
        except OSError:
            pass


def build_answer_pdf(question, result):
    pdf = fitz.open()
    page = pdf.new_page(width=595, height=842)
    y = 54

    page.insert_textbox(fitz.Rect(54, y, 541, y + 28), "NeuroDocs AI Answer", fontsize=16, fontname="helv")
    y += 38
    body = f"Question: {question}\n\n{result.get('answer', '')}"
    page.insert_textbox(fitz.Rect(54, y, 541, 520), body, fontsize=10.5, fontname="helv", lineheight=1.35)

    sources = result.get("sources", [])
    if sources:
        page.insert_textbox(fitz.Rect(54, 545, 541, 570), "Sources", fontsize=14, fontname="helv")
        source_text = "\n\n".join(
            f"[{source.get('id')}] Page {source.get('page', 'unknown')}: {source.get('snippet', '')}"
            for source in sources
        )
        page.insert_textbox(fitz.Rect(54, 578, 541, 790), source_text, fontsize=9.5, fontname="helv", lineheight=1.25)

    output = pdf.tobytes()
    pdf.close()
    return output


def render_voice_tools(answer_text):
    answer_json = json.dumps(answer_text or "")
    components.html(
        f"""
<div style="display:flex; gap:8px; flex-wrap:wrap; font-family:Segoe UI, sans-serif;">
  <button id="listen" style="padding:9px 12px;border-radius:8px;border:1px solid #334155;background:#0f172a;color:white;">Speak answer</button>
  <button id="stop" style="padding:9px 12px;border-radius:8px;border:1px solid #334155;background:#0f172a;color:white;">Stop</button>
  <button id="dictate" style="padding:9px 12px;border-radius:8px;border:1px solid #334155;background:#0f172a;color:white;">Dictate question</button>
  <input id="voiceText" placeholder="Voice transcript appears here" style="min-width:260px;flex:1;padding:9px;border-radius:8px;border:1px solid #334155;background:#020617;color:white;" />
  <button id="use" style="padding:9px 12px;border-radius:8px;border:1px solid #38bdf8;background:#075985;color:white;">Use transcript</button>
</div>
<script>
const answer = {answer_json};
const synth = window.speechSynthesis;
document.getElementById('listen').onclick = () => {{
  if (!answer) return;
  synth.cancel();
  const utterance = new SpeechSynthesisUtterance(answer);
  utterance.rate = 1.02;
  synth.speak(utterance);
}};
document.getElementById('stop').onclick = () => synth.cancel();
document.getElementById('dictate').onclick = () => {{
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {{
    document.getElementById('voiceText').value = 'Speech recognition is not supported in this browser.';
    return;
  }}
  const recognition = new SpeechRecognition();
  recognition.lang = 'en-US';
  recognition.interimResults = false;
  recognition.onresult = (event) => {{
    document.getElementById('voiceText').value = event.results[0][0].transcript;
  }};
  recognition.start();
}};
document.getElementById('use').onclick = () => {{
  const value = document.getElementById('voiceText').value.trim();
  if (value) window.parent.location.search = '?voice_query=' + encodeURIComponent(value);
}};
</script>
""",
        height=96,
    )


require_login()

with st.sidebar:
    st.markdown("## NeuroDocs AI")
    st.caption("Multimodal document intelligence")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is not None and st.session_state.current_filename != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            process_pdf(uploaded_file)

    if st.session_state.processed:
        st.success("Document ready")
        st.markdown("### Analytics")
        st.metric("Pages", st.session_state.page_count)
        st.metric("Chunks", len(st.session_state.chunks))
        st.metric("Images", st.session_state.image_count)
        st.metric("Processing", f"{st.session_state.processing_time}s")
        st.caption("This is session-local storage. Restarting the app clears the document.")

        if st.button("Clear session", use_container_width=True):
            for key, value in DEFAULTS.items():
                if key != "authenticated":
                    st.session_state[key] = value
            st.rerun()

st.markdown('<div class="hero-title">NeuroDocs AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Ask PDFs with page citations, exportable answers, images, and browser voice tools.</div>', unsafe_allow_html=True)

if not st.session_state.processed:
    st.info("Upload a PDF from the sidebar to begin.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
metrics = [
    ("Pages", st.session_state.page_count),
    ("Text Chunks", len(st.session_state.chunks)),
    ("Extracted Images", st.session_state.image_count),
    ("Processing Time", f"{st.session_state.processing_time}s"),
]
for col, (label, value) in zip([m1, m2, m3, m4], metrics):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

voice_query = st.query_params.get("voice_query", "")
if voice_query:
    st.session_state.last_question = voice_query
    st.query_params.clear()

st.markdown('<div class="section-title">Ask Your Document</div>', unsafe_allow_html=True)
with st.form("question_form", clear_on_submit=False):
    q_col, k_col, ask_col = st.columns([5, 1.2, 1])
    with q_col:
        question = st.text_input(
            "Question",
            value=st.session_state.last_question,
            placeholder="Ask a focused question. Vague questions get vague answers.",
            label_visibility="collapsed",
        )
    with k_col:
        top_k = st.slider("Search depth", min_value=3, max_value=15, value=10, help="Higher is slower but can rescue harder questions.")
    with ask_col:
        ask = st.form_submit_button("Ask", use_container_width=True)

if ask:
    if not question.strip():
        st.warning("Enter a real question. Empty prompts are not intelligence tests.")
    else:
        st.session_state.last_question = question.strip()
        with st.spinner("Searching cited evidence..."):
            st.session_state.last_result = generate_rag_result(
                question,
                st.session_state.chunks,
                st.session_state.index,
                top_k=top_k,
            )

result = st.session_state.last_result

if result:
    safe_answer = html.escape(result.get("answer", "")).replace("\n", "<br>")
    pages = result.get("pages", [])
    page_label = ", ".join(str(page) for page in pages) if pages else "No cited page"
    st.markdown(
        f"""
<div class="answer-box">
  <div class="answer-title">
    <h2>Answer</h2>
    <div class="confidence">Confidence: {html.escape(result.get('confidence', 'unknown'))} | Pages: {html.escape(page_label)} | {result.get('answer_time', 0)}s</div>
  </div>
  <div class="answer-text">{safe_answer}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    pdf_bytes = build_answer_pdf(st.session_state.last_question, result)
    st.download_button(
        "Export answer to PDF",
        data=pdf_bytes,
        file_name="neurodocs-answer.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    with st.expander("Voice tools"):
        render_voice_tools(result.get("answer", ""))

    sources = result.get("sources", [])
    if sources:
        st.markdown('<div class="section-title">Source Citations</div>', unsafe_allow_html=True)
        cards = []
        for source in sources:
            snippet = html.escape(source.get("snippet", ""))
            page = html.escape(str(source.get("page", "unknown")))
            score = html.escape(str(source.get("score", "")))
            cards.append(
                f'<div class="source-card"><strong>[{source.get("id")}] Page {page}</strong><p>{snippet}</p><p class="small-muted">Keyword overlap: {score}</p></div>'
            )
        st.markdown(f'<div class="source-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

if st.session_state.image_count > 0:
    st.markdown('<div class="section-title">Relevant Image Carousel</div>', unsafe_allow_html=True)
    image_folder = st.session_state.image_folder
    image_files = sorted([
        filename for filename in os.listdir(image_folder)
        if filename.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    if image_files:
        selected = st.slider("Image", 1, len(image_files), 1, label_visibility="collapsed")
        image_path = os.path.join(image_folder, image_files[selected - 1])
        st.image(image_path, caption=image_files[selected - 1], use_container_width=True)

st.markdown('<p class="small-muted">Not legal, medical, or financial advice. Verify critical answers against the cited page.</p>', unsafe_allow_html=True)

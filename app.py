import streamlit as st
import os
import requests
import time
from rag_engine import (
    load_and_index_pdf,
    load_existing_vectorstore,
    build_chain,
    ask_question,
    get_chunk_count
)

st.set_page_config(
    page_title="AIActBot — EU AI Compliance",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #f8f9fc;
    color: #1a1c2e;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8eaf0;
}
[data-testid="stSidebar"] * { color: #4a4d6a !important; }
[data-testid="stSidebar"] .stButton button {
    background: #f0f0ff;
    border: 1px solid #dddaff;
    color: #4f46e5 !important;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #4f46e5;
    color: #ffffff !important;
    border-color: #4f46e5;
}

/* Header */
.app-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.2rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}
.app-header::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 40%;
    width: 300px; height: 300px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.header-logo {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 10px;
    padding: 6px 14px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    color: rgba(255,255,255,0.9);
}
.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 0.4rem;
    letter-spacing: -0.03em;
    position: relative;
    z-index: 1;
}
.header-sub {
    color: rgba(255,255,255,0.75);
    font-size: 0.95rem;
    font-weight: 400;
    position: relative;
    z-index: 1;
}
.header-tags {
    display: flex;
    gap: 10px;
    margin-top: 1.2rem;
    position: relative;
    z-index: 1;
    flex-wrap: wrap;
}
.htag {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: rgba(255,255,255,0.9);
    font-size: 0.75rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 99px;
}

/* Onboarding card */
.onboard-wrap {
    max-width: 600px;
    margin: 0 auto;
    padding: 1rem 0 2rem;
}
.onboard-card {
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 16px;
    padding: 2.5rem;
    box-shadow: 0 4px 24px rgba(79,70,229,0.06);
}
.onboard-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1a1c2e;
    margin-bottom: 0.5rem;
}
.onboard-sub {
    font-size: 0.9rem;
    color: #6b6f8a;
    margin-bottom: 2rem;
    line-height: 1.6;
}
.onboard-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #4f46e5;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.5rem;
}

/* Chat bubbles */
.chat-row {
    display: flex;
    margin: 1rem 0;
    gap: 12px;
    align-items: flex-start;
}
.chat-row.user { flex-direction: row-reverse; }
.avatar {
    width: 34px;
    height: 34px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    flex-shrink: 0;
    letter-spacing: 0.02em;
}
.avatar-bot {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
}
.avatar-user {
    background: #f0f0ff;
    color: #4f46e5;
    border: 1px solid #dddaff;
}
.bubble {
    max-width: 72%;
    padding: 1rem 1.3rem;
    border-radius: 14px;
    font-size: 0.92rem;
    line-height: 1.75;
}
.bubble-bot {
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-top-left-radius: 3px;
    color: #1a1c2e;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.bubble-user {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-top-right-radius: 3px;
    color: #ffffff;
}

/* Source cards */
.source-card {
    background: #f8f9fc;
    border: 1px solid #e8eaf0;
    border-left: 3px solid #4f46e5;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.82rem;
}
.page-badge {
    background: #ede9fe;
    color: #4f46e5;
    padding: 2px 9px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
}
.snippet {
    color: #9094b0;
    font-style: italic;
    margin-top: 0.4rem;
    line-height: 1.5;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 8px;
    margin: 0.5rem 0 1rem;
}
.metric-card {
    flex: 1;
    background: #f8f9fc;
    border: 1px solid #e8eaf0;
    border-radius: 8px;
    padding: 0.9rem;
    text-align: center;
}
.metric-num {
    font-size: 1.4rem;
    font-weight: 700;
    color: #4f46e5;
}
.metric-label {
    font-size: 0.68rem;
    color: #9094b0;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 3px;
}

/* Section label */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #9094b0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.2rem 0 0.6rem;
}

/* Upload hint */
.upload-hint {
    background: #f0f0ff;
    border: 1px dashed #c4c0ff;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #6b6f8a;
    text-align: center;
    margin-bottom: 0.8rem;
    line-height: 1.5;
}

/* Disclaimer */
.disclaimer {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.78rem;
    color: #92690a;
    margin-top: 1.5rem;
    line-height: 1.6;
}

/* Welcome */
.welcome-card {
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}

/* User context pill */
.context-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #ede9fe;
    border: 1px solid #dddaff;
    border-radius: 99px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #4f46e5;
    font-weight: 500;
    margin-bottom: 1.2rem;
}

/* Hide default streamlit chat */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { display: none !important; }
div[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* Input */
[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f8f9fc; }
::-webkit-scrollbar-thumb { background: #dddaff; border-radius: 3px; }

hr { border-color: #e8eaf0 !important; }

[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e8eaf0 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
for key, val in {
    "chain": None,
    "messages": [],
    "chunk_count": 0,
    "docs_loaded": [],
    "questions_asked": 0,
    "onboarded": True,
    "user_role": "",
    "user_sector": "",
    "user_type": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="header-logo">AIActBot</div>
    <div class="header-title" style="font-size:1.3rem;">EU AI Act Compliance Assistant</div>
    <div class="header-sub">Understand your obligations under the EU AI Act — in plain English.</div>
    <div class="header-tags">
        <span class="htag">EU AI Act 2024</span>
        <span class="htag">RAG Powered</span>
        <span class="htag">Llama 3.2</span>
        <span class="htag">Ireland Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">Document</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-hint">
        Upload the latest official EU AI Act PDF from<br>
        <b>artificialintelligenceact.eu</b>
    </div>
    """, unsafe_allow_html=True)

    upload_own = st.file_uploader(
        "Upload EU AI Act PDF",
        type="pdf",
        label_visibility="collapsed"
    )

    if upload_own:
        pdf_path = f"/tmp/{upload_own.name}"
        with open(pdf_path, "wb") as f:
            f.write(upload_own.read())
        progress_text = st.empty()
        try:
            progress_text.info("Loading file...")
            count, vs = load_and_index_pdf(pdf_path)
            progress_text.info("File loaded...")
            st.session_state.chunk_count = count
            st.session_state.chain = build_chain(vs)
            st.session_state.docs_loaded.append(upload_own.name)
            progress_text.success("Ready for questions.")
        except Exception as e:
            progress_text.error(f"Error: {e}")

    if st.session_state.chain is None:
        vs = load_existing_vectorstore()
        if vs:
            st.session_state.chain = build_chain(vs)
            st.session_state.chunk_count = get_chunk_count()

    st.markdown("---")

    if st.session_state.chunk_count > 0:
        st.markdown('<div class="section-label">Knowledge Base</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-num">{st.session_state.chunk_count}</div>
                <div class="metric-label">Chunks</div>
            </div>
            <div class="metric-card">
                <div class="metric-num">{st.session_state.questions_asked}</div>
                <div class="metric-label">Questions</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        for doc in st.session_state.docs_loaded:
            st.markdown(
                f"<span style='font-size:0.82rem;color:#4f46e5;font-weight:500;'>&#10003; {doc}</span>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    if st.session_state.onboarded:
        if st.session_state.user_role:
            st.markdown(
                f"<div style='font-size:0.8rem;color:#6b6f8a;margin-bottom:4px;'>Role: <b style='color:#4f46e5'>{st.session_state.user_role}</b></div>",
                unsafe_allow_html=True
            )
        if st.session_state.user_sector:
            st.markdown(
                f"<div style='font-size:0.8rem;color:#6b6f8a;margin-bottom:12px;'>Sector: <b style='color:#4f46e5'>{st.session_state.user_sector}</b></div>",
                unsafe_allow_html=True
            )

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.questions_asked = 0
        st.session_state.onboarded = False
        st.session_state.user_role = ""
        st.session_state.user_sector = ""
        st.session_state.user_type = ""
        st.rerun()

    st.markdown("""
    <div style="font-size:0.72rem;color:#c0c3d8;margin-top:1.5rem;text-align:center;line-height:1.6;">
        Powered by Ollama · LangChain · ChromaDB
    </div>
    """, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────
def render_bubble(role, content):
    if role == "user":
        st.markdown(f"""
        <div class="chat-row user">
            <div class="avatar avatar-user">You</div>
            <div class="bubble bubble-user">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-row">
            <div class="avatar avatar-bot">AI</div>
            <div class="bubble bubble-bot">{content}</div>
        </div>
        """, unsafe_allow_html=True)


def render_sources(sources):
    if sources:
        with st.expander(f"View sources — {len(sources)} chunks retrieved"):
            for src in sources:
                st.markdown(f"""
                <div class="source-card">
                    <span class="page-badge">Page {src['page']}</span>
                    &nbsp;<span style="color:#4a4d6a;font-weight:500;">{src['source']}</span>
                    <div class="snippet">"{src['snippet']}..."</div>
                </div>
                """, unsafe_allow_html=True)


def stream_answer(text):
    placeholder = st.empty()
    displayed = ""
    for word in text.split():
        displayed += word + " "
        placeholder.markdown(f"""
        <div class="chat-row">
            <div class="avatar avatar-bot">AI</div>
            <div class="bubble bubble-bot">{displayed}<span style="opacity:0.3;">|</span></div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.025)
    placeholder.markdown(f"""
    <div class="chat-row">
        <div class="avatar avatar-bot">AI</div>
        <div class="bubble bubble-bot">{displayed}</div>
    </div>
    """, unsafe_allow_html=True)


def handle_question(q):
    st.session_state.messages.append({"role": "user", "content": q})
    context = ""
    if st.session_state.user_role:
        context += f"The user is a {st.session_state.user_role}"
    if st.session_state.user_sector:
        context += f" working in the {st.session_state.user_sector} sector"
    full_q = f"{context}. Question: {q}" if context else q
    with st.spinner("Retrieving relevant sections..."):
        result = ask_question(st.session_state.chain, full_q)
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
    st.session_state.questions_asked += 1
    st.rerun()


# ── No document loaded ────────────────────────────────────────
if st.session_state.chain is None:
    st.markdown("""
    <div class="welcome-card">
        <div style="font-size:1.05rem;font-weight:600;color:#1a1c2e;margin-bottom:0.5rem;">
            Get started in two steps
        </div>
        <div style="font-size:0.9rem;color:#6b6f8a;line-height:1.7;">
            1. Download the latest EU AI Act PDF from
            <a href="https://artificialintelligenceact.eu" target="_blank"
               style="color:#4f46e5;font-weight:500;">artificialintelligenceact.eu</a><br>
            2. Upload it using the panel on the left
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">What you can ask</div>', unsafe_allow_html=True)
    questions = [
        "What is a high-risk AI system?",
        "Which AI systems are completely banned?",
        "What must an AI provider do before deploying?",
        "What are the fines for non-compliance?",
        "Does GDPR apply to AI systems?",
        "What is a conformity assessment?",
    ]
    col1, col2 = st.columns(2)
    for i, q in enumerate(questions):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <div class="source-card" style="border-left:3px solid #c4c0ff;margin:4px 0;">
                <span style="color:#4a4d6a;font-size:0.88rem;">{q}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        Disclaimer: AIActBot is for informational purposes only and does not constitute legal advice.
        Always consult a qualified legal professional for compliance decisions.
    </div>
    """, unsafe_allow_html=True)

# ── Onboarding ────────────────────────────────────────────────
elif not st.session_state.onboarded:
    st.markdown('<div class="onboard-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div class="onboard-card">
        <div class="onboard-title">Before we begin</div>
        <div class="onboard-sub">
            A few quick questions to help personalise your compliance answers.
            Please only share what you are comfortable with.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="onboard-label">What is your role?</div>', unsafe_allow_html=True)
        role = st.selectbox(
            "role",
            ["Select your role", "Developer / Engineer", "Legal / Compliance", "Business Owner", "Policy Maker", "Student / Researcher"],
            label_visibility="collapsed"
        )

        st.markdown('<div class="onboard-label" style="margin-top:1rem;">Which sector are you in?</div>', unsafe_allow_html=True)
        sector = st.selectbox(
            "sector",
            ["Select your sector", "Healthcare", "Finance / Fintech", "HR / Recruitment", "Education", "Government / Public Sector", "Technology", "Other"],
            label_visibility="collapsed"
        )

        st.markdown('<div class="onboard-label" style="margin-top:1rem;">What type of organisation?</div>', unsafe_allow_html=True)
        org = st.selectbox(
            "org",
            ["Select organisation type", "Startup", "SME", "Enterprise", "Public Body", "Individual / Freelancer"],
            label_visibility="collapsed"
        )

        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

        if st.button("Start using AIActBot", use_container_width=True):
            st.session_state.user_role = role if role != "Select your role" else ""
            st.session_state.user_sector = sector if sector != "Select your sector" else ""
            st.session_state.user_type = org if org != "Select organisation type" else ""
            st.session_state.onboarded = True
            st.rerun()

        st.markdown("""
        <div style="text-align:center;margin-top:0.8rem;">
            <span style="font-size:0.8rem;color:#9094b0;">
                Prefer to skip?
            </span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Skip and continue", use_container_width=True):
            st.session_state.onboarded = True
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ── Main chat ─────────────────────────────────────────────────
else:
    if st.session_state.user_role or st.session_state.user_sector:
        parts = []
        if st.session_state.user_role:
            parts.append(st.session_state.user_role)
        if st.session_state.user_sector:
            parts.append(st.session_state.user_sector)
        if st.session_state.user_type:
            parts.append(st.session_state.user_type)
        st.markdown(f"""
        <div class="context-pill">
            Personalised for: {" · ".join(parts)}
        </div>
        """, unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown('<div class="section-label">Suggested Questions</div>', unsafe_allow_html=True)
        suggested = [
            "What is a high-risk AI system?",
            "Which AI systems are banned?",
            "What are the fines for non-compliance?",
            "What must an AI provider do?",
        ]
        cols = st.columns(4)
        for i, q in enumerate(suggested):
            with cols[i]:
                if st.button(q, key=f"chip_{i}", use_container_width=True):
                    handle_question(q)

    for msg in st.session_state.messages:
        render_bubble(msg["role"], msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))

    if prompt := st.chat_input("Ask anything about the EU AI Act..."):
        render_bubble("user", prompt)
        context = ""
        if st.session_state.user_role:
            context += f"The user is a {st.session_state.user_role}"
        if st.session_state.user_sector:
            context += f" working in the {st.session_state.user_sector} sector"
        full_q = f"{context}. Question: {prompt}" if context else prompt
        with st.spinner("Retrieving relevant sections..."):
            result = ask_question(st.session_state.chain, full_q)
        stream_answer(result["answer"])
        render_sources(result["sources"])
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })
        st.session_state.questions_asked += 1
# chatbot_app.py — Final submission-ready version (console-warning minimized, iframe-safe)
# Note: This file expects GEMINI_API_KEY to be provided via environment variable (GEMINI_API_KEY).
# Do not commit secrets to source control.

import os
import uuid
import json
import time
import html
import logging
from typing import List, Generator

import streamlit as st
import streamlit.components.v1 as components

# Optional libs (fail gracefully)
try:
    from langdetect import detect
except Exception:
    detect = None

try:
    from autocorrect import Speller
    spell_en = Speller(lang="en")
    AUTOCORRECT_AVAILABLE = True
except Exception:
    spell_en = None
    AUTOCORRECT_AVAILABLE = False

try:
    from googletrans import Translator
    translator = Translator()
    TRANSLATE_AVAILABLE = True
except Exception:
    translator = None
    TRANSLATE_AVAILABLE = False

# requests used for font download fallback (optional)
try:
    import requests
except Exception:
    requests = None

# PDF lib
try:
    from fpdf import FPDF
except Exception:
    raise RuntimeError("Please install fpdf: pip install fpdf")

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------------- config / fonts ----------------
FONTS_DIR = "fonts"
FONT_FILE = "NotoSans-Regular.ttf"
FONT_PATH = os.path.join(FONTS_DIR, FONT_FILE)
NOTO_URL = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf"
SHARED_FILE = "shared_chats.json"

# ---------------- GEMINI KEY (env preferred) ----------------
FALLBACK_HARDCODED_KEY = None
GEMINI_API_KEY = "AIzaSyAVRJ_VPaF-dqbppFq8PCnKjIoyQ1JIP1s" or FALLBACK_HARDCODED_KEY
MODEL_NAME = "gemini-2.5-flash"

model = None
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        logging.info("Gemini client initialized.")
    except Exception as e:
        model = None
        logging.exception("Failed to initialize Gemini client: %s", e)
else:
    logging.warning("GEMINI_API_KEY not set. App will run in demo mode (no live Gemini).")

# ---------------- font helpers ----------------
def ensure_font_downloaded(timeout: int = 15) -> bool:
    os.makedirs(FONTS_DIR, exist_ok=True)
    if os.path.exists(FONT_PATH):
        return True
    if not requests:
        logging.info("requests not available; skipping font download.")
        return False
    try:
        resp = requests.get(NOTO_URL, timeout=timeout)
        if resp.status_code == 200 and resp.content:
            with open(FONT_PATH, "wb") as f:
                f.write(resp.content)
            return True
    except Exception as e:
        logging.warning("Font download failed: %s", e)
        return False
    return os.path.exists(FONT_PATH)

def safe_set_font(pdf_obj: FPDF, fam: str, style: str, size: int):
    try:
        pdf_obj.set_font(fam, style, size)
    except Exception:
        try:
            pdf_obj.set_font("Helvetica", style, size)
        except Exception:
            pdf_obj.set_font("Times", style, size)

def create_pdf_bytes(title: str, messages: List[dict]) -> bytes:
    ensure_font_downloaded()
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    family = "Arial"
    try:
        if os.path.exists(FONT_PATH):
            try:
                pdf.add_font("Noto", "", FONT_PATH)
                pdf.add_font("Noto", "B", FONT_PATH)
                family = "Noto"
            except Exception as e:
                logging.warning("add_font Noto failed: %s", e)
                family = "Arial"
    except Exception as e:
        logging.warning("Font registration check failed: %s", e)
        family = "Arial"

    safe_set_font(pdf, family if family else "Helvetica", "B", 16)
    try:
        pdf.set_text_color(0, 102, 153)
    except Exception:
        pass
    try:
        pdf.cell(0, 10, title, ln=True, align="C")
    except Exception:
        safe_title = title.encode("latin-1", errors="replace").decode("latin-1")
        try:
            pdf.cell(0, 10, safe_title, ln=True, align="C")
        except Exception:
            pdf.cell(0, 10, safe_title, align="C")
    pdf.ln(8)

    safe_set_font(pdf, family if family else "Helvetica", "", 12)
    page_w = pdf.w - 2 * pdf.l_margin
    for m in messages:
        txt = m["content"].replace("\r\n", "\n").replace("\n", " ")
        if m["role"] == "user":
            bubble_w = page_w * 0.65
            x_start = pdf.l_margin + (page_w - bubble_w)
            pdf.set_x(x_start)
            try:
                pdf.set_fill_color(220, 235, 255)
            except Exception:
                pass
            try:
                pdf.multi_cell(bubble_w, 7, "You: " + txt, border=0, fill=True)
            except Exception:
                safe_txt = txt.encode("latin-1", errors="replace").decode("latin-1")
                pdf.multi_cell(bubble_w, 7, "You: " + safe_txt, border=0, fill=True)
            pdf.ln(3)
        else:
            bubble_w = page_w * 0.72
            pdf.set_x(pdf.l_margin)
            try:
                pdf.set_fill_color(245, 245, 245)
            except Exception:
                pass
            try:
                pdf.multi_cell(bubble_w, 7, "Bot: " + txt, border=0, fill=True)
            except Exception:
                safe_txt = txt.encode("latin-1", errors="replace").decode("latin-1")
                pdf.multi_cell(bubble_w, 7, "Bot: " + safe_txt, border=0, fill=True)
            pdf.ln(3)

    pdf.ln(6)
    safe_set_font(pdf, family if family else "Helvetica", "I", 10)
    try:
        pdf.set_text_color(100, 100, 100)
    except Exception:
        pass
    footer = "✨ Made by Bapan Ghosh ✨"
    try:
        pdf.cell(0, 8, footer, ln=True, align="C")
    except Exception:
        try:
            pdf.cell(0, 8, footer.encode("ascii", errors="replace").decode("ascii"), ln=True, align="C")
        except Exception:
            pdf.cell(0, 8, footer, align="C")
    try:
        raw = pdf.output(dest="S")
    except TypeError:
        raw = pdf.output()
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    if isinstance(raw, str):
        return raw.encode("latin-1")
    return bytes(raw)

# ---------------- sharing ----------------
def save_shared_chat(chat_title: str, messages: List[dict]) -> str:
    sid = str(uuid.uuid4())[:8]
    data = {"title": chat_title, "messages": messages}
    shared = {}
    if os.path.exists(SHARED_FILE):
        try:
            with open(SHARED_FILE, "r", encoding="utf-8") as f:
                shared = json.load(f)
        except Exception:
            shared = {}
    shared[sid] = data
    try:
        with open(SHARED_FILE, "w", encoding="utf-8") as f:
            json.dump(shared, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.warning("Failed to save shared chat: %s", e)
    return sid

def load_shared_chat(sid: str):
    if os.path.exists(SHARED_FILE):
        try:
            with open(SHARED_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get(sid)
        except Exception:
            return None
    return None

# ---------------- helpers ----------------
def detect_language(text: str):
    if detect:
        try:
            return detect(text)
        except Exception:
            return None
    return None

def autocorrect_text(text: str, lang: str):
    if AUTOCORRECT_AVAILABLE and (lang is None or lang.startswith("en")):
        try:
            return spell_en(text)
        except Exception:
            return text
    return text

def maybe_translate_for_model(text: str, lang: str):
    if TRANSLATE_AVAILABLE and lang and not lang.startswith("en"):
        try:
            res = translator.translate(text, src=lang, dest="en")
            return res.text
        except Exception:
            return text
    return text

def update_memory_summary(messages: List[dict], max_chars: int = 300) -> str:
    last_user_msgs = [m["content"] for m in messages if m["role"] == "user"][-6:]
    summary = " | ".join([m.replace("\n", " ")[:200] for m in last_user_msgs])
    return summary[:max_chars]

# ---------------- multi-agent helpers ----------------
import re

def is_research_query(prompt: str) -> bool:
    """Detect if user message needs research/summarization."""
    keywords = ["research", "analyze", "explain", "summarize", "report", "find", "data", "article"]
    return any(k in prompt.lower() for k in keywords)

def run_research_agent(prompt: str) -> str:
    """First agent: gather factual/analytical content."""
    if model is None:
        return "[Research Agent offline – Gemini not configured]"
    try:
        response = model.generate_content(
            f"You are Research Agent. Collect structured reasoning, factual analysis, and relevant context for:\n{prompt}",
            stream=False
        )
        return getattr(response, "text", str(response))
    except Exception as e:
        logging.warning("Research agent failed: %s", e)
        return f"[Research Agent error: {e}]"

def run_summarizer_agent(context: str) -> str:
    """Second agent: summarize the research agent output."""
    if model is None:
        return "[Summarizer Agent offline – Gemini not configured]"
    try:
        response = model.generate_content(
            f"You are Summarizer Agent. Summarize and simplify this research in 3-5 clear bullet points:\n\n{context}",
            stream=False
        )
        return getattr(response, "text", str(response))
    except Exception as e:
        logging.warning("Summarizer agent failed: %s", e)
        return f"[Summarizer Agent error: {e}]"

# ---------------- model streaming ----------------
def ask_gemini_smart_stream(user_text: str, system_prompt: str = "", max_retries: int = 2) -> Generator[str, None, None]:
    if model is None:
        yield "[Model not configured — demo response]\n"
        yield "Demo: This is a simulated response because Gemini is not configured. Set GEMINI_API_KEY environment variable."
        return

    try:
        lang = detect_language(user_text) if detect else None
    except Exception:
        lang = None

    corrected = autocorrect_text(user_text, lang)
    model_input = maybe_translate_for_model(corrected, lang)

    attempt = 0
    last_exception = None

    while attempt <= max_retries:
        attempt += 1
        try:
            content_list = []
            if system_prompt:
                content_list.append({"role": "model", "parts": [system_prompt]})
            content_list.append({"role": "user", "parts": [model_input]})

            response_stream = model.generate_content(content_list, stream=True)
            for chunk in response_stream:
                try:
                    yield getattr(chunk, "text", str(chunk))
                except Exception:
                    yield str(chunk)
            return

        except Exception as e:
            last_exception = e
            msg = str(e).lower()
            logging.warning("Gemini request failed (attempt %d): %s", attempt, e)
            if "role" in msg and ("user" in msg or "model" in msg) and attempt <= max_retries:
                try:
                    combined = (system_prompt + "\n\n" + model_input).strip() if system_prompt else model_input
                    logging.info("Falling back to concatenated prompt due to role error.")
                    response_stream = model.generate_content([{"role": "user", "parts": [combined]}], stream=True)
                    for chunk in response_stream:
                        try:
                            yield getattr(chunk, "text", str(chunk))
                        except Exception:
                            yield str(chunk)
                    return
                except Exception as e2:
                    logging.warning("Fallback concat also failed: %s", e2)
                    last_exception = e2
            if attempt > max_retries:
                yield f"[Error talking to model: {last_exception}]"
                return
            time.sleep(1 + attempt)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="MINI CHATBOT", page_icon="🤖", layout="wide")

# suppress browser "Unrecognized feature" spam (Permissions-Policy) — expanded list
st.markdown("""
<meta http-equiv="Permissions-Policy"
 content="
   ambient-light-sensor=(),
   autoplay=(),
   battery=(),
   camera=(),
   document-domain=(),
   display-capture=(),
   fullscreen=(),
   geolocation=(),
   layout-animations=(),
   legacy-image-formats=(),
   microphone=(),
   midi=(),
   oversized-images=(),
   payment=(),
   picture-in-picture=(),
   publickey-credentials-get=(),
   sync-xhr=(),
   usb=(),
   vr=(),
   wake-lock=()
 " />
""", unsafe_allow_html=True)

# Keep header spacing and footer small; removed the problematic global hide CSS
st.markdown("""
<style>
    .main .block-container { padding-top: 78px !important; }
    header[data-testid="stHeader"] { display: none; }
    [data-testid="stChatInput"] { padding-bottom: 35px; }
    #app-footer { position: fixed; bottom: 0; left: 0; right: 0; padding: 5px 0; padding-left: 21rem;
      background: #fff; z-index: 101; text-align: center; color: gray; font-size: 13px; border-top: 1px solid #e8e8e8; }
</style>
""", unsafe_allow_html=True)

# ---------------- session init ----------------
if "chats" not in st.session_state:
    st.session_state["chats"] = {}
if "current" not in st.session_state:
    cid = str(uuid.uuid4())
    st.session_state["current"] = cid
    st.session_state["chats"][cid] = {"title": "New Chat", "messages": []}
if "options_target" not in st.session_state:
    st.session_state["options_target"] = None
if "editing_index" not in st.session_state:
    st.session_state["editing_index"] = None
if "agent_role" not in st.session_state:
    st.session_state["agent_role"] = "General Assistant"
if "memory_summary" not in st.session_state:
    st.session_state["memory_summary"] = ""

# ---------------- URL query param handler for edit/create_share actions (robust) ----------------
qp = {}
try:
    qp = st.query_params
except Exception:
    try:
        qp = st.experimental_get_query_params()
    except Exception:
        qp = {}

# handle edit param -> open edit UI
if "edit" in qp:
    try:
        raw = qp.get("edit")
        if isinstance(raw, list):
            raw = raw[0]
        edit_idx = int(str(raw).strip())
        st.session_state["editing_index"] = edit_idx
    except Exception:
        pass

    new_qp = dict(qp)
    if "edit" in new_qp:
        del new_qp["edit"]

    try:
        st.set_query_params(**new_qp)
    except Exception:
        try:
            st.experimental_set_query_params(**new_qp)
        except Exception:
            try:
                st.experimental_set_query_params()
            except Exception:
                pass

    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.rerun()
        except Exception:
            pass

# handle create_share param -> create a share link for current chat AND open popup after reload
if "create_share" in qp:
    try:
        cur = st.session_state.get("current")
        chat = st.session_state["chats"].get(cur, {"title": "New Chat", "messages": []})
        sid = save_shared_chat(chat.get("title", "shared"), chat.get("messages", []))
        # default local link; change if deploying to other host
        link = f"http://localhost:8501/?share={sid}"
        st.session_state["share_link"] = {"link": link, "source": "header", "position_hint": "top-right"}
        # set flag so frontend auto-opens header popup after rerun
        st.session_state["open_share_popup"] = True
    except Exception:
        logging.exception("Failed to create share link from create_share param.")

    new_qp = dict(qp)
    if "create_share" in new_qp:
        del new_qp["create_share"]

    try:
        st.set_query_params(**new_qp)
    except Exception:
        try:
            st.experimental_set_query_params(**new_qp)
        except Exception:
            try:
                st.experimental_set_query_params()
            except Exception:
                pass

    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.rerun()
        except Exception:
            pass

# ---------------- If in edit-mode: render only the edit UI (early return) ----------------
if st.session_state.get("editing_index") is not None:
    cur = st.session_state["current"]
    chat = st.session_state["chats"].get(cur, {"title": "New Chat", "messages": []})
    idx = st.session_state["editing_index"]

    can_edit = False
    if isinstance(idx, int) and 0 <= idx < len(chat["messages"]):
        can_edit = (chat["messages"][idx]["role"] == "user")

    if not can_edit:
        st.session_state["editing_index"] = None
    else:
        st.markdown("<div style='max-width:900px; margin:20px auto;'>", unsafe_allow_html=True)
        st.markdown("### ✏️ Edit message")
        message_to_edit = chat["messages"][idx]["content"]
        edited_prompt = st.text_area("Edit your message:", value=message_to_edit, height=200, key="edit_text_area_focus")
        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("✅ Save & Submit", use_container_width=True):
                chat["messages"][idx]["content"] = edited_prompt
                st.session_state["chats"][cur]["messages"] = chat["messages"][:idx + 1]
                if st.session_state.get("share_link"):
                    st.session_state.pop("share_link", None)
                st.session_state["editing_index"] = None
                try:
                    st.experimental_rerun()
                except Exception:
                    st.rerun()
        with c2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state["editing_index"] = None
                try:
                    st.experimental_rerun()
                except Exception:
                    st.rerun()
        st.markdown("<p style='text-align:center;color:gray;font-size:13px;'>⚠️ Chatbot can make mistakes. © 2025 BAPAN GHOSH</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("### 🤖 MINI CHATBOT")
    st.markdown("##### ✨ MADE BY BAPAN GHOSH ✨")
    st.write("---")

    if st.button("➕ New Chat"):
        nid = str(uuid.uuid4())
        st.session_state["chats"][nid] = {"title": "New Chat", "messages": []}
        st.session_state["current"] = nid
        st.session_state["editing_index"] = None
        try:
            st.experimental_rerun()
        except Exception:
            st.rerun()

    st.markdown("#### 💾 Chat History")
    for cid, chat_item in list(st.session_state["chats"].items()):
        cols = st.columns([9,1])
        with cols[0]:
            if st.button(chat_item["title"], key=f"open_{cid}", use_container_width=True):
                st.session_state["current"] = cid
                st.session_state["editing_index"] = None
                if st.session_state.get("share_link"):
                    st.session_state.pop("share_link", None)
                try:
                    st.experimental_rerun()
                except Exception:
                    st.rerun()
        with cols[1]:
            if st.button("⋯", key=f"opts_{cid}", help="Options for this chat (rename/share/delete)", use_container_width=True):
                st.session_state["options_target"] = cid if st.session_state.get("options_target") != cid else None

    if st.session_state.get("options_target"):
        t = st.session_state["options_target"]
        st.write("---")
        st.markdown(f"**Options for:** {st.session_state['chats'][t]['title']}")
        if st.button("Rename", key="opt_rename"):
            st.session_state["rename_target"] = t
        if st.button("Delete", key="opt_delete"):
            if t in st.session_state["chats"]:
                del st.session_state["chats"][t]
            if t == st.session_state["current"]:
                st.session_state["current"] = next(iter(st.session_state["chats"]), str(uuid.uuid4()))
                if st.session_state["current"] not in st.session_state["chats"]:
                    st.session_state["chats"][st.session_state["current"]] = {"title": "New Chat", "messages": []}
            st.session_state["options_target"] = None
            try:
                st.experimental_rerun()
            except Exception:
                st.rerun()
        if st.button("Share (copy link)", key="opt_share"):
            sid = save_shared_chat(st.session_state["chats"][t]["title"], st.session_state["chats"][t]["messages"])
            link = f"http://localhost:8501/?share={sid}"
            st.session_state["share_link"] = {"link": link, "source": "sidebar", "position": "footer"}
            st.session_state["options_target"] = None
            try:
                st.experimental_rerun()
            except Exception:
                st.rerun()

    if st.session_state.get("rename_target"):
        rt = st.session_state["rename_target"]
        new_title = st.text_input("Rename chat", st.session_state["chats"][rt]["title"], key="rename_input")
        if st.button("Save rename", key="rename_save"):
            if new_title.strip():
                st.session_state["chats"][rt]["title"] = new_title.strip()
            st.session_state["rename_target"] = None
            try:
                st.experimental_rerun()
            except Exception:
                st.rerun()

    st.write("---")
    cur = st.session_state["current"]
    msgs = st.session_state["chats"][cur]["messages"]
    if msgs:
        try:
            pdf_bytes = create_pdf_bytes(st.session_state["chats"][cur]["title"], msgs)
            st.download_button("📥 Download Chat (PDF)", pdf_bytes,
                                file_name=f"{st.session_state['chats'][cur]['title']}.pdf",
                                mime="application/pdf")
        except Exception as e:
            logging.warning("PDF generation failed: %s", e)
            st.error("PDF generation failed: " + str(e))
    else:
        st.info("No messages yet.")

    st.write("---")
    role = st.selectbox("Agent role", ["General Assistant", "Study Buddy", "Resume Coach", "Code Helper", "Summarizer"], key="role_select")
    st.session_state["agent_role"] = role

# ---------------- header (sticky) with reliable Share create + hover-pencil CSS ----------------
st.markdown("""
<style>
/* sticky header: always visible, above sidebar */
#sticky-header {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  padding-left: 260px; /* leave space for sidebar so title is visible */
  background: linear-gradient(180deg,#ffffff,#fbfbfb);
  border-bottom: 1px solid #e6e6e6;
  z-index: 1600; /* larger than sidebar */
  box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}

/* header left title text */
#sticky-header .title {
  display:flex;
  gap:12px;
  align-items:center;
  font-weight:700;
  font-size:18px;
  color:#111;
  user-select: none;
}

/* header right area */
#sticky-header .header-actions {
  display:flex;
  gap:8px;
  align-items:center;
}

/* keep main content padded so header doesn't overlap */
.main .block-container {
  padding-top: 78px !important;
}

/* share popup anchored under header, hidden by default */
#headerSharePopup {
  position: fixed;
  right: 20px;
  top: 72px;
  width: 360px;
  z-index: 1650;
  background: #fff;
  border: 1px solid #ddd;
  box-shadow: 0 8px 24px rgba(0,0,0,0.12);
  padding: 12px;
  border-radius: 10px;
  display: none;
  font-family: system-ui, Roboto, Arial, sans-serif;
}

/* show pencil on bubble hover (ChatGPT-like) */
.bubble .pencil { display: none; }
.bubble:hover .pencil { display: inline-block !important; }

/* small responsive tweaks */
@media(max-width:680px) {
  #sticky-header { padding-left: 16px; padding-right: 12px; }
  #headerSharePopup { left: 8px; right: 8px; width: auto; }
}
</style>

<!-- hidden anchor kept for same-origin link generation but we no longer click it from iframe JS -->
<a id="createShareAnchor" href="?create_share=1" style="display:none;"></a>

<div id="sticky-header">
  <div class="title">
    <div style="font-size:20px;">🤖</div>
    <div style="line-height:1; margin-left:6px;">
      <div style="font-weight:800;">MINI CHATBOT</div>
      <div style="font-size:11px; color:#666; margin-top:1px;">fast replies • save chats</div>
    </div>
  </div>

  <div class="header-actions">
    <button id="headerShareBtn" title="Share chat" style="background:#0d6efd; color:#fff; border:none; padding:8px 12px; border-radius:8px; cursor:pointer; font-weight:600;">
      🔗 Share
    </button>
  </div>
</div>

<div id="headerSharePopup" role="dialog" aria-hidden="true">
  <div style="display:flex; gap:8px; align-items:center;">
    <input id="headerShareInput" value="" style="flex:1; padding:8px 10px; border:1px solid #eee; border-radius:6px;" readonly />
    <button id="headerCopyBtn" style="padding:7px 10px; background:#0d6efd; color:#fff; border:none; border-radius:6px; cursor:pointer;">Copy</button>
  </div>
  <div id="headerCopyStatus" style="margin-top:8px; font-size:13px; color:green; display:none;">Share link copied to clipboard!</div>
  <div style="margin-top:8px; text-align:right;">
    <button id="headerClosePopup" style="background:#f1f1f1; border:1px solid #e3e3e3; padding:6px 8px; border-radius:6px; cursor:pointer;">Close</button>
  </div>
</div>

<script>
(function(){
  const shareBtn = document.getElementById('headerShareBtn');
  const popup = document.getElementById('headerSharePopup');
  const copyBtn = document.getElementById('headerCopyBtn');
  const input = document.getElementById('headerShareInput');
  const status = document.getElementById('headerCopyStatus');
  const closeBtn = document.getElementById('headerClosePopup');

  function showPopup(val) { popup.style.display = val ? 'block' : 'none'; popup.setAttribute('aria-hidden', !val); }

  shareBtn.addEventListener('click', function(e){
    try {
      // DO NOT attempt top-level navigation from inside iframe (sandbox blocks it).
      // If the server has already created a share link it will be available via window._latestShareLink.
      if (window._latestShareLink) {
        input.value = window._latestShareLink;
        showPopup(true);
        return;
      }
      // If link not yet available, simply open the popup and instruct the user to use the Share button in the sidebar
      showPopup(true);
    } catch(err) {
      // fallback: just show popup
      showPopup(true);
    }
  });

  copyBtn.addEventListener('click', function(){
    if (!input.value) return;
    if (navigator.clipboard) {
      navigator.clipboard.writeText(input.value).then(function(){
        status.style.display = 'block';
        setTimeout(()=>status.style.display='none', 1800);
      }).catch(function(err){
        try { input.select(); document.execCommand('copy'); status.style.display = 'block'; setTimeout(()=>status.style.display='none',1800);} catch(e){ console.warn('copy failed',e); }
      });
    }
  });

  closeBtn.addEventListener('click', function(){ showPopup(false); });

  // if server set flag to auto-open popup after reload, open it now
  try {
    if (window._openSharePopup) {
      if (window._latestShareLink) {
        input.value = window._latestShareLink;
      }
      showPopup(true);
      window._openSharePopup = false;
    }
  } catch(e){}
})();
</script>

""", unsafe_allow_html=True)

# populate header share input if share_link present and pass open flag
_latest = ""
_open_flag = False
if st.session_state.get("share_link"):
    _latest = html.escape(st.session_state["share_link"].get("link", "")) if isinstance(st.session_state["share_link"], dict) else html.escape(str(st.session_state["share_link"]))
if st.session_state.get("open_share_popup"):
    _open_flag = True
    # clear flag so it doesn't reopen again on next rerun
    st.session_state["open_share_popup"] = False

st.markdown(f"<script>window._latestShareLink = '{_latest}'; window._openSharePopup = {str(_open_flag).lower()}; try{{ var el=document.getElementById('headerShareInput'); if(el) el.value='{_latest}'; }}catch(e){{}}</script>", unsafe_allow_html=True)

# ---------------- main chat area (click-to-select + pencil) ----------------
chat_container = st.container()
cur = st.session_state["current"]
chat = st.session_state["chats"].get(cur, {"title": "New Chat", "messages": []})

html_bubbles = ["<div style='max-width:980px; margin: 6px auto; font-family: system-ui, Roboto, Arial, sans-serif;'>"]
for index, m in enumerate(chat["messages"]):
    role = m["role"]
    content = html.escape(m["content"]).replace("\n", "<br/>")
    if role == "user":
        bubble_style = "background:#e8f4ff; color:#111; margin-left:22%; margin-right:8px;"
    else:
        bubble_style = "background:#f3f3f3; color:#111; margin-right:22%; margin-left:8px;"

    if role == "user":
        edit_icon = f'<button class="pencil" data-edit-index="{index}" title="Edit this message" style="display:none; border:none; background:rgba(255,255,255,0.9); padding:6px; border-radius:8px; cursor:pointer;">✏️</button>'
    else:
        edit_icon = ''

    bubble_html = f'''
    <div class="chat-row" data-index="{index}" style="display:flex; justify-content:{'flex-end' if role=='user' else 'flex-start'}; padding:8px 6px;">
      <div class="bubble" data-index="{index}" style="position:relative; {bubble_style} padding:12px 14px; border-radius:12px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); max-width:72%;">
        <div class="bubble-content">{content}</div>
        <div class="bubble-edit-holder" style="position:absolute; top:6px; right:6px;">{edit_icon}</div>
      </div>
    </div>
    '''
    html_bubbles.append(bubble_html)

html_bubbles.append("</div>")

js = """
<style>
.bubble.selected { box-shadow: 0 6px 18px rgba(13,110,253,0.12); transform: translateY(-1px); }
/* Ensure pencil is only visible on hover (ChatGPT-like) */
.bubble .pencil { display: none; }
.bubble:hover .pencil { display: inline-block !important; }
</style>
<script>
(function(){
  // No bubble click handlers — pencil appears on hover only (CSS).
  // Attach click handlers to all pencils so clicking them navigates to ?edit=<index>
  function attachPencilHandlers() {
    document.querySelectorAll('.pencil').forEach(function(p){
      // replace node to remove old listeners reliably
      var newp = p.cloneNode(true);
      p.parentNode.replaceChild(newp, p);
      newp.addEventListener('click', function(ev){
        ev.stopPropagation();
        var idx = newp.getAttribute('data-edit-index');
        if (!idx) return;
        try {
          var u = window.location.protocol + '//' + window.location.host + window.location.pathname + '?edit=' + encodeURIComponent(idx);
          // safer single-frame navigation (same-origin)
          try { window.location.href = u; return; } catch(e){ console.warn('navigation failed', e); }
        } catch(err) {
          console.warn('edit navigation failed', err);
        }
      });
    });
  }

  // initial attach and try again (Streamlit may rerender)
  attachPencilHandlers();
  setTimeout(attachPencilHandlers, 500);
  setTimeout(attachPencilHandlers, 1200);
  setTimeout(attachPencilHandlers, 2200);

  // hide any selected styling when clicking elsewhere
  document.addEventListener('click', function(e){
    document.querySelectorAll('.bubble').forEach(function(b){ b.classList.remove('selected'); });
  });

  // optional keyboard: Esc hides (no-op mostly because hover governs visibility)
  document.addEventListener('keydown', function(ev){
    if (ev.key === 'Escape') {
      document.querySelectorAll('.bubble').forEach(function(b){ b.classList.remove('selected'); });
    }
  });
})();
</script>
"""

components.html("\n".join(html_bubbles) + js, height=min(900, 140 + 90 * len(chat["messages"])), scrolling=True)

# ---------------- typing indicator + streaming (MULTI-AGENT patched) ----------------
if chat["messages"] and chat["messages"][-1]["role"] == "user" and st.session_state.get("editing_index") is None:
    user_msg = chat["messages"][-1]["content"]
    system_prompt = f"You are a helpful {st.session_state.get('agent_role', 'assistant')}. Keep answers concise and practical."
    memory = st.session_state.get("memory_summary", "")
    if memory:
        system_prompt += f" Conversation memory summary: {memory}"

    with chat_container:
        with st.chat_message("assistant"):
            anim_html = """
            <div style='display:flex; gap:10px; align-items:center; font-weight:600;'>
              <div>⏳ Just a second</div>
              <div id='dots' style='font-weight:600; color:#666;'>• • •</div>
            </div>
            <style>
              @keyframes blink { 0% {opacity:0.2} 50% {opacity:1} 100% {opacity:0.2} }
              #dots { letter-spacing:6px; }
              #dots span { animation: blink 1.2s infinite; }
            </style>
            <script>
              const d = document.getElementById('dots');
              if (d) {
                d.innerHTML = "<span>•</span><span style='animation-delay:0.2s'>•</span><span style='animation-delay:0.4s'>•</span>";
              }
            </script>
            """
            status_msg_container = st.empty()
            with status_msg_container:
                components.html(anim_html, height=45)
            placeholder = st.empty()
            reply_accum = ""

            # --- NEW multi-agent logic ---
            try:
                if is_research_query(user_msg):
                    # Research agent (synchronous) -> Summarizer agent (synchronous)
                    research_out = run_research_agent(user_msg)
                    summary_out = run_summarizer_agent(research_out)
                    reply_accum = (
                        "### 🧠 Research Agent Findings\n\n"
                        f"{research_out}\n\n"
                        "### 📝 Summarizer Agent Output\n\n"
                        f"{summary_out}\n\n"
                        "🤖 Research Agent → Summarizer Agent completed ✅"
                    )
                    placeholder.markdown(reply_accum)
                else:
                    # Normal single-agent streaming
                    for chunk in ask_gemini_smart_stream(user_msg, system_prompt=system_prompt):
                        if status_msg_container is not None:
                            status_msg_container.empty()
                            status_msg_container = None
                        if chunk:
                            reply_accum += chunk
                            placeholder.markdown(reply_accum)
            except Exception as e:
                logging.exception("Error in multi-agent flow: %s", e)
                reply_accum = f"[Agent error: {e}]"
                placeholder.markdown(reply_accum)

            reply = reply_accum if reply_accum else "[No reply]"

    chat["messages"].append({"role": "assistant", "content": reply})
    st.session_state["chats"][cur] = chat
    st.session_state["memory_summary"] = update_memory_summary(chat["messages"])
    try:
        st.experimental_rerun()
    except Exception:
        st.rerun()

# ---------------- editing UI or input UI (fallback) ----------------
if st.session_state.get("editing_index") is not None:
    idx = st.session_state["editing_index"]
    can_edit = False
    if isinstance(idx, int) and 0 <= idx < len(chat["messages"]):
        can_edit = (chat["messages"][idx]["role"] == "user")
    if not can_edit:
        st.session_state["editing_index"] = None
        try:
            st.experimental_rerun()
        except Exception:
            st.rerun()
    message_to_edit = chat["messages"][idx]["content"]
    st.markdown("---")
    edited_prompt = st.text_area("Edit your message:", value=message_to_edit, height=100, key="edit_text_area")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Save & Submit", use_container_width=True):
            chat["messages"][idx]["content"] = edited_prompt
            st.session_state["chats"][cur]["messages"] = chat["messages"][:idx + 1]
            if st.session_state.get("share_link"):
                st.session_state.pop("share_link", None)
            st.session_state["editing_index"] = None
            try:
                st.experimental_rerun()
            except Exception:
                st.rerun()
    with c2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state["editing_index"] = None
            try:
                st.experimental_rerun()
            except Exception:
                st.rerun()
    st.markdown("<p style='text-align:center;color:gray;font-size:13px;'>⚠️ Chatbot can make mistakes. © 2025 BAPAN GHOSH</p>", unsafe_allow_html=True)
else:
    prompt = st.chat_input("Type your message...")
    if prompt:
        chat["messages"].append({"role": "user", "content": prompt})
        if chat["title"] == "New Chat":
            st.session_state["chats"][cur]["title"] = prompt[:30] + "…" if len(prompt) > 30 else prompt
        st.session_state["chats"][cur] = chat
        try:
            st.experimental_rerun()
        except Exception:
            st.rerun()

st.markdown("<p id='app-footer'>🤖 Multi-Agent Edition • ⚠️ Chatbot may make mistakes • © 2025 BAPAN GHOSH</p>", unsafe_allow_html=True)

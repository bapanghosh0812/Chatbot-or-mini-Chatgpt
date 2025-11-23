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

# --- SAFE GENERATE (FINAL FIX) ---
def safe_generate(messages, system_instruction: str = "", max_retries: int = 2):
    # 1. Quick check: Is SDK loaded?
    # Hum global 'genai' check karenge bas confirmation ke liye
    if not globals().get("genai"):
        yield "[DEMO MODE] Gemini API Key missing or SDK not loaded.\n"
        return

    # 2. Local Import to avoid variable conflict
    # Kyunki tumne bahar 'genai' variable ko Model bana diya hai, hum library ko dobara import karenge
    import google.generativeai as lib_genai

    # 3. History Conversion (Streamlit -> Gemini Format)
    gemini_contents = []
    
    if isinstance(messages, str):
        gemini_contents.append({"role": "user", "parts": [messages]})
    elif isinstance(messages, list):
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            content = m.get("content", "")
            if content.strip(): 
                gemini_contents.append({"role": role, "parts": [content]})

    # 4. Helper to extract text
    def extract_text(obj):
        try:
            if hasattr(obj, "text"): return obj.text
            if isinstance(obj, dict): return obj.get("text", "")
            return str(obj)
        except: return ""

    last_exc = None
    
    # 5. Model Instance Creation (Correct Way)
    try:
        # Hum fresh library object (lib_genai) use kar rahe hain class create karne ke liye
        if system_instruction:
            model_instance = lib_genai.GenerativeModel(
                model_name=MODEL_NAME,
                system_instruction=system_instruction
            )
        else:
            model_instance = lib_genai.GenerativeModel(MODEL_NAME)
            
    except Exception as e:
        yield f"[System Error] Model setup failed: {e}"
        return

    # 6. Generate with History (Streaming)
    try:
        response = model_instance.generate_content(gemini_contents, stream=True)
        for chunk in response:
            txt = extract_text(chunk)
            if txt:
                yield txt
        return
    except Exception as e:
        last_exc = e

    # 7. Fallback: Non-streaming
    try:
        response = model_instance.generate_content(gemini_contents)
        txt = extract_text(response)
        if txt:
            yield txt
            return
    except Exception as e:
        last_exc = e

    yield f"[LLM ERROR] Could not generate response. Details: {str(last_exc)}"
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
from os import getenv
import traceback

# Primary place to put the key: env var GEMINI_API_KEY (or FALLBACK_HARDCODED_KEY for quick local dev)
# ---------------- GEMINI KEY (env preferred) ----------------
from os import getenv
import traceback

# Primary place to put the key
FALLBACK_HARDCODED_KEY = None
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or FALLBACK_HARDCODED_KEY
MODEL_NAME = os.getenv("MODEL_NAME") or "gemini-2.5-flash"

genai = None
model_available = False

if GEMINI_API_KEY:
    import os as _os
    _os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    try:
        import google.generativeai as _genai
        try:
            if hasattr(_genai, "configure"):
                _genai.configure(api_key=GEMINI_API_KEY)
        except Exception:
            pass
        
        # --- CHANGE IS HERE: Initialize the model object, not just the module ---
        genai = _genai.GenerativeModel(MODEL_NAME) 
        model_available = True
        logging.info(f"Gemini SDK loaded. Model: {MODEL_NAME}")
    except Exception as e:
        genai = None
        model_available = False
        logging.exception("Failed to import/configure google.generativeai: %s", e)
else:
    logging.warning("GEMINI_API_KEY not set. Running in demo mode.")

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
    """First agent: gather factual/analytical content using safe_generate."""
    if not (GEMINI_API_KEY and genai):
        return "[Research Agent offline – Gemini not configured]"
    try:
        messages = [
            {"role": "system", "content": "You are Research Agent. Collect structured reasoning, factual analysis, and relevant context."},
            {"role": "user", "content": f"Collect structured reasoning, factual analysis, and relevant context for:\n{prompt}"}
        ]
        chunks = list(safe_generate(messages))
        return "".join(chunks).strip()
    except Exception as e:
        logging.warning("Research agent failed: %s", e)
        return f"[Research Agent error: {e}]"

def run_summarizer_agent(context: str) -> str:
    """Second agent: summarize the research agent output using safe_generate."""
    if not (GEMINI_API_KEY and genai):
        return "[Summarizer Agent offline – Gemini not configured]"
    try:
        messages = [
            {"role": "system", "content": "You are Summarizer Agent. Summarize and simplify into 3-5 clear bullet points."},
            {"role": "user", "content": f"Summarize and simplify this research in 3-5 clear bullet points:\n\n{context}"}
        ]
        chunks = list(safe_generate(messages))
        return "".join(chunks).strip()
    except Exception as e:
        logging.warning("Summarizer agent failed: %s", e)
        return f"[Summarizer Agent error: {e}]"

# ---------------- model streaming (UPDATED) ----------------
def ask_gemini_smart_stream(user_text: str, system_prompt: str = "", history: List[dict] = None):
    # language detection / autocorrect / translation
    lang = detect_language(user_text) if detect else None
    corrected = autocorrect_text(user_text, lang)
    model_input = maybe_translate_for_model(corrected, lang)

    # 1. Build the full message list for the model
    # Start with existing history (excluding the very last user message because we act on 'user_text')
    messages_payload = []
    
    if history:
        # Deep copy to avoid messing up session state
        import copy
        # Filter out the current pending prompt if it's already in history to avoid duplication
        # But usually, we pass the HISTORY before the current prompt.
        messages_payload = copy.deepcopy(history)

    # Append the current processed/translated user prompt
    # Note: If the last message in history is already the user prompt, don't add it again.
    if not messages_payload or messages_payload[-1]["content"] != model_input:
        messages_payload.append({"role": "user", "content": model_input})

    acc = ""
    try:
        # Pass system_prompt explicitly as instruction, and the full messages payload
        for out in safe_generate(messages_payload, system_instruction=system_prompt):
            if out:
                acc += out
                yield out
    except Exception as e:
        logging.exception("ask_gemini_smart_stream failed: %s", e)
        yield f"[Error talking to model: {e}]"
        return

    # Translate back if needed
    try:
        orig_lang = lang or None
        if TRANSLATE_AVAILABLE and orig_lang and not orig_lang.startswith("en") and acc.strip():
            try:
                trans_back = translator.translate(acc, src="en", dest=orig_lang)
                translated_text = trans_back.text if hasattr(trans_back, "text") else str(trans_back)
                if translated_text and translated_text.strip() and translated_text.strip() != acc.strip():
                    yield "\n\n[Translated to your language]\n" + translated_text
            except Exception:
                pass
    except Exception:
        pass

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

# ---------------- URL query param handler (UPDATED) ----------------
qp = {}
try:
    qp = st.query_params
except Exception:
    try:
        qp = st.experimental_get_query_params()
    except Exception:
        qp = {}

# --- 1. NEW: HANDLE LOADING SHARED CHAT FROM LINK ---
# Yeh logic missing tha, isliye chat khali aa rahi thi
if "share" in qp:
    try:
        sid = qp["share"] if isinstance(qp["share"], str) else qp["share"][0]
        
        # Agar chat pehle se loaded nahi hai, toh file se load karo
        # Ya agar user kisi aur chat pe hai, toh switch karo
        loaded_data = load_shared_chat(sid)
        if loaded_data:
            st.session_state["chats"][sid] = loaded_data
            st.session_state["current"] = sid
            # URL clean karne ki zaroorat nahi taaki user refresh kare toh chat rahe
    except Exception as e:
        logging.error(f"Failed to load shared chat: {e}")

# --- 2. HANDLE EDIT PARAM ---
if "edit" in qp:
    try:
        raw = qp.get("edit")
        if isinstance(raw, list):
            raw = raw[0]
        edit_idx = int(str(raw).strip())
        st.session_state["editing_index"] = edit_idx
    except Exception:
        pass

    # URL clean karo
    new_qp = dict(qp)
    if "edit" in new_qp:
        del new_qp["edit"]
    try:
        st.query_params.clear()
        st.query_params.update(new_qp)
        st.rerun()
    except: pass

# --- 3. HANDLE CREATE SHARE PARAM ---
if "create_share" in qp:
    try:
        cur = st.session_state.get("current")
        chat = st.session_state["chats"].get(cur, {"title": "New Chat", "messages": []})
        sid = save_shared_chat(chat.get("title", "shared"), chat.get("messages", []))
        
        # Link banao
        link = f"http://localhost:8501/?share={sid}"
        st.session_state["share_link"] = link
        st.session_state["open_share_popup"] = True
    except Exception:
        logging.exception("Failed to create share link.")

    # URL clean karo
    new_qp = dict(qp)
    if "create_share" in new_qp:
        del new_qp["create_share"]
    try:
        st.query_params.clear()
        st.query_params.update(new_qp)
        st.rerun()
    except: pass

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
    # --- TELEPORT BUTTON (Header Share) ---
    if st.button("🔗 Share", key="teleport_share", help="Share Chat"):
        try:
            cur = st.session_state.get("current")
            chat = st.session_state["chats"].get(cur, {"title": "New Chat", "messages": []})
            sid = save_shared_chat(chat.get("title", "shared"), chat.get("messages", []))
            link = f"http://localhost:8501/?share={sid}"
            st.session_state["share_link"] = link
            st.session_state["open_share_popup"] = True
            st.rerun()
        except Exception:
            st.rerun()

    st.markdown("### 🤖 MINI CHATBOT")
    st.markdown("<div style='text-align: center; color: #666; font-weight: bold; margin-bottom: 10px;'>✨ Made by Bapan Ghosh ✨</div>", unsafe_allow_html=True)
    st.write("---")

    if st.button("➕ New Chat", use_container_width=True):
        nid = str(uuid.uuid4())
        st.session_state["chats"][nid] = {"title": "New Chat", "messages": []}
        st.session_state["current"] = nid
        st.session_state["editing_index"] = None
        st.rerun()

    st.write("")

    # --- HISTORY EXPANDER ---
    with st.expander("💾 Chat History", expanded=True):
        if not st.session_state["chats"]:
            st.caption("No history.")

        for cid, chat_item in list(st.session_state["chats"].items()):
            cols = st.columns([5, 1])
            with cols[0]:
                if st.button(chat_item["title"], key=f"open_{cid}", use_container_width=True):
                    st.session_state["current"] = cid
                    st.session_state["editing_index"] = None
                    if st.session_state.get("share_link"):
                        st.session_state.pop("share_link", None)
                    st.rerun()
            with cols[1]:
                if st.button("⋮", key=f"opts_{cid}", help="Options"):
                    if st.session_state.get("options_target") == cid:
                        st.session_state["options_target"] = None
                    else:
                        st.session_state["options_target"] = cid
                    st.rerun()

        # --- OPTIONS MENU (Rename/Delete/Share) ---
        # Yeh tabhi dikhega jab '...' dabaya ho
        if st.session_state.get("options_target"):
            t = st.session_state["options_target"]
            st.markdown("---")
            st.caption(f"Options: **{st.session_state['chats'][t]['title']}**")
            
            # Rename
            if st.button("✏️ Rename", key="opt_rename", use_container_width=True):
                st.session_state["rename_target"] = t
            
            # Delete
            if st.button("🗑️ Delete", key="opt_delete", use_container_width=True):
                if t in st.session_state["chats"]:
                    del st.session_state["chats"][t]
                if t == st.session_state["current"]:
                    st.session_state["current"] = next(iter(st.session_state["chats"]), str(uuid.uuid4()))
                    if st.session_state["current"] not in st.session_state["chats"]:
                        st.session_state["chats"][st.session_state["current"]] = {"title": "New Chat", "messages": []}
                st.session_state["options_target"] = None
                st.rerun()
            
            # Share (Copy Link) - CORRECTED INDENTATION
            if st.button("🔗 Share Chat", key="opt_share", use_container_width=True):
                chat_data = st.session_state["chats"][t]
                sid = save_shared_chat(chat_data["title"], chat_data["messages"])
                link = f"http://localhost:8501/?share={sid}"
                st.session_state["share_link"] = link
                st.session_state["open_share_popup"] = True
                st.session_state["options_target"] = None
                st.rerun()

        # Rename Input
        if st.session_state.get("rename_target"):
            rt = st.session_state["rename_target"]
            st.markdown("---")
            new_title = st.text_input("New Name:", st.session_state["chats"][rt]["title"], key="rename_input")
            if st.button("✅ Save Name", key="rename_save", use_container_width=True):
                if new_title.strip():
                    st.session_state["chats"][rt]["title"] = new_title.strip()
                st.session_state["rename_target"] = None
                st.rerun()

    st.write("---")
    
    # PDF Logic
    cur = st.session_state["current"]
    msgs = st.session_state["chats"][cur]["messages"]
    if msgs:
        try:
            pdf_bytes = create_pdf_bytes(st.session_state["chats"][cur]["title"], msgs)
            st.download_button("📥 Download Chat (PDF)", pdf_bytes,
                               file_name=f"chat.pdf",
                               mime="application/pdf",
                               use_container_width=True)
        except Exception:
            pass

    st.write("---")
    role = st.selectbox("Agent role", ["General Assistant", "Study Buddy", "Resume Coach", "Code Helper", "Summarizer"], key="role_select")
    st.session_state["agent_role"] = role

# ---------------- FINAL NATIVE POPUP (100% Working Copy & Close) ----------------
if st.session_state.get("open_share_popup") and st.session_state.get("share_link"):
    link = st.session_state['share_link']
    
    # 1. CSS: Isse hum Native Streamlit Box ko "Popup" jaisa banayenge
    st.markdown("""
    <style>
        /* Background ko dhundhla karna */
        .stApp {
            /* Background blur trick */
        }
        
        /* Expander ko screen ke beech mein fix karna (Popup Effect) */
        div[data-testid="stExpander"] {
            position: fixed !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) !important;
            z-index: 999999 !important;
            width: 400px !important;
            background-color: white !important;
            border-radius: 12px !important;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3) !important;
            border: 1px solid #e0e0e0 !important;
        }
        
        /* Expander ke upar ki line (Header) ko chupana */
        div[data-testid="stExpander"] summary {
            display: none !important;
        }
        
        /* Close button styling */
        div[data-testid="stExpander"] button {
            border: 1px solid #ddd;
            background: #f9f9f9;
            color: #333;
        }
        div[data-testid="stExpander"] button:hover {
            border: 1px solid #bbb;
            background: #eee;
            color: red;
        }
    </style>
    """, unsafe_allow_html=True)

    # 2. Content: Native Widgets (Jo kabhi fail nahi hote)
    # Hum 'expander' use kar rahe hain kyunki isko CSS se popup banana aasaan hai
    with st.expander("Share Popup", expanded=True):
        st.markdown("<h3 style='text-align:center; margin-top:0;'>Link Generated! 🎉</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:gray; font-size:13px;'>Share this link with others</p>", unsafe_allow_html=True)
        
        # --- ASLI COPY BUTTON (Top-Right corner pe icon aayega) ---
        st.code(link, language="text")
        
        st.caption("👆 The ‘copy’ icon is at the corner of the box.")
        
        # --- CLOSE BUTTON ---
        if st.button("❌ Close Popup", use_container_width=True):
            st.session_state["open_share_popup"] = False
            st.rerun()

# ---------------- FINAL HEADER & TELEPORT BUTTON ----------------
st.markdown("""
<style>
    /* 1. Header Strip (White Background) */
    .fixed-header {
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 60px;
        background: white;
        border-bottom: 1px solid #e0e0e0;
        z-index: 9000;
        display: flex;
        align-items: center;
        padding-left: 280px; /* Sidebar width */
    }
    
    /* 2. Hide Default Streamlit Header */
    header[data-testid="stHeader"] { display: none; }
    
    /* 3. Main Content Adjustment */
    .main .block-container { padding-top: 80px !important; }

    /* 4. TELEPORT MAGIC: Target the Share Button inside Sidebar */
    /* Hum sidebar ke 'Share' button ko pakad kar Header mein fix kar rahe hain */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] div.stButton:has(button[title="Share Chat"]) {
        position: fixed !important;
        top: 12px !important;
        right: 20px !important;
        width: auto !important;
        z-index: 10000 !important;
    }
    
    /* Styling the Teleported Button */
    [data-testid="stSidebar"] button[title="Share Chat"] {
        background-color: #0d6efd;
        color: white;
        border: none;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    [data-testid="stSidebar"] button[title="Share Chat"]:hover {
        background-color: #0b5ed7;
    }
    
    /* Mobile Fix */
    @media(max-width: 600px) {
        .fixed-header { padding-left: 60px; }
    }
</style>

<div class="fixed-header">
    <div style="font-size:20px; font-weight:bold; color:#333; margin-left:20px; display:flex; align-items:center; gap:8px;">
        <span>🤖</span> MINI CHATBOT
    </div>
</div>
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

# ---------------- MAIN CHAT AREA (Fixed Visibility & Position) ----------------

# 1. CSS MAGIC: Edit button ko chupane aur hover pe dikhane ke liye (Robust)
st.markdown("""
<style>
    /* By Default: Chat Message ke andar wala har Button invisible rahega */
    div[data-testid="stChatMessage"] div.stButton button {
        opacity: 0;
        transition: opacity 0.3s ease;
        border: none;
        background: transparent;
        color: gray;
    }

    /* Desktop Hover: Jab mouse message pe aaye, tab button dikhao */
    div[data-testid="stChatMessage"]:hover div.stButton button {
        opacity: 1;
        background: #f0f2f6; /* Thoda highlight */
    }

    /* Mobile: Hover nahi hota, isliye hamesha thoda sa dikhao */
    @media (hover: none) {
        div[data-testid="stChatMessage"] div.stButton button {
            opacity: 0.5 !important;
        }
        /* Touch karne pe full dikhe */
        div[data-testid="stChatMessage"] div.stButton button:active {
            opacity: 1 !important;
            background: #e0e0e0;
        }
    }
</style>
""", unsafe_allow_html=True)

chat_container = st.container()
cur = st.session_state["current"]
chat = st.session_state["chats"].get(cur, {"title": "New Chat", "messages": []})

# 2. CHAT HISTORY DISPLAY
with chat_container:
    for index, m in enumerate(chat["messages"]):
        with st.chat_message(m["role"]):
            # User Message Logic
            if m["role"] == "user":
                c1, c2 = st.columns([19, 1])
                with c1:
                    # Line breaks fix
                    st.markdown(m["content"].replace("\n", "  \n"))
                with c2:
                    # Button ab CSS se control hoga
                    if st.button("✏️", key=f"edit_btn_{index}", help="Edit"):
                        st.session_state["editing_index"] = index
                        st.rerun()
            # Assistant Message Logic
            else:
                st.markdown(m["content"])

# ---------------- EDITING UI (Fallback) ----------------
if st.session_state.get("editing_index") is not None:
    idx = st.session_state["editing_index"]
    
    # Validation
    if isinstance(idx, int) and 0 <= idx < len(chat["messages"]) and chat["messages"][idx]["role"] == "user":
        message_to_edit = chat["messages"][idx]["content"]
        
        st.markdown("---")
        st.markdown("### ✏️ Edit your message")
        edited_prompt = st.text_area("Edit:", value=message_to_edit, height=100, key="edit_text_area")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Save & Submit", use_container_width=True):
                chat["messages"][idx]["content"] = edited_prompt
                chat["messages"] = chat["messages"][:idx + 1]
                st.session_state["chats"][cur] = chat
                st.session_state["editing_index"] = None
                st.rerun()
        with c2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state["editing_index"] = None
                st.rerun()
        st.stop() # Stop execution to focus on edit box
    else:
        st.session_state["editing_index"] = None
        st.rerun()

# ---------------- INPUT & GENERATION LOGIC ----------------
else:
    import datetime
    import pytz
    import random
    import time
    import PIL.Image
    
    # --- 1. SETUP VARIABLES ---
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = str(random.randint(1000, 9999))
    
    # Initialize temp storage for current turn
    vision_content = None
    pdf_text_content = ""
    display_text = ""

    # --- 2. ATTACH BUTTON (Just above Input) ---
    # Simple Popover - No complex CSS needed, sits perfectly above input
    with st.container():
        col1, col2 = st.columns([0.1, 0.9])
        with col1:
            with st.popover("📎", help="Attach Image or PDF"):
                uploaded = st.file_uploader("Select File", 
                                          type=["png", "jpg", "jpeg", "pdf"], 
                                          key=st.session_state["uploader_key"])
                if uploaded:
                    st.toast(f"Attached: {uploaded.name}", icon="✅")

    # --- 3. INPUT HANDLING ---
    prompt = st.chat_input("Type your message...")
    
    if prompt:
        display_text = prompt
        
        # A. Handle File Logic (If file exists)
        if uploaded:
            # Case 1: Image
            if "image" in uploaded.type:
                img = PIL.Image.open(uploaded)
                vision_content = [prompt, img]
                display_text = f"{prompt}\n\n*(📸 Image Attached: {uploaded.name})*"
            
            # Case 2: PDF
            elif "pdf" in uploaded.type:
                try:
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(uploaded)
                    pdf_text = ""
                    for page in pdf_reader.pages: pdf_text += page.extract_text() + "\n"
                    pdf_text_content = f"PDF CONTENT ({uploaded.name}):\n{pdf_text}"
                    display_text = f"{prompt}\n\n*(📄 PDF Attached: {uploaded.name})*"
                except: pass
            
            # Reset Uploader for next time (Auto-Reset Logic)
            st.session_state["uploader_key"] = str(random.randint(1000, 9999))

        # B. Save State for Generation
        st.session_state["temp_vision_input"] = vision_content
        st.session_state["temp_pdf_content"] = pdf_text_content

        # C. Append to History & Rerun
        chat["messages"].append({"role": "user", "content": display_text})
        if chat["title"] == "New Chat":
            chat["title"] = prompt[:30] + "…" if len(prompt) > 30 else prompt
        st.session_state["chats"][cur] = chat
        st.rerun()

# ---------------- TYPING INDICATOR & STREAMING ----------------
if chat["messages"] and chat["messages"][-1]["role"] == "user":
    
    with chat_container:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            
            # Retrieve Temp Data
            vision_data = st.session_state.get("temp_vision_input")
            pdf_data = st.session_state.get("temp_pdf_content", "")

            # --- TERA ANIMATION FUNCTION (As Requested) ---
            def render_animation(text, icon):
                anim_html = f"""
                <div style='display:flex; gap:10px; align-items:center; font-weight:600; padding: 5px 0; color:#333; font-family:sans-serif;'>
                  <div>{icon} {text}</div>
                  <div id='dots' style='font-weight:600; color:#666;'>• • •</div>
                </div>
                <style>
                  @keyframes blink {{ 0% {{opacity:0.2}} 50% {{opacity:1}} 100% {{opacity:0.2}} }}
                  #dots span {{ animation: blink 1.2s infinite; }}
                </style>
                <script>
                  var d = document.getElementById('dots');
                  if (d) {{ d.innerHTML = "<span>•</span><span style='animation-delay:0.2s'>•</span><span style='animation-delay:0.4s'>•</span>"; }}
                </script>
                """
                with placeholder:
                    components.html(anim_html, height=45)

            # 1. Default Animation
            render_animation("Thinking", "🤔")
            
            reply_accum = ""
            # Clean message for logic checks
            user_msg = chat["messages"][-1]["content"].split("\n\n*(")[0]
            msg_lower = user_msg.lower()
            
            # Intent Logic
            is_image_gen = any(w in msg_lower for w in ["generate", "create", "bana de"]) and \
                           any(w in msg_lower for w in ["image", "pic", "photo", "apple"])
            is_search = any(w in msg_lower for w in ["score", "match", "live", "news", "vs", "price"]) and not is_image_gen

            try:
                # --- SCENARIO A: IMAGE GENERATION ---
                if is_image_gen:
                    render_animation("Creating Image", "🎨")
                    img_url = generate_fake_image(user_msg)
                    if not img_url: img_url = "https://picsum.photos/800/600"
                    
                    # Clear animation & Show Result
                    placeholder.empty()
                    st.markdown(f"Here is your image for: **{user_msg}**")
                    st.image(img_url, caption=user_msg)
                    reply_accum = f"Here is your image for: **{user_msg}**\n\n![Image]({img_url})"

# --- SCENARIO B: WEB SEARCH ---
                elif is_search:
                    render_animation("Searching Web", "🌐")
                    
                    try:
                        # Check if search_web function exists
                        if 'search_web' not in globals():
                            raise NameError("Search function not defined")

                        # Smart Query Cleaning
                        search_q = user_msg.replace("abhi", "").strip()
                        search_res = search_web(search_q)
                        
                        if not search_res: 
                            raise ValueError("No data found")
                        
                        render_animation("Analyzing", "⚡")
                        
                        # Prompt construction
                        IST = pytz.timezone('Asia/Kolkata')
                        current_time = datetime.datetime.now(IST).strftime("%d %b %Y, %I:%M %p")
                        full_prompt = f"SYSTEM: Time {current_time}. QUERY: {user_msg}. DATA: {search_res}. Answer accurately."
                        
                        # Stream Response
                        placeholder.empty()
                        st_stream = st.empty()
                        for chunk in ask_gemini_smart_stream(full_prompt):
                            reply_accum += chunk
                            st_stream.markdown(reply_accum + "▌")
                        st_stream.markdown(reply_accum)

                    except Exception as e:
                        # Agar search_web nahi mila ya koi error aaya, toh yeh fallback message aayega
                        placeholder.empty()
                        reply_accum = "I'm sorry ! I can't provide answer in this question, please Ask something else..."
                        st.markdown(reply_accum)

                # --- SCENARIO C: MULTI-AGENT (TERA LOGIC) ---
                elif 'is_research_query' in globals() and is_research_query(user_msg):
                    render_animation("Researching", "🧠")
                    research_out = run_research_agent(user_msg)
                    
                    render_animation("Summarizing", "📝")
                    summary_out = run_summarizer_agent(research_out)
                    
                    reply_accum = (
                        f"### 🧠 Research Findings\n\n{research_out}\n\n"
                        f"### 📝 Summary\n\n{summary_out}\n\n"
                        "🤖 Research & Summary Completed ✅"
                    )
                    placeholder.markdown(reply_accum)

# --- SCENARIO D: NORMAL STREAMING (TEXT / VISION / PDF) ---
                else:
                    role = st.session_state.get('agent_role', 'assistant')
                    IST = pytz.timezone('Asia/Kolkata')
                    current_time = datetime.datetime.now(IST).strftime("%d %b %Y, %I:%M %p")
                    
                    # System Prompt Setup
                    sys_p = f"You are a helpful {role}. Current Time: {current_time}.\n"
                    sys_p += "Response Guidelines:\n1. Use triple backticks for code.\n2. Be concise but helpful."
                    
                    # Inject PDF Context if exists
                    if pdf_data: sys_p += f"\n\nREFERENCE DOCUMENT CONTEXT:\n{pdf_data}\n(Answer based on this context if relevant)"
                    
                    mem = st.session_state.get("memory_summary", "")
                    if mem: sys_p += f"\nUser Summary Context: {mem}"

                    # Handling Vision vs Text
                    if vision_data:
                          render_animation("Analyzing Image", "👁️")
                          # Vision requests usually handle single turn best
                          response = genai.generate_content(vision_data)
                          reply_accum = response.text
                          placeholder.markdown(reply_accum)
                          st.session_state["temp_vision_input"] = None # Clear
                    else:
                        # Normal Text Stream with HISTORY
                        # We pass the existing chat history EXCLUDING the latest message
                        current_history = chat["messages"][:-1] 
                        
                        # Call updated function with history
                        for chunk in ask_gemini_smart_stream(user_msg, system_prompt=sys_p, history=current_history):
                            if chunk:
                                reply_accum += chunk
                                placeholder.markdown(reply_accum + "▌")
                        placeholder.markdown(reply_accum)

            # --- ERROR HANDLING ---
            except Exception as e:
                import traceback
                traceback.print_exc()
                reply_accum = f"⚠️ **Something went wrong:** {str(e)}"
                placeholder.error(reply_accum)

            # --- SAVE CONVERSATION HISTORY ---
            # Save whatever reply was generated (even error messages)
            if reply_accum:
                chat["messages"].append({"role": "assistant", "content": reply_accum})
                st.session_state["chats"][cur] = chat
                
                # Update memory summary
                if "update_memory_summary" in globals():
                    try:
                        st.session_state["memory_summary"] = update_memory_summary(chat["messages"])
                    except:
                        pass
                
                # Refresh UI
                st.rerun()
st.markdown("<p id='app-footer'>🤖 Multi-Agent Edition • ⚠️ Chatbot may make mistakes • © 2025 BAPAN GHOSH</p>", unsafe_allow_html=True)
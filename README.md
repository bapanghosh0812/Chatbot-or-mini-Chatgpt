									  🚀 MINI CHATBOT — Multi-Agent Edition
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------														
Author: Bapan Ghosh
Main App: /mnt/data/chatbot_app.py
Mode: Works Offline (Demo) + Online (Gemini)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
											🌟 Overview
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
This project is a Streamlit-based Multi-Agent Chatbot designed for the Agents Intensive Capstone.
It includes:

🧠 Research Agent → Performs structured factual reasoning

✍️ Summarizer Agent → Converts research into clean bullet-points

🔤 Autocorrect + Language Detection + Auto-Translation

📄 PDF Export of full chats

🔗 Share & Load Chats (shared_chats.json)

🖊️ Edit previous messages

🎨 Clean UI with persistent state

💬 Runs in Demo Mode (no API key needed) or Gemini Mode

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🚀 Quick Start (2 Steps)
**************************

1️⃣ Install Required Packages

pip install -r requirements.txt

2️⃣ Run the App

Demo Mode (no API key required):

streamlit run /mnt/data/chatbot_app.py


Gemini Mode (optional):
**********************
export GEMINI_API_KEY="your_key_here"
streamlit run /mnt/data/chatbot_app.py

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧩 Features
***************

1. Multi-Agent Architecture

Research Agent: Generates deep, structured analysis

Summarizer Agent: Converts long analysis into clear 3–5 bullet points

Both agents are routed automatically based on prompt intent


2. Robust Gemini Integration

Smart fallback: if SDK/key missing → auto Demo Mode

safe_generate() avoids crashes by trying:

Streaming generation

Non-streaming

Full fallback with readable error


3. Smart Pre-Processing

Detects language

Auto-corrects English

Auto-translates non-English to English

Final response translated back to original language when required


4. Share & Edit

Chats saved in shared_chats.json

Generate shareable chat links

Edit any previous user message (version-safe)


5. Export Chat as PDF

Beautiful layout

Automatic line wrapping

Supports NotoSans font for multilingual text

Includes signature footer: “✨ Made by Bapan Ghosh ✨”

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🎛️ File Structure
***********************

		File			        |		    Purpose
------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------
chatbot_app.py					|		Main Streamlit application
requirements.txt				|		All dependencies
shared_chats.json				|		Stored chats / share history
NotoSans-Regular.ttf				|		Font for multilingual PDF
						|
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔍 Evaluation (How judges can verify)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Use these prompts inside the chatbot:
*****************************************
1) Research Task:
“Research the future of agent-based AI systems and summarize insights.”

2) Planning Task:
“Create a 3-step study plan for learning Python automation.”

3) Editing Task:
“Improve this sentence: ‘Worked on ML models for data.’”

Then export all 3 chats as PDFs using the “Download as PDF” button.
These serve as reproducible outputs.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🏗️ Architecture (High Level)
***********************************

User Input
   ↓
Language Detection → Autocorrect → Translation
   ↓
Intent Check → (General Chat OR Multi-Agent Flow)
   ↓
safe_generate() with fallback options
   ↓
Post-translate → Streamed Output to UI
   ↓
Save → Edit → Share → PDF Export
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

⚠️ Known Limitations
***************************

Online Gemini quality depends on API key availability

Some old debug chats in shared_chats.json may include historical error strings

Long-term memory not included (session-based only)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

✨ Author
Bapan Ghosh
(Also embedded inside the app footer & PDFs)
🎙️ AI Twin Voice Bot
End-to-End Audio Intelligence • Ultra-Low Latency • Persona-Driven Agentic System






A production-grade, fully asynchronous voice AI agent that mimics a specific persona (Parthiv S) to answer interview-style questions.
Engineered for zero budgeting, near-instant responses, and strict hallucination control.

This bot combines a Hybrid Memory Architecture, massive-context reasoning, and resilient audio streaming to deliver a natural, real-time conversational experience.

🚀 Features & Engineering Highlights
🔥 1. Hybrid Memory Architecture (Instant + Deep Reasoning)

The bot chooses between two thinking modes:

⚡ Fast Path (0ms latency) — RAM “Golden Answers”

For predictable queries like:

“Tell me about yourself”

“What are your strengths?”

A semantic router instantly maps the intent → RAM audio cache.

No DB calls

No LLM calls

No TTS calls

Just pure instant audio playback.

🧠 Slow Path (Deep Research Mode) — Massive Context LLM

For complex queries like:
“Explain your AI Market Intelligence pipeline end-to-end.”

A 120B “Researcher” model synthesizes the answer using:

persona.json

project_chunks.txt

Context window (>200k tokens)

Strict persona identity constraints

🔐 2. Defense-in-Depth Language Guardrail System

To stop Spanish drift or LLM hallucination:

Whisper STT forced to English

langdetect validator checks every LLM output

If drift detected → “Fixer LLM” (Llama-8b) auto-translates

Summarizer enforces:

4–5 sentences

70–80 words

No preambles

No broken sentences

This stack ensures the bot sounds clean, precise, and perfectly in-character.

🎛️ 3. Resiliency Cascades (Zero-Failure Audio)
✔ Multi-key Groq TTS failover

If one API key fails → automatically switches to next.

✔ Full fallback to Google gTTS

Even if all Groq keys fail → the bot still speaks.

✔ Token Tracker

Prevents API throttling by estimating tokens per TTS request.

🏗️ System Architecture
graph TD
    User[User Microphone Input] --> STT[Whisper Large V3]
    STT --> Router{Semantic Router<br>Llama 3.1 8B}

    Router -- "Known Intent" --> RAM[RAM Audio Cache<br>(Preloaded from Supabase)]
    Router -- "Unknown / Deep Query" --> Researcher[Researcher LLM (GPT-OSS-120B)]

    Researcher --> Summarizer[Summarizer Agent (Llama-70B)]
    Summarizer --> Validator[Language Drift Validator]

    RAM --> Stream[Audio Stream ↦ UI]
    Validator --> TTS[Groq PlayAI / Multi-Key Failover]
    TTS --> Stream

    Stream --> UI[Frontend<br>HTML+JS Glassmorphic UI]

🛠️ Tech Stack
Backend

Python 3.11

FastAPI

Uvicorn

AsyncOpenAI SDK

Groq Cloud (Whisper / Llama / GPT-OSS / PlayAI TTS)

Memory & Storage

Supabase (PostgreSQL)

Base64 audio storage

RAM hydration on startup

Frontend

Vanilla JS

Glassmorphism UI

Streaming audio playback

Render “wake-up” ping endpoint

Deployment

Render (Backend)

GitHub Pages / Vercel (Frontend)

⚙️ Installation & Setup
1. Clone + Environment Setup
git clone https://github.com/parthivqw/voice-bot.git
cd voice-bot/backend
python -m venv venv
venv\Scripts\activate   # On Windows
pip install -r requirements.txt

2. Configure .env

Create this file inside /backend:

# Groq API Keys
GROQ_API_KEY_1=gsk_...
GROQ_API_KEY_2=gsk_...
GROQ_API_KEY_3=gsk_...

# Supabase
SUPABASE_URL=https://yourproject.supabase.co
SUPABASE_KEY=public_anon_key
SUPABASE_SERVICE_ROLE_KEY=service_role_key

3. Seed the Memory (Generate “Golden Audio”)

This script generates:

Base64 audio files

Canonical Q/A answers

Uploads to Supabase

Prepares RAM cache

python seeder.py

4. Run the Backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload


Look for:

🚀 Cache Hydrated: X items loaded into RAM.

🧩 API Endpoints
🎤 POST /chat

Send raw audio → receive streaming audio response.

✍️ POST /chat/text

Silent mode (text input → audio stream)

🌡️ GET /health

Used by frontend to wake Render free tier.

🧠 Engineering Rationale
1. ❌ Why Not RAG?

RAG required:

FAISS index in memory

Embedding model

1GB RAM usage

Render free tier = 512MB RAM → instant OOM crashes.

Solution:
Used massive context prompting instead of embeddings.

2. ⚡ Why Supabase + RAM?

DB fetch latency = 400–800ms
RAM lookup = <1ms

Golden audio = ~2MB ⇒ perfect for memory hydration.

3. 📝 “Header Hijack” Trick

We inject the transcript into this header:

X-AI-Response-Text


Allows frontend to show “typewriter effect”
without WebSockets.

🔮 Future Upgrades

 HeyGen avatar video synthesis

 LangGraph rewrite (Node-based agentic reasoning)

 Socket.IO interruption support

 Telemetry dashboards (router accuracy vs fallback rate)

👨‍💻 Author
Parthiv S

AI/ML Engineer • Multi-Agent Systems • GenAI Specialist
Kerala, India

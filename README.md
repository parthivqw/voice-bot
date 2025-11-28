# 🎙️ AI Twin Voice Bot

### **End-to-End Audio Intelligence • Ultra-Low Latency • Persona-Driven Agentic System**

A production-grade, fully asynchronous voice AI agent that mimics a specific persona (Parthiv S) to answer interview-style questions.
Engineered for **zero-budgeting**, **near-instant responses**, and **strict hallucination control**.

This bot combines a **Hybrid Memory Architecture**, **massive-context reasoning**, and **resilient audio streaming** to deliver a natural, real-time conversational experience.

---

## 🚀 Features & Engineering Highlights

### 🔥 1. Hybrid Memory Architecture (Instant + Deep Reasoning)

The bot chooses between two thinking modes:

#### ⚡ **Fast Path (0ms latency) — RAM "Golden Answers"**

For predictable queries like:

* "Tell me about yourself"
* "What are your strengths?"

A semantic router instantly maps the intent → RAM audio cache.

* No DB calls
* No LLM calls
* No TTS calls

👉 **Pure instant audio playback.**

#### 🧠 **Slow Path (Deep Research Mode) — 120B Researcher Model**

When a question is complex or unique, the system invokes the massive-context researcher model and synthesizes a personalized answer using:

* `persona.json`
* `project_chunks.txt`
* 200k+ token context
* Strict identity enforcement

---

## 🔐 2. Defense-in-Depth Language Guardrails

Stops Spanish drift or hallucinations with:

* Forced-English Whisper STT
* `langdetect` validation
* Auto-correction via Llama-8B Translation/Fixer
* Summarizer with:

  * 4–5 sentences
  * 70–80 words
  * No preamble keywords
  * No incomplete sentences

This ensures **clean, English-only, controlled** voice output.

---

## 🎛️ 3. Resiliency Cascades (Zero-Failure Audio)

* ✔ Multiple Groq API key failover
* ✔ Fallback to gTTS (Google TTS)
* ✔ Custom TokenTracker for TPM enforcement
* ✔ Handles Render cold starts gracefully

---



---

## 🛠️ Tech Stack

### **Backend**

* Python 3.11
* FastAPI
* Uvicorn
* Async Groq API SDK

### **AI Models**

* Whisper Large V3
* Llama 3.1 8B (router)
* Llama 3.3 70B (summarizer)
* GPT-OSS-120B (researcher)
* PlayAI TTS

### **Storage**

* Supabase (PostgreSQL)
* Base64 encoded audio blobs
* RAM hydration at startup

### **Frontend**

* Vanilla JS
* HTML
* CSS Glassmorphism
* Streaming audio playback

### **Deployment**

* Render (Backend)
* GitHub Pages / Vercel (Frontend)

---

## ⚙️ Local Setup

### 1. Clone Repo
```bash
git clone https://github.com/parthivqw/voice-bot.git
cd voice-bot/backend
```

---

### 2. Environment Setup
```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

### 3. Create `.env`
```
# Groq
GROQ_API_KEY_1=gsk_...
GROQ_API_KEY_2=gsk_...
GROQ_API_KEY_3=gsk_...

# Supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=public_anon_key
SUPABASE_SERVICE_ROLE_KEY=service_role_key
```

---

### 4. Seed Memory (Generate "Golden Audio")
```bash
python seeder.py
```

---

### 5. Run Backend Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
🚀 Cache Hydrated: X items loaded into RAM.
```

---

## 🧪 API Endpoints

### **POST /chat**

Audio → Audio (stream)

### **POST /chat/text**

Text → Audio (stream)

### **GET /health**

Render warm-up ping

---

## 🧠 Engineering Rationale

### ❌ Why Not RAG?

Render free tier = **512MB**
RAG with embeddings + FAISS = **1GB+ RAM**

→ Crashes instantly
→ Switched to **Massive Context Prompting** (120B)

### ⚡ Why RAM Caching?

Supabase fetch latency = 400–800ms
RAM lookup = <1ms

→ 400× faster responses

### 📝 Why Header Hijack?

We embed text response in HTTP header:

`X-AI-Response-Text`

→ Enables typewriter effect **without WebSockets**

---

## 🔮 Future Roadmap

* [ ] HeyGen Video Avatar
* [ ] LangGraph multi-agent refactor
* [ ] Socket.IO interrupt support
* [ ] Telemetry dashboards

---

## 👨‍💻 Author

**Parthiv S**
AI/ML Engineer • Multi-Agent Systems • GenAI Specialist

---

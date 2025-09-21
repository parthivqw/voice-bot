import os
import json
# import faiss  # Commented out for slim deployment
# from sentence_transformers import SentenceTransformer # Commented out for slim deployment
import numpy as np
from openai import AsyncOpenAI
from gtts import gTTS
import io

# --- MODIFIED: Simplified Startup ---
# We no longer load the heavy RAG models, only the text and persona files.
try:
    print("Loading AI services (Massive Context Version)...")
    
    # --- RAG Models (Commented Out) ---
    # print("Loading embedding model and FAISS index...")
    # embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # faiss_index = faiss.read_index('faiss_index.bin')
    
    with open('project_chunks.txt', 'r', encoding='utf-8') as f:
        # Read the entire project knowledge base into a single string
        project_kb = f.read()
    
    with open('persona.json', 'r') as f:
        persona_data = json.load(f)
    persona_prompt = json.dumps(persona_data, indent=2)

    # --- Initialize clients for TTS cascade ---
    groq_api_keys = [
        os.environ.get("GROQ_API_KEY_1"),
        os.environ.get("GROQ_API_KEY_2"),
        os.environ.get("GROQ_API_KEY_3"),
    ]
    groq_clients = [
        AsyncOpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        for key in groq_api_keys if key
    ]
    if not groq_clients:
        raise ValueError("No Groq API keys found in .env file.")
    
    main_client = groq_clients[0]
    print(f"✅ AI services (massive context) loaded successfully with {len(groq_clients)} Groq clients.")
except Exception as e:
    print(f"❌ CRITICAL ERROR during AI service initialization: {e}")
    project_kb = persona_prompt = main_client = groq_clients = None # embedding_model = faiss_index = None

# --- NEW: Massive System Prompt ---
# We inject the ENTIRE project knowledge base directly into the prompt.
SYSTEM_PROMPT_TEMPLATE = f"""
You are the AI Twin of Parthiv S. Your personality, history, and core knowledge are defined by the following JSON object. You MUST answer from this perspective, in this style.

### Persona & Core Knowledge:
{persona_prompt}

You also have access to the full, detailed reports of all of Parthiv's projects. When a question is asked about a project, you MUST use the following text as your source of truth to provide a specific, in-depth answer. Synthesize this information as if it's your own direct memory.

### Detailed Project Knowledge Base:
---
{project_kb}
---

---
**CRITICAL RESPONSE CONSTRAINT FOR VOICE OUTPUT:**
You MUST keep your answers concise and conversational. Aim for a **maximum of 150 words**. For lists or tables, summarize them into a few key bullet points. Do NOT generate overly long, multi-paragraph responses as this will fail the text-to-speech engine. Be brief and impactful.
---
"""

# --- HELPER 1: Speech-to-Text (Unchanged) ---
async def get_text_from_speech(audio_bytes: bytes) -> str:
    print("Transcribing audio with Whisper...")
    try:
        transcription = await main_client.audio.transcriptions.create(
            file=("request.wav", audio_bytes, "audio/wav"),
            model="whisper-large-v3",
        )
        user_text = transcription.text
        print(f"Transcription complete: '{user_text}'")
        return user_text
    except Exception as e:
        print(f"❌ ERROR during transcription: {e}")
        return "I'm sorry, I had trouble understanding what you said. Could you please try again?"

# --- HELPER 2: The LLM Brain (Now Simplified) ---
async def get_ai_response_text(user_question: str) -> str:
    print("Generating response using massive context prompt...")
    try:
        # --- The RAG logic is now commented out ---
        # retrieved_context = ""
        # is_proj_related = await is_project_related(user_question)
        # if is_proj_related:
        #     print("Project-related intent detected. Running RAG search...")
        #     query_embedding = embedding_model.encode([user_question]).astype('float32')
        #     distances, indices = faiss_index.search(query_embedding, k=1)
        #     if len(indices) > 0 and indices[0][0] != -1:
        #         retrieved_context = project_chunks[indices[0][0]]
        #         print("Retrieved context: Project details found.")
        
        # final_system_prompt = SYSTEM_PROMPT_TEMPLATE
        # if retrieved_context:
        #     final_system_prompt += f"\nADDITIONAL CONTEXT...\n{retrieved_context}\n---"
        
        chat_completion = await main_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE}, # Using the massive context prompt directly
                {"role": "user", "content": user_question},
            ],
            model="openai/gpt-oss-120b",
            temperature=0.7,
            max_tokens=300, # Constrained for TTS limits
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"❌ ERROR during main LLM call: {e}")
        return "I seem to be having trouble thinking right now. Please try again."

# --- Intent Detection Helper (Commented Out) ---
# async def is_project_related(question: str) -> bool:
#     try:
#         completion = await main_client.chat.completions.create(
#             model="llama-3.1-8b-instant", 
#             messages=[{...}],
#             ...
#         )
#         response_json = json.loads(completion.choices[0].message.content)
#         return response_json.get("is_project_related", False)
#     except Exception as e:
#         print(f"Intent detection failed: {e}")
#         return False

# --- HELPER 3: Text-to-Speech Cascade (Unchanged) ---
async def get_speech_from_text(text: str):
    # This function with the TTS cascade is unchanged
    for i, client in enumerate(groq_clients):
        try:
            print(f"Generating speech with Groq TTS (Client {i+1})...")
            response = await client.audio.speech.create(
                model="playai-tts-1",
                voice="aura-asteria-en",
                input=text,
            )
            return (chunk for chunk in response.iter_bytes())
        except Exception as e:
            print(f"❌ Groq TTS (Client {i+1}) failed: {e}")
            continue

    try:
        print("⚠️ All Groq TTS clients failed. Falling back to gTTS.")
        tts = gTTS(text=text, lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return iter([fp.read()])
    except Exception as e:
        print(f"❌ FINAL FALLBACK FAILED: gTTS error: {e}")
        return None

# --- MASTER PIPELINE (Unchanged) ---
async def process_audio_query(audio_bytes: bytes):
    transcribed_text = await get_text_from_speech(audio_bytes)
    if not transcribed_text: return None
    
    response_text = await get_ai_response_text(transcribed_text)
    if not response_text: return None
        
    audio_iterator = await get_speech_from_text(response_text)
    return audio_iterator


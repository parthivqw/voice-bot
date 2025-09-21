import os
import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import AsyncOpenAI
from gtts import gTTS
import io

# --- This part is updated to handle multiple clients ---
try:
    print("Loading AI services...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    faiss_index = faiss.read_index('faiss_index.bin')
    with open('project_chunks.txt', 'r', encoding='utf-8') as f:
        project_chunks = [chunk for chunk in f.read().split('\n===\n') if chunk]
    
    with open('persona.json', 'r') as f:
        persona_data = json.load(f)
    persona_prompt = json.dumps(persona_data, indent=2)

    # --- NEW: LOAD MULTIPLE GROQ CLIENTS FROM .env ---
    # Expects GROQ_API_KEY_1, GROQ_API_KEY_2, etc. in your .env file
    groq_api_keys = [
        os.environ.get("GROQ_API_KEY_1"),
        os.environ.get("GROQ_API_KEY_2"),
        os.environ.get("GROQ_API_KEY_3"),
        os.environ.get("GROQ_API_KEY_4")
    ]
    groq_clients = [
        AsyncOpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        for key in groq_api_keys if key
    ]
    if not groq_clients:
        raise ValueError("No Groq API keys found. Please check your .env file for GROQ_API_KEY_1, etc.")
    
    # Use the first client for non-TTS tasks like transcription and LLM calls
    client = groq_clients[0]

    print(f"✅ AI services loaded successfully with {len(groq_clients)} Groq clients.")
except Exception as e:
    print(f"❌ CRITICAL ERROR during AI service initialization: {e}")
    embedding_model = faiss_index = project_chunks = persona_prompt = client = groq_clients = None

SYSTEM_PROMPT_TEMPLATE = f"""
You are the AI Twin of Parthiv S. Your personality, history, and core knowledge are defined by this JSON object. You MUST answer from this perspective, in this style.
{persona_prompt}

---
**CRITICAL RESPONSE CONSTRAINT FOR VOICE OUTPUT:**
You MUST keep your answers concise and conversational. Aim for a **maximum of 150 words**. For lists or tables, summarize them into a few key bullet points. Do NOT generate overly long, multi-paragraph responses as this will fail the text-to-speech engine. Be brief and impactful.
---
"""

# --- NEW: End-to-End Audio Pipeline ---
async def process_audio_query(audio_bytes: bytes):
    """
    This is the new main pipeline function.
    It handles the full STT -> RAG/LLM -> TTS flow.
    """
    # 1. Speech-to-Text (STT) with Groq Whisper
    user_text = await get_text_from_speech(audio_bytes)

    # 2. Get Text Response using our existing RAG/LLM logic
    response_text = await get_text_response(user_text)

    # 3. Text-to-Speech (TTS) with Groq PlayAI and fallback
    audio_iterator = await get_speech_from_text(response_text)
    
    return audio_iterator


# --- HELPER FUNCTION 1: Speech-to-Text ---
async def get_text_from_speech(audio_bytes: bytes) -> str:
    print("Transcribing audio with Whisper...")
    try:
        transcription = await client.audio.transcriptions.create(
            file=("request.wav", audio_bytes, "audio/wav"),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
        )
        user_text = transcription.text
        print(f"Transcription complete: '{user_text}'")
        return user_text
    except Exception as e:
        print(f"❌ ERROR during transcription: {e}")
        return "I'm sorry, I had trouble understanding what you said. Could you please try again?"


# --- HELPER FUNCTION 2: The Original RAG/LLM Logic ---
async def get_text_response(user_question: str) -> str:
    if not all([embedding_model, faiss_index, project_chunks, client]):
        raise RuntimeError("AI services are not properly initialized.")

    retrieved_context = ""
    is_proj_related = await is_project_related(user_question)
    
    if is_proj_related:
        print("Project-related intent detected. Running RAG search...")
        query_embedding = embedding_model.encode([user_question]).astype('float32')
        distances, indices = faiss_index.search(query_embedding, k=1)
        
        if len(indices) > 0 and indices[0][0] != -1:
            retrieved_context = project_chunks[indices[0][0]]
            print("Retrieved context: Project details found.")
    
    final_system_prompt = SYSTEM_PROMPT_TEMPLATE
    if retrieved_context:
        final_system_prompt += f"""
ADDITIONAL CONTEXT: You MUST use the following retrieved information to answer. Synthesize it as if it's your own direct memory.

### Retrieved Project Details:
---
{retrieved_context}
---
"""
    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": user_question},
            ],
            model="openai/gpt-oss-120b",
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"❌ ERROR during main LLM call: {e}")
        return "I seem to be having trouble thinking right now. Please try again."

# --- HELPER FUNCTION 3: Intent Detection ---
async def is_project_related(question: str) -> bool:
    try:
        completion = await client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[{
                "role": "system",
                "content": """Is the user's question about a technical project, work experience, or implementation detail? Respond with ONLY a JSON object: {"is_project_related": boolean}"""
            }, { "role": "user", "content": question }],
            temperature=0.0,
            max_tokens=50,
            response_format={"type": "json_object"}
        )
        response_text = completion.choices[0].message.content
        print(f"Intent detection for '{question}': {response_text}")
        response_json = json.loads(response_text)
        return response_json.get("is_project_related", False)
    except Exception as e:
        print(f"Intent detection failed: {e}")
        return False

# --- HELPER FUNCTION 4: Text-to-Speech (with Fallback Cascade) ---
async def get_speech_from_text(text: str):
    # Attempt 1, 2, 3: Cycle through all available Groq clients
    for i, tts_client in enumerate(groq_clients):
        try:
            print(f"Generating speech with Groq TTS (Client {i+1})...")
            speech_response = await tts_client.audio.speech.create(
                model="playai-tts",
                voice="Mason-PlayAI",
                input=text,
                response_format="wav",
            )
            print("✅ Groq TTS successful.")
            # We return an iterator that can be streamed back to the client
            return speech_response.iter_bytes()
        except Exception as e:
            print(f"❌ Groq TTS (Client {i+1}) failed: {e}")
            continue # Try the next key

    # Final Fallback: Use gTTS if all Groq clients fail
    try:
        print("⚠️ All Groq TTS clients failed. Falling back to gTTS.")
        tts = gTTS(text=text, lang='en', slow=False)
        # Save the audio to an in-memory binary stream
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        # Return the raw bytes, wrapped in an iterator to match the stream format
        print("✅ gTTS fallback successful.")
        return iter([fp.read()])
    except Exception as e:
        print(f"❌ FINAL FALLBACK FAILED: gTTS error: {e}")
        return None


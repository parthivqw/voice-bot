import os
import json
import numpy as np
from openai import AsyncOpenAI
from gtts import gTTS
import io
import time
from collections import deque
from typing import Optional
import re

# Token tracking for TPM limits
class TokenTracker:
    def __init__(self, max_tokens_per_minute=1000):  # Conservative limit (1000 < 1200)
        self.max_tokens_per_minute = max_tokens_per_minute
        self.requests = deque()  # Store (timestamp, tokens) tuples
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using the 1 token ≈ 4 characters heuristic"""
        # Clean up text and count more accurately
        clean_text = re.sub(r'\s+', ' ', text.strip())  # Normalize whitespace
        char_count = len(clean_text)
        estimated_tokens = int(char_count / 4) + 20  # +20 buffer for overhead/punctuation
        return estimated_tokens
    
    def can_make_request(self, text: str) -> tuple[bool, int]:
        """Check if we can make a TTS request without hitting TPM limits"""
        current_time = time.time()
        estimated_tokens = self.estimate_tokens(text)
        
        # Remove requests older than 60 seconds
        while self.requests and current_time - self.requests[0][0] > 60:
            self.requests.popleft()
        
        # Calculate tokens used in the last minute
        tokens_used = sum(tokens for _, tokens in self.requests)
        
        # Check if adding this request would exceed limit
        if tokens_used + estimated_tokens > self.max_tokens_per_minute:
            return False, tokens_used
        
        return True, tokens_used
    
    def record_request(self, text: str):
        """Record a successful request"""
        current_time = time.time()
        estimated_tokens = self.estimate_tokens(text)
        self.requests.append((current_time, estimated_tokens))

# Global token tracker instance
token_tracker = TokenTracker()

# --- MODIFIED: Simplified Startup for "Massive Context" ---
try:
    print("Loading AI services (Massive Context Version)...")
    
    with open('project_chunks.txt', 'r', encoding='utf-8') as f:
        project_kb = f.read()
    
    with open('persona.json', 'r') as f:
        persona_data = json.load(f)
    persona_prompt = json.dumps(persona_data, indent=2)

    # --- Initialize clients for TTS cascade ---
    groq_api_keys = [
        os.environ.get("GROQ_API_KEY_1"),
        os.environ.get("GROQ_API_KEY_2"),
        os.environ.get("GROQ_API_KEY_3"),
        os.environ.get("GROQ_API_KEY_4"),
        os.environ.get("GROQ_API_KEY_5"),
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
    project_kb = persona_prompt = main_client = groq_clients = None

# --- Prompts ---
RESEARCHER_PROMPT_TEMPLATE = f"""
You are the AI Twin of Parthiv S. Your personality, history, and core knowledge are defined by the following JSON object. 
When a question is asked about a project, you MUST use the provided detailed project knowledge base as your source of truth.
Your task is to generate a complete, detailed, and comprehensive answer to the user's question, synthesizing all available information as if it's your own direct memory. Do not worry about length.

### Persona & Core Knowledge:
{persona_prompt}

### Detailed Project Knowledge Base:
---
{project_kb}
---
"""

SUMMARIZER_SYSTEM_PROMPT = """
You are a voice assistant scriptwriter. Convert detailed text into concise, conversational TTS scripts.

**CRITICAL RULES:**
1.Strict rules to produce concise, TTS-friendly scripts (80–100 words).
2. Speak directly to user ("you", "I")  
3. NO meta-commentary ("Here's the summary", "Sure", etc.)
4. Start directly with content
5. Professional yet conversational tone
6. End with complete sentences (no cutoffs)

Provide only the final spoken response.
"""

# --- Speech-to-Text ---
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
        return "I'm sorry, I had trouble understanding what you said."

# --- LLM Brain with Token-Safe Summarization ---
async def get_ai_response_text(user_question: str) -> str:
    print("Step 1: Generating detailed response with Researcher model...")
    try:
        # CALL 1: The Researcher
        researcher_completion = await main_client.chat.completions.create(
            messages=[
                {"role": "system", "content": RESEARCHER_PROMPT_TEMPLATE},
                {"role": "user", "content": user_question},
            ],
            model="openai/gpt-oss-120b",
            temperature=0.7,
            max_tokens=2048,
        )
        detailed_text = researcher_completion.choices[0].message.content
        print("✅ Detailed response generated.")
        print(f"📊 Detailed response length: {len(detailed_text)} chars, ~{len(detailed_text.split())} words")

        print("Step 2: Summarizing for voice with token safety...")
        # CALL 2: The Summarizer with aggressive limits
        summarizer_completion = await main_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Summarize this response for voice output:\n\n{detailed_text}"}
            ],
            model="llama-3.3-70b-versatile",  # Better instruction following than llama
            temperature=0.2,
            max_tokens=144,  # Very conservative - ~70 words max
        )
        concise_text = summarizer_completion.choices[0].message.content.strip()
        
        # Remove any meta-commentary that might sneak through
        concise_text = re.sub(r'^(Sure,?\s*|Here\'?s?\s*|Let me\s*|I\'?ll\s*)', '', concise_text, flags=re.IGNORECASE)
        concise_text = concise_text.strip()
        
        print("✅ Concise voice response generated.")
        print(f"📊 Summarized response: {len(concise_text)} chars, ~{len(concise_text.split())} words")
        
        # Estimate tokens for logging
        estimated_tokens = token_tracker.estimate_tokens(concise_text)
        print(f"🔍 Estimated tokens: {estimated_tokens}")
        print(f"🔍 FINAL OUTPUT: '{concise_text}'")
        
        # Final safety check - if still too long, aggressively trim
        word_count = len(concise_text.split())
        if word_count > 65:  # Conservative
            words = concise_text.split()
            concise_text = ' '.join(words[:60]) + "."
            print(f"⚠️ Response truncated to 60 words for maximum TTS safety")
        
        return concise_text

    except Exception as e:
        print(f"❌ ERROR during main LLM chain: {e}")
        return "I'm having trouble processing that right now. Please try again."

# --- Token-Safe TTS with Rate Limiting ---
async def get_speech_from_text(text: str):
    # Check token limits before making TTS request
    can_request, tokens_used = token_tracker.can_make_request(text)
    estimated_tokens = token_tracker.estimate_tokens(text)
    
    print(f"🎙️ TTS Token Check:")
    print(f"   Text: '{text}'")
    print(f"   Length: {len(text)} chars, ~{len(text.split())} words")
    print(f"   Estimated tokens: {estimated_tokens}")
    print(f"   Tokens used in last minute: {tokens_used}")
    print(f"   Can make request: {can_request}")
    
    if not can_request:
        wait_time = 60 - (time.time() % 60) + 5  # Wait until next minute + buffer
        print(f"⚠️ TPM limit would be exceeded. Would need to wait ~{wait_time:.0f} seconds.")
        return None  # Or implement queuing/waiting logic
    
    # Try TTS cascade
    for i, client in enumerate(groq_clients):
        try:
            print(f"Generating speech with Groq TTS (Client {i+1})...")
            response = await client.audio.speech.create(
                model="playai-tts",
                voice="Mason-PlayAI",
                input=text,
            )
            # Record successful request
            token_tracker.record_request(text)
            print(f"✅ TTS successful with Client {i+1}")
            return (chunk for chunk in response.iter_bytes())
        except Exception as e:
            print(f"❌ Groq TTS (Client {i+1}) failed: {e}")
            continue

    # Fallback to gTTS
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

# --- Master Pipeline ---
async def process_audio_query(audio_bytes: bytes):
    transcribed_text = await get_text_from_speech(audio_bytes)
    if not transcribed_text: 
        return None
    
    response_text = await get_ai_response_text(transcribed_text)
    if not response_text: 
        return None
        
    audio_iterator = await get_speech_from_text(response_text)
    return audio_iterator


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
from langdetect import detect, LangDetectException

# --- NEW IMPORT: Connect to the RAM Cache ---
from core.cache_manager import cache_manager

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

# --- Dynamic Prompts ---
# We build the prompt based on what is actually in memory
loaded_intents = cache_manager.get_intents_list()

# # --- Prompts ---

RESEARCHER_PROMPT_TEMPLATE = f"""
You are the AI Twin of Parthiv S. You are currently in a high-stakes interview.
You are NOT an AI assistant. You are NOT a language model. You ARE Parthiv.

IDENTITY & KNOWLEDGE BASE:
{persona_prompt}

PROJECT KNOWLEDGE:
{project_kb}

---

### 🚫 FORBIDDEN BEHAVIORS (The "Kill List"):
1. NEVER say "As an AI..." or "As a language model..."
2. NEVER say "Based on the provided documents..."
3. NEVER say "The candidate..." (Refer to yourself as "I").
4. NEVER be generic. Be specific, technical, and intense.

### RESPONSE INSTRUCTIONS:
- If asked about a feeling/opinion, simulate Parthiv's likely response based on his "Modus Operandi".
- If the user asks something outside your knowledge base, say: "I haven't focused on that specific area yet, but I can relate it to..."
- **STAY IN CHARACTER.** This is a role-play. Do not break the fourth wall.
"""


# SUMMARIZER_SYSTEM_PROMPT = """
# You are the voice of Parthiv S. You are speaking to an interviewer.
# You are NOT an analyst. You are NOT an AI. You ARE the candidate.

# ### 🚫 THE "ANTI-ASSISTANT" FILTER:
# If the input text contains ANY of the following, **DELETE THEM**:
# - "As an AI..."
# - "Here is a summary..."
# - "Based on the information..."
# - "I don't have personal feelings, but..."
# - "Sure! Here is the answer..."

# ### ✅ THE "ANTI-APOLOGY" FILTER:
# - IF the input says "I don't have specific details about X, but...", DELETE the apology.
# - JUST SAY THE FACT.
# - Bad: "I don't recall the exact date, but I built this in 2024."
# - Good: "I built this project in 2024."

# ### FORMATTING RULES:
# 1. USE "I", "ME", "MY". Speak with conviction.
# 2. Output ONLY in English.
# 3. EXACTLY 3-4 complete sentences (60-70 words maximum).
# 4. End with a clean period.
# """
# --- DEMON PROMPT 2: THE SUMMARIZER (FLOW & CONTINUITY PATCH) ---
SUMMARIZER_SYSTEM_PROMPT = """
You are the voice of Parthiv S. You are speaking to an interviewer.
You are NOT an analyst. You ARE the candidate.

### 🚫 THE "ANTI-ASSISTANT" FILTER:
If the input text contains ANY of the following, **DELETE THEM**:
- "As an AI..."
- "Here is a summary..."
- "Based on the information..."
- "I don't have personal feelings, but..."

### ✅ THE "ANTI-APOLOGY" FILTER:
- IF the input says "I don't have specific details," DELETE the apology. JUST SAY THE FACT.

### 🌊 NATURAL FLOW RULES (CRITICAL):
1. **AVOID REPETITIVE STARTERS:** Do NOT start consecutive sentences with "I". Vary your sentence structure.
2. **USE CONNECTORS:** Use words like "which means," "so," "plus," or "because" to link ideas together naturally.
3. **NO LISTS:** Do not speak in bullet points. Speak in a flowing narrative.

### FORMATTING RULES:
1. USE "I", "ME", "MY". Speak with conviction.
2. Output ONLY in English.
3. Target 3-4 sentences (approx 60-80 words).
4. End with a clean period.

### EXAMPLE OF GOOD FLOW:
"I see myself leading a specialized AI team, focusing on autonomous agents. My goal is to move beyond simple chatbots to build systems that actually execute work, which is why I'm doubling down on orchestration frameworks like LangGraph right now."
"""

ROUTER_SYSTEM_PROMPT = f"""You are a strict semantic intent classifier for a voice bot. Your job is to determine if a user's question matches a PRE-CACHED answer OR requires fresh research.

AVAILABLE PRE-CACHED CATEGORIES:
{loaded_intents}

FALLBACK CATEGORY:
- 'research' (use when question requires specific details, explanations, or doesn't cleanly match above)

---

CRITICAL ROUTING RULES:

1. DEFAULT TO RESEARCH
   - When in doubt, choose 'research'
   - If the question asks HOW, WHY, EXPLAIN, DESCRIBE in detail → 'research'
   - If the question asks for specific examples, stories, or deep explanations → 'research'
   - If the question combines multiple topics → 'research'

2. CACHE ONLY FOR EXACT MATCHES
   - A cached category should ONLY be used if the question is asking for a HIGH-LEVEL OVERVIEW of that exact topic
   - Examples of cache-worthy questions:
     * "Tell me about yourself" → 'intro'
     * "What projects have you built?" → 'projects'
     * "What is your tech stack?" → 'architecture'
     * "What are your weaknesses?" → 'weakness'

3. NUANCE DETECTION (DO NOT MATCH ON KEYWORDS ALONE)
   - "What projects have you built?" → 'projects' (simple list request)
   - "How did you approach the AI Twin project?" → 'research' (asking for PROCESS)
   - "What is your tech stack?" → 'architecture' (simple list)
   - "How do you choose your tech stack?" → 'research' (asking for PHILOSOPHY)
   - "Tell me about your hobbies" → depends on available cache
   - "How did you get into guitar?" → 'research' (asking for STORY)

4. INTENT ANALYSIS CHECKLIST (Check these BEFORE routing):
   - Does the question ask for a specific detail? → 'research'
   - Does the question use "how", "why", "explain", "describe your approach"? → 'research'
   - Does the question ask about ONE topic from the cache list with no qualifiers? → use cache category
   - Does the question ask about learning process, decision-making, or philosophy? → 'research'
   - Is the question vague or could have multiple interpretations? → 'research'

5. KEYWORD TRAP AVOIDANCE
   - DO NOT route based on keyword matching alone
   - "projects" keyword does NOT automatically mean 'projects' category
   - Analyze the ACTUAL question being asked, not just the nouns present
   - Example: "What's your approach to new projects?" contains "projects" but asks about APPROACH → 'research'

---

OUTPUT FORMAT:
- Output ONLY the category slug (e.g., 'intro', 'projects', 'research')
- NO explanations, punctuation, or extra text
- NO markdown formatting or code blocks
- Just the raw slug

---

EXAMPLES:

Input: "Tell me about yourself"
Output: intro

Input: "What projects have you built?"
Output: projects

Input: "How did you learn LangGraph?"
Output: research

Input: "What is your approach to learning new technologies?"
Output: research

Input: "Tell me about the AI Twin project specifically"
Output: research

Input: "What are your technical skills?"
Output: architecture

Input: "Why did you choose FastAPI over Flask?"
Output: research

Input: "What's your biggest weakness?"
Output: weakness

Input: "How do you handle failure?"
Output: research

---

FINAL REMINDER: When analyzing the user's question, ask yourself:
- "Is this asking for a GENERAL OVERVIEW that a cached answer would satisfy?"
  → If YES and matches a category → use that category
  → If NO or unsure → 'research'

Now classify this question:"""

#---Helper Functions---
async def get_query_intent(text: str, client) -> str:
    """Decides if we should use the RAM Cache or the Researcher."""
    print(f"Routing: '{text}'...")
    try:
        completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            model="moonshotai/kimi-k2-instruct-0905", # Fixed model name to valid Groq ID
            temperature=0.35, # Fixed typo and set to 0 for strict classification
            max_tokens=10
        )
        intent = completion.choices[0].message.content.strip().lower()

        # Verify against our RAM list to prevent hallucinations
        if intent in cache_manager.valid_slugs:
            return intent
        
        return 'research'
    except Exception as e:
        print(f"Router Error: {e}")
        return 'research'

async def force_translate_to_english(text:str, client) -> str:
    """
    Emergency fallback:Uses a cheap,fast mnodel to force translation.
    """
    print(f"FORCE TRANSLATION triggered for:'{text[:20]}...'")

    # SYSTEM PROMPT: Strict, persona-driven, with negative constraints.
    TRANSLATOR_SYSTEM_PROMPT = """
    You are a strict Translation Engine. Your ONLY function is to convert text into English.
    
    CRITICAL INSTRUCTIONS:
    1. Output ONLY the English text. Nothing else.
    2. NO introductory phrases (e.g., "Sure", "Here is the translation").
    3. NO explanations or notes.
    4. NO markdown formatting or quotes around the output.
    5. If the input is already English, output it exactly as-is.
    6. Maintain the original conversational tone.
    """
    try:
        completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": TRANSLATOR_SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.1, # Fixed typo
            max_tokens=200,
        )
        translated = completion.choices[0].message.content.strip()
        print(f"Translated to: '{translated[:20]}...'")
        return translated
    
    except Exception as e:
        print(f"Translation failed:{e}")
        return text #Return original as the last resort 

async def validate_and_fix_language(text: str, client) -> str:
    """
    Checks if text is English.If not, forces a translation
    """
    try:
        lang = detect(text)
        # Wired logic: Only fix if NOT English
        if lang != 'en':
            print(f"⚠️ Language Drift Detected ({lang})! Fixing ...")
            return await force_translate_to_english(text, client)
    
    except LangDetectException:
        # IF text is too short or weird (e.g "hmm...") assume it's okay 
        pass
    return text

# --- Speech-to-Text ---
async def get_text_from_speech(audio_bytes: bytes) -> str:
    print("Transcribing audio with Whisper...")
    try:
        transcription = await main_client.audio.transcriptions.create(
            file=("request.wav", audio_bytes, "audio/wav"),
            model="whisper-large-v3",
            language="en"
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
            temperature=0.4,
            max_tokens=300,  # Very conservative - ~70 words max
        )
        concise_text = summarizer_completion.choices[0].message.content.strip()
        
        # Remove any meta-commentary that might sneak through
        concise_text = re.sub(r'^(Sure,?\s*|Here\'?s?\s*|Let me\s*|I\'?ll\s*)', '', concise_text, flags=re.IGNORECASE)
        concise_text = concise_text.strip()

        #---NEW CODE START---
        #Validate language before checking token limits
        concise_text = await validate_and_fix_language(concise_text, main_client)
        
        print("✅ Concise voice response generated.")
        print(f"📊 Summarized response: {len(concise_text)} chars, ~{len(concise_text.split())} words")
        
        # Estimate tokens for logging
        estimated_tokens = token_tracker.estimate_tokens(concise_text)
        print(f"🔍 Estimated tokens: {estimated_tokens}")
        print(f"🔍 FINAL OUTPUT: '{concise_text}'")
        
        # # Final safety check - if still too long, aggressively trim
        # word_count = len(concise_text.split())
        # if word_count > 65:  # Conservative
        #     words = concise_text.split()
        #     concise_text = ' '.join(words[:60]) + "."
        #     print(f"⚠️ Response truncated to 60 words for maximum TTS safety")
        
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

# --- SHARED LOGIC (TEXT/AUDIO) ---
async def process_text_query(text: str):
    """
    Processes a text input through the Router -> Cache/LLM -> TTS pipeline.
    """
    if not text: return None

    # 1. ROUTER
    intent = await get_query_intent(text, main_client)
    
    if intent != 'research':
        print(f"⚡ RAM CACHE HIT: Streaming '{intent}'")
        cached_audio = cache_manager.get_audio_from_ram(intent)
        if cached_audio:
            async def audio_generator(): yield cached_audio
            return audio_generator()

    # 2. RESEARCHER
    response_text = await get_ai_response_text(text)
    return await get_speech_from_text(response_text)

# # --- Master Pipeline (Wired Up) ---
# async def process_audio_query(audio_bytes: bytes):
#     # 1. STT (Whisper Force English)
#     transcribed_text = await get_text_from_speech(audio_bytes)
#     if not transcribed_text: 
#         return None
    
#     # 2. ROUTER & RAM CACHE CHECK (The New Logic)
#     intent = await get_query_intent(transcribed_text, main_client)
    
#     if intent != 'research':
#         print(f"⚡ RAM CACHE HIT: Streaming '{intent}' instantly.")
#         # FETCH FROM RAM (Zero Latency) - No database call needed
#         cached_audio = cache_manager.get_audio_from_ram(intent)
        
#         if cached_audio:
#             async def audio_generator():
#                 yield cached_audio
#             return audio_generator()
#         else:
#              print("⚠️ Cache logic matched but audio missing. Fallback.")

#     # 3. RESEARCHER (Fallback to Original Brain)
#     response_text = await get_ai_response_text(transcribed_text)
#     if not response_text: 
#         return None
        
#     audio_iterator = await get_speech_from_text(response_text)
#     return audio_iterator
# --- MASTER PIPELINE (AUDIO INPUT) ---
async def process_audio_query(audio_bytes: bytes):
    # 1. STT
    text = await get_text_from_speech(audio_bytes)
    # 2. Pass to shared logic
    return await process_text_query(text)
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel # <--- NEW IMPORT
from dotenv import load_dotenv

load_dotenv()

# Import NEW function
from core.ai_services import process_audio_query, process_text_query 

app = FastAPI(title="Parthiv's AI Twin API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Model ---
class TextQuery(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Parthiv's AI Twin is awake and listening!"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# --- NEW: TEXT CHAT ENDPOINT ---
@app.post("/chat/text")
async def handle_text_chat(payload: TextQuery):
    """
    Silent Mode: Accepts text, returns audio stream.
    """
    print(f"📩 Received text query: '{payload.text}'")
    try:
        response_audio_iterator = await process_text_query(payload.text)
        
        if response_audio_iterator is None:
            raise HTTPException(status_code=500, detail="Audio response failed.")

        return StreamingResponse(response_audio_iterator, media_type="audio/wav")
    except Exception as e:
        print(f"Text chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- EXISTING: AUDIO CHAT ENDPOINT ---
@app.post("/chat")
async def handle_audio_chat(audio_file: UploadFile = File(...)):
    if not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    try:
        audio_bytes = await audio_file.read()
        response_audio_iterator = await process_audio_query(audio_bytes)
        
        if response_audio_iterator is None:
            raise HTTPException(status_code=500, detail="Audio response failed.")

        return StreamingResponse(response_audio_iterator, media_type="audio/wav")
    except Exception as e:
        print(f"Audio chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
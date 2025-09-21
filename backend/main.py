from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# Load environment variables first, so the core module can use them
load_dotenv()

# Import our single, powerful audio pipeline function from the core services
from core.ai_services import process_audio_query

app = FastAPI(title="Parthiv's AI Twin API")

# Standard CORS middleware to allow the frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, you might restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def handle_audio_chat(audio_file: UploadFile = File(...)):
    """
    Main audio chat endpoint. Receives a user's audio recording,
    processes it through the full STT -> RAG/LLM -> TTS pipeline,
    and streams the final audio response back.
    """
    # Basic validation to ensure we're receiving an audio file
    if not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")
    
    print(f"Received audio file: {audio_file.filename} ({audio_file.content_type})")
    
    try:
        # Read the entire audio file into memory as bytes
        audio_bytes = await audio_file.read()
        
        # This one line calls our entire AI brain: STT -> RAG/LLM -> TTS
        response_audio_iterator = await process_audio_query(audio_bytes)
        
        if response_audio_iterator is None:
            # This happens if the TTS service fails in the core logic
            raise HTTPException(status_code=500, detail="Audio response generation failed.")

        # Stream the audio response back to the client efficiently
        return StreamingResponse(response_audio_iterator, media_type="audio/wav")

    except RuntimeError as e:
        # Catches the "AI services not initialized" error from ai_services.py
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Catches any other unexpected errors in the pipeline
        print(f"An unexpected error occurred in the chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

@app.get("/health")
async def health_check():
    """
    A simple health check endpoint to confirm the server is running.
    Useful for deployment monitoring.
    """
    return {"status": "ok"}


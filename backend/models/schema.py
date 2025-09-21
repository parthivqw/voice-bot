from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_text: str

class ChatResponse(BaseModel):
    response_text: str
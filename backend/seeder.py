import os
import json
import base64
import asyncio
from openai import AsyncOpenAI
from supabase import create_client,Client
from dotenv import load_dotenv

load_dotenv()

#Config---
SUPABASE_URL=os.environ.get("SUPABASE_URL")
#IMPORTANT: USE SERVICE_ROLE_KEY for writing data!
SUPABASE_KEY=os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GROQ_API_KEY=os.environ.get("GROQ_API_KEY_7")

if not SUPABASE_KEY or not GROQ_API_KEY:
    raise ValueError("Missing Keys! Make sure SUPABASE_SRVICE_ROLE_KEY and GROQ_API_KEY_6 are in .env")

#Init Clients
supabase: Client=create_client(SUPABASE_URL,SUPABASE_KEY)
groq_client =AsyncOpenAI(api_key=GROQ_API_KEY,base_url="https://api.groq.com/openai/v1")

async def generate_audio_base64(text: str) -> str:
    print(f"Generating audio for: '{text[:30]}'...")
    try:
        response=await groq_client.audio.speech.create(
            model="playai-tts",
            voice="Mason-PlayAI",
            input=text,
            
        )
        # Read binary content
        audio_content=response.content
        #Convert to Base64 string
        base64_string=base64.b64encode(audio_content).decode('utf-8')
        return base64_string
    
    except Exception as e:
        print(f"Audio generation failed:{e}")
        return None
async def seed_database():
    print("Starting Database Seeder...")

    with open('seeds.json', 'r', encoding='utf-8') as f:
        seeds=json.load(f)

    for item in seeds:
        slug=item['slug']
        print(f"\nProcessing '{slug}'...")

        #1.Check if exists
        existing=supabase.table('canonical_qa').select('id').eq('slug',slug).execute()
        if existing.data:
            print(f"Slug '{slug}' already exists.Skipping.")
            continue
        #2.Generate Audio
        audio_b64=await generate_audio_base64(item['text_answer'])
        if not audio_b64:
            print("Skpping due to audio error.")
            continue
        
        #3.Insert into Supabase
        data={
            "slug":slug,
            "triggers":item['triggers'],
            "text_answer":item['text_answer'],
            "audio_base64":audio_b64,
            "description":item['description']
        }

        try:
            supabase.table('canonical_qa').insert(data).execute()
            print(f"Successfully inserted '{slug}' into DB!")

        except Exception as e:
            print(f"DB Inserted failed:{e}")
    
    print("\n Seeding Complete!")


if __name__=="__main__":
    asyncio.run(seed_database())
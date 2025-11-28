import os
import base64
from supabase import create_client, Client
from dotenv import load_dotenv

# Load env vars
load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY") # Public key is fine for reading

if not url or not key:
    raise ValueError("Missing Supabase credentials in .env")

supabase: Client = create_client(url, key)

def save_audio_from_db(slug: str, filename: str):
    print(f"🎧 Fetching '{slug}' from Supabase...")
    
    try:
        # 1. Fetch the row
        response = supabase.table('canonical_qa').select('audio_base64').eq('slug', slug).single().execute()
        
        if not response.data:
            print(f"❌ Error: Slug '{slug}' not found!")
            return

        # 2. Decode Base64 -> Bytes
        b64_string = response.data['audio_base64']
        audio_bytes = base64.b64decode(b64_string)
        
        # 3. Save to file
        with open(filename, "wb") as f:
            f.write(audio_bytes)
            
        print(f"✅ Saved audio to: {filename}")
        print("👉 Go play this file now!")
        
    except Exception as e:
        print(f"❌ Error fetching audio: {e}")

if __name__ == "__main__":
    # Test the 'intro' and 'superpower' slugs
    save_audio_from_db("intro", "test_intro.wav")
    save_audio_from_db("superpower", "test_superpower.wav")
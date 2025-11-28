# import os
# import base64
# from supabase import create_client, Client
# from dotenv import load_dotenv

# load_dotenv()

# class CacheManager:
#     def __init__(self):
#         url=os.environ.get("SUPABASE_URL")
#         key=os.environ.get("SUPABASE_KEY")

#         #In-memory stores
#         self.audio_cache={} #{'intro':/...}
#         self.trigger_map={} #{'intro':['WHo are you',...]}
#         self.valid_slugs=[] #['intro','superpower',...]

#         if not url or not key:
#             print("Supabase credentials missing. Cache disabled.")
#             self.client=None
#         else:
#             self.client: Client=create_client(url,key)
#             print("CacheManager connected. Pre-loading memory...")
#             self.preload_content()
        
    
#     def preload_content(self):
#         """
#         Downloads ALL cached answers (audio+text) into RAM at startup.
#         """
#         try:
#             # Fetch everything, The files are small enough for RAM
#             response=self.client.table('canonical_qa').select('*').execute()

#             if not response.data:
#                 print("Database is empty. No cache loaded")
#                 return
            
#             count=0
#             for row in response.data:
#                 slug=row['slug']
#                 b64_str=row['audio_base64']
#                 triggers=row['triggers']

#                 #decode and store in RAM
#                 self.audio_cache[slug]=base64.b16decode(b64_str)
#                 self.trigger_map[slug]=triggers
#                 self.valid_slugs.append(slug)
#                 count+=1

#                 print("Cache Hydrated: {count} answers loaded into RAM.Zero latency mode ON.")
#                 print(f"Available intents:{self.valid_slugs}")
            
#         except Exception as e:
#             print(f"Cache Preload failed:{e}")
    

#     def get_audio_from_ram(self,slug:str) -> bytes:
#         """
#         Instant lookup.No DB call
#         """
#         return self.audio_cache.get(slug)
    
#     def get_intents_list(self) -> str:
#         """
#         Returns a formatted string of intents for the LLM Prompt.
#         e.g. "1. intro\n2. superpower..."
#         """

#         if not self.valid_slugs: return ""
#         return "\n".join([f"- '{slug}'" for slug in self.valid_slugs])
        


# # Singleton instance
# cache_manager=CacheManager()    

import os
import base64
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class CacheManager:
    def __init__(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        
        # In-memory stores
        self.audio_cache = {}    # { 'intro': b'\x00...' }
        self.trigger_map = {}    # { 'intro': ['who are you', ...] }
        self.valid_slugs = []    # ['intro', 'superpower', ...]

        if not url or not key:
            print("⚠️ Supabase credentials missing. Cache disabled.")
            self.client = None
        else:
            try:
                self.client: Client = create_client(url, key)
                print("✅ CacheManager connected. Pre-loading memory...")
                self.preload_content()
            except Exception as e:
                print(f"❌ Connection Error: {e}")

    def preload_content(self):
        """
        Downloads ALL cached answers into RAM at startup.
        Includes error handling per-row so one bad apple doesn't kill the bot.
        """
        try:
            response = self.client.table('canonical_qa').select('*').execute()
            
            if not response.data:
                print("⚠️ Database is empty. No cache loaded.")
                return

            count = 0
            for row in response.data:
                slug = row.get('slug', 'unknown')
                try:
                    b64_str = row.get('audio_base64', '')
                    triggers = row.get('triggers', [])

                    # Skip empty data
                    if not b64_str:
                        print(f"⚠️ Skipping '{slug}': No audio data found.")
                        continue

                    # Decode and store
                    # We use standard b64decode. If it fails, we catch it below.
                    decoded_audio = base64.b64decode(b64_str)
                    
                    self.audio_cache[slug] = decoded_audio
                    self.trigger_map[slug] = triggers
                    self.valid_slugs.append(slug)
                    count += 1
                except Exception as row_error:
                    print(f"❌ CORRUPT DATA in '{slug}': {row_error}")
                    continue
            
            print(f"🚀 Cache Hydrated: {count} answers loaded. Zero latency mode ON.")
            print(f"📋 Available intents: {self.valid_slugs}")

        except Exception as e:
            print(f"❌ Global Cache Failure: {e}")

    def get_audio_from_ram(self, slug: str) -> bytes:
        return self.audio_cache.get(slug)

    def get_intents_list(self) -> str:
        if not self.valid_slugs: return ""
        return "\n".join([f"- '{slug}'" for slug in self.valid_slugs])

# Singleton instance
cache_manager = CacheManager()
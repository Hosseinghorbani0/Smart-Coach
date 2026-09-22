from openai import OpenAI
from .config import Config
from .voice_processor import VoiceProcessor
from .chat_history import ChatHistory

class CoachManager:
    def __init__(self):
        self.api_key = Config.get_openai_api_key()
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

        self.voice_processor = VoiceProcessor()
        self.chat_history = ChatHistory()
        self.coach_persona = Config.COACH_PERSONA

        Config.init_storage()
    
    def process_voice(self, voice_path):
       
        try:
           
            info = self.voice_processor.get_audio_info(voice_path)
            if not info:
                return None
                
            wav_path = self.voice_processor.convert_audio(voice_path)
            if not wav_path:
                return None
                
            text = self.voice_processor.convert_to_text(wav_path)
            return text
            
        except Exception as e:
            print(f" خطا در پردازش صدا: {str(e)}")
            return None
    
    def get_response(self, text, callback):
        if not self.client:
            callback("برای استفاده از مربی هوشمند، یک کلید OpenAI در فایل .env با نام SMART_COACH_OPENAI_API_KEY تنظیم کنید.")
            return

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.coach_persona},
                    {"role": "user", "content": text}
                ],
                temperature=1.1,
                max_tokens=400
            )

            coach_response = response.choices[0].message.content
            callback(coach_response)

        except Exception as e:
            print(f" خطا در دریافت پاسخ: {str(e)}")
            callback(None)

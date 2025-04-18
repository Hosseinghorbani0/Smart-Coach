import speech_recognition as sr


class VoiceProcessor:
    def __init__(self):
      
        self.recognizer = sr.Recognizer()
     
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        
    def record_audio(self, duration=5):
  
        try:
            with sr.Microphone() as source:
                print(" در حال گوش دادن...")
            
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
         
                audio = self.recognizer.listen(source, timeout=duration)
                return audio
                
        except sr.WaitTimeoutError:
            print(" زمان ضبط به پایان رسید")
            return None
        except Exception as e:
            print(f" خطا در ضبط صدا: {str(e)}")
            return None

    def convert_to_text(self, audio):
     
        try:
            if isinstance(audio, str):  
                with sr.AudioFile(audio) as source:
                    audio = self.recognizer.record(source)
            
           
            text = self.recognizer.recognize_google(
                audio,
                language='fa-IR',
                show_all=False
            )
            print(f"✓ متن تشخیص داده شده: {text}")
            return text
                
        except sr.UnknownValueError:
            print(" صدا قابل تشخیص نبود")
            return None
        except sr.RequestError as e:
            print(f" خطا در ارتباط با سرویس گوگل: {str(e)}")
            return None
        except Exception as e:
            print(f" خطا در تبدیل صدا به متن: {str(e)}")
            return None

    def process_voice_command(self):
        
        try:

            audio = self.record_audio()
            if not audio:
                return None
                
            text = self.convert_to_text(audio)
            if not text:
                return None
                
            return text
            
        except Exception as e:
            print(f" خطا در پردازش دستور صوتی: {str(e)}")
            return None

    def calculate_range_score(self, value, min_val, max_val):
       
        if value < min_val:
            return 0.0
        elif value > max_val:
            return 0.0
        else:
            
            mid = (min_val + max_val) / 2
            range_size = (max_val - min_val) / 2
            distance = abs(value - mid)
            score = 1.0 - (distance / range_size)
            return max(0.0, min(1.0, score))

    def calculate_all_angles(self, landmarks):
        
        try:
            angles = {}
            
            angles['knee_left'] = self.calculate_angle(
                [landmarks[23].x, landmarks[23].y],  
                [landmarks[25].x, landmarks[25].y],  
                [landmarks[27].x, landmarks[27].y]  
            )
            
            angles['knee_right'] = self.calculate_angle(
                [landmarks[24].x, landmarks[24].y], 
                [landmarks[26].x, landmarks[26].y],  
                [landmarks[28].x, landmarks[28].y]   
            )
            
            angles['knee'] = (angles['knee_left'] + angles['knee_right']) / 2
            
            angles['elbow_left'] = self.calculate_angle(
                [landmarks[11].x, landmarks[11].y], 
                [landmarks[13].x, landmarks[13].y],  
                [landmarks[15].x, landmarks[15].y]   
            )
            
            angles['elbow_right'] = self.calculate_angle(
                [landmarks[12].x, landmarks[12].y],  
                [landmarks[14].x, landmarks[14].y],  
                [landmarks[16].x, landmarks[16].y] 
            )
            
            angles['elbow'] = (angles['elbow_left'] + angles['elbow_right']) / 2
            
            angles['back'] = self.calculate_angle(
                [landmarks[11].x, landmarks[11].y], 
                [landmarks[23].x, landmarks[23].y],  
                [landmarks[25].x, landmarks[25].y]   
            )
            
            angles['hip'] = self.calculate_angle(
                [landmarks[11].x, landmarks[11].y],  
                [landmarks[23].x, landmarks[23].y],  
                [landmarks[25].x, landmarks[25].y]   
            )
            
            print(f"✓ زوایا محاسبه شدند:")
            print(f"  زانو: {angles['knee']:.1f}")
            print(f"  آرنج: {angles['elbow']:.1f}")
            print(f"  کمر: {angles['back']:.1f}")
            print(f"  لگن: {angles['hip']:.1f}")
            
            return angles
            
        except Exception as e:
            print(f" خطا در محاسبه زوایا: {str(e)}")
            return None

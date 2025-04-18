import os
from pygame import mixer
import random

class ExerciseAudioManager:
    def __init__(self):
        
        self.base_path ='./audio_files'
        
        mixer.init()
        
        self.exercises = ['squat', 'pushup', 'pullup', 'deadlift']
        
        self.max_count = 15
        
        self._verify_audio_files()
    
    def _verify_audio_files(self):
        
        for exercise in self.exercises:
            exercise_path = os.path.join(self.base_path, exercise)
            if not os.path.exists(exercise_path):
                print(f" پوشه {exercise} وجود ندارد")
    
    def play_count(self, exercise_type, count):
    
        if exercise_type not in self.exercises or not (1 <= count <= self.max_count):
            return
        
        try:
            audio_file = os.path.join(self.base_path, exercise_type, f"{count}.mp3")
            if os.path.exists(audio_file):
             
                mixer.music.stop()
                
                mixer.music.load(audio_file)
                mixer.music.play()
            else:
                print(f"  فایل صدا پیدا نشد : {audio_file}")
            
        except Exception as e:
            print(f"خطا ی پخش  شمارنده: {e}")

    def play_random_wrong_form(self):
        
        feedback_dir = os.path.join(self.base_path, "feedback")
        feedback_files = [f for f in os.listdir(feedback_dir) if f.endswith('.mp3')]
        
        if feedback_files:
            chosen_file = random.choice(feedback_files)
            audio_file = os.path.join(feedback_dir, chosen_file)
            
            try:
                mixer.music.stop()  
                mixer.music.load(audio_file)
                mixer.music.play()
            except Exception as e:
                print(f"  خطا ی پخش فید بک  : {e}")

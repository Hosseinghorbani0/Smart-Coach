import pyaudio
import wave
import os
from datetime import datetime

class VoiceRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        
        self.save_dir = "voices"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def start_recording(self):
        
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        self.frames = []
        self.is_recording = True
        
    def stop_recording(self):
      
        if not self.is_recording:
            return None
            
        self.is_recording = False
        self.stream.stop_stream()
        self.stream.close()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"voice_{timestamp}.wav"
        filepath = os.path.join(self.save_dir, filename)
        
        wf = wave.open(filepath, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        return filepath
        
    def __del__(self):
       
        self.audio.terminate()

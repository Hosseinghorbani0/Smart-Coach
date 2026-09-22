# Smart Coach

Smart Coach is a Python desktop app for exercise monitoring, AI coaching, and real-time movement feedback.
It combines computer vision, pose analysis, and an AI coach assistant to help users train with better form.

## Highlights
- Real-time exercise detection using MediaPipe pose estimation
- AI-driven coaching assistant with OpenAI integration
- Desktop UI built with PyQt5
- Voice transcription support for Persian audio inputs
- Storage and configuration cleanly separated from app logic

## Quick start

```bash
git clone https://github.com/Hosseinghorbani0/Smart-Coach.git
cd Smart-Coach
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
python main.py
```

## Environment variables
Create a `.env` file based on [.env.example](.env.example) and set your key:

```env
SMART_COACH_OPENAI_API_KEY=your_key_here
```

The app will also create the needed storage directories automatically when it starts.

## Project structure

```text
Smart-Coach/
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── README_FA.md
├── core/
│   ├── coach_manager.py
│   ├── config.py
│   ├── voice_processor.py
│   └── chat_history.py
├── processes/
│   ├── audio_manager.py
│   ├── exercise_detector.py
│   └── model.py
├── ui/
│   ├── coach_window.py
│   ├── home_window.py
│   ├── training_window.py
│   └── voice_recorder.py
├── Model_training/
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── audio_files/
├── tests/
└── storage/
```

## Development

```bash
pytest -q
```

## Notes
- This project is designed for Windows-first usage, but it can be adapted to Linux/macOS.
- If you do not provide an OpenAI key, the coach UI will still open and explain how to enable AI assistance.
- Media and model weights are intentionally kept out of version control by default.
AUDIO_FORMAT=wav

# Processing Settings
GPU_ENABLED=true
DETECTION_CONFIDENCE=0.5
TRACKING_CONFIDENCE=0.5
```

### Model Configuration
```python
# config/model_config.py
POSE_DETECTION = {
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'model_complexity': 2
}

MOVEMENT_THRESHOLDS = {
    'squat': {'start': 160, 'end': 110},
    'pushup': {'start': 160, 'end': 90},
    'pullup': {'start': 160, 'end': 60},
    'deadlift': {'start': 160, 'end': 90}
}
```

## Usage

### Basic Operation
```python
from core.detector import ExerciseDetector
from core.analyzer import FormAnalyzer

# Initialize system
detector = ExerciseDetector()
analyzer = FormAnalyzer()

# Start analysis
detector.start_camera()
while True:
    frame = detector.get_frame()
    poses = detector.detect_pose(frame)
    feedback = analyzer.analyze_form(poses)
    detector.display_feedback(feedback)
```

### Advanced Features
```python
# Custom exercise configuration
detector.set_exercise_type('squat')
detector.set_difficulty('advanced')
detector.enable_audio_feedback(True)
```

## Development

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document all functions
- Maintain test coverage >80%

### Building from Source
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run build script
python setup.py build
```

## Testing

### Unit Tests
```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/test_detector.py
```

### Performance Testing
```bash
# Run benchmarks
python benchmarks/run_all.py
```

## API Documentation

### Core Classes
```python
class ExerciseDetector:
    """
    Main detection class for exercise analysis.
    
    Attributes:
        confidence_threshold (float): Detection confidence level
        frame_buffer (int): Number of frames to buffer
        
    Methods:
        detect_pose(): Returns pose landmarks
        analyze_movement(): Analyzes exercise form
        generate_feedback(): Creates feedback message
    """
```

## Contributing
1. Fork repository
2. Create feature branch
3. Implement changes
4. Add tests
5. Submit pull request

## Contact

### Developer
**Hossein Ghorbani**
- Email: hosseingh1068@gmail.com
- Website: [hosseinghorbani0.ir](http://hosseinghorbani0.ir)


### Project Links
- Issue Tracker: GitHub Issues
- Source Code: GitHub Repository

# Smart Coach

Smart Coach is a Windows-first Python desktop application for real-time exercise analysis and AI-assisted coaching. It combines MediaPipe pose estimation, OpenCV video processing, PyTorch models, PyQt5 screens, audio feedback, and an optional OpenAI coach.

> Smart Coach is an educational training assistant. It does not replace a qualified coach or medical advice.

<p align="center">
  <a href="https://github.com/Hosseinghorbani0/Smart-Coach/actions"><img src="https://img.shields.io/github/actions/workflow/status/Hosseinghorbani0/Smart-Coach/python-package.yml?branch=master&label=CI" alt="CI status"></a>
  <a href="https://github.com/Hosseinghorbani0/Smart-Coach"><img src="https://img.shields.io/github/last-commit/Hosseinghorbani0/Smart-Coach" alt="Last commit"></a>
  <a href="https://github.com/Hosseinghorbani0/Smart-Coach/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT license"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9 or newer"></a>
</p>

**Documentation:** [English](README.md) | [فارسی](README_FA.md)

## What it does

- Detects body landmarks from a camera stream or a local video.
- Tracks repetitions for squat, push-up, pull-up, and deadlift workflows.
- Provides visual and audio feedback while a movement is being analyzed.
- Offers a Persian AI coaching chat and optional Persian voice transcription.
- Keeps runtime data, credentials, media, and model weights out of Git by default.

## How it works

```text
Camera / Video
      |
      v
OpenCV frame capture -> MediaPipe Pose landmarks
      |                         |
      v                         v
ExerciseDetector          angle and form analysis
      |                         |
      +----------+--------------+
                 v
        PyQt5 training interface
                 |
       audio feedback / repetition count

Chat or voice input -> CoachManager -> OpenAI (optional)
```

## Supported movements

| Movement | Current analysis focus |
| --- | --- |
| Squat | Knee angle, depth, standing transition, form validation |
| Push-up | Elbow angle and body position during the repetition |
| Pull-up | Elbow movement and chin-to-shoulder transition |
| Deadlift | Knee, hip, and back-angle conditions |

## Analysis pipeline

The current detector combines two complementary layers:

1. MediaPipe landmarks are filtered by visibility confidence before angles are calculated.
2. Knee and elbow angles use both body sides when both are visible, reducing one-sided camera noise.
3. A lightweight temporal smoother reduces unstable frame-to-frame measurements.
4. The PyTorch temporal model samples a fixed 32-frame window and uses adaptive 3D pooling before its LSTM, so different video resolutions produce a consistent feature shape.
5. Audio feedback is optional; video analysis continues when `pygame` or audio assets are unavailable.

## Requirements

- Windows 10/11 is the primary supported platform.
- Python 3.9 or newer. Python 3.11 is recommended for the current dependency set.
- A webcam for live sessions, or an `.mp4`, `.avi`, or `.mkv` video for offline analysis.
- A microphone only if voice features are needed.
- At least 8 GB RAM is recommended. A GPU is optional; CPU inference is supported.
- An OpenAI API key is optional and only required for AI chat and Whisper transcription.

## Installation

### Windows PowerShell

```powershell
git clone https://github.com/Hosseinghorbani0/Smart-Coach.git
Set-Location Smart-Coach

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python main.py
```

If PowerShell blocks activation, run the following once in an appropriate PowerShell session, then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Linux/macOS

```bash
git clone https://github.com/Hosseinghorbani0/Smart-Coach.git
cd Smart-Coach
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
python main.py
```

## Configuration

`.env.example` is safe to commit. Copy it to `.env` and replace the placeholder value:

```dotenv
SMART_COACH_OPENAI_API_KEY=your_key_here
```

The application also accepts the standard `OPENAI_API_KEY` variable. Never commit a real API key. When no key is available, the desktop application still starts and the coach explains how to enable the optional AI features.

`SMART_COACH_BASE_DIR` can be used to move runtime storage to another location:

```dotenv
SMART_COACH_BASE_DIR=C:\Users\YourName\SmartCoachData
```

On startup, Smart Coach creates `storage/chat` and `storage/voice` automatically.

## Using the application

1. Run `python main.py`.
2. Select **تمرین کردن** from the home screen.
3. Enable automatic detection or select a movement manually.
4. Start a webcam session or choose a local video.
5. Open the coach screen to send Persian text or use voice features.

The training screen shows the processed video, detected movement, and repetition counts. For the most reliable results, keep the full body visible, use good lighting, and place the camera far enough away to capture all relevant joints.

## Project layout

```text
Smart-Coach/
├── main.py                         # Application entry point
├── requirements.txt                # Runtime dependencies
├── .env.example                    # Safe configuration template
├── core/
│   ├── config.py                   # Environment and runtime paths
│   ├── coach_manager.py            # OpenAI coach integration
│   ├── chat_history.py             # Chat persistence
│   └── voice_processor.py          # Speech recognition helpers
├── processes/
│   ├── exercise_detector.py        # Pose and repetition analysis
│   ├── audio_manager.py            # Exercise audio feedback
│   └── model.py                    # PyTorch movement model
├── ui/
│   ├── home_window.py              # Main menu
│   ├── training_window.py          # Camera/video training screen
│   ├── coach_window.py             # AI coach screen
│   └── voice_recorder.py           # Recording UI
├── Model_training/                 # Dataset and training utilities
├── audio_files/                    # Bundled feedback audio
└── tests/                          # Regression tests
```

## Development and tests

Activate the virtual environment, then run:

```bash
python -m pytest -q
```

The CI workflow also performs a Python syntax/undefined-name check and runs the test suite. Heavy model weights, recordings, videos, secrets, and runtime storage are excluded by `.gitignore`.

## Troubleshooting

### PyQt5, MediaPipe, or audio dependency errors

Make sure the virtual environment is active and reinstall the pinned dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyAudio may require an OS-level audio driver. The camera/video analysis can still be used without voice recording if audio setup is unavailable.

### The camera does not open

Close other applications using the webcam, check Windows camera permissions, and try a local video from the training screen.

### The coach does not answer

Check that `.env` exists, that `SMART_COACH_OPENAI_API_KEY` is set, and that the key has access to the configured OpenAI models. The key is never required for exercise detection.

## Contributing

1. Create a feature branch from `master`.
2. Keep changes focused and update tests for behavior changes.
3. Run `python -m pytest -q` before opening a pull request.
4. Do not commit `.env`, API keys, model weights, personal recordings, or generated runtime data.

## License and contact

This project is distributed under the [MIT License](LICENSE).

| Contact | Details |
| --- | --- |
| Developer | Hossein Ghorbani |
| Email | hosseingh1068@gmail.com |
| Website | [hosseinghorbani0.ir](http://hosseinghorbani0.ir) |

Project repository: [github.com/Hosseinghorbani0/Smart-Coach](https://github.com/Hosseinghorbani0/Smart-Coach)

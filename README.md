<p align="center">
  <img src="Face Spoofing Detection System Poster.png" />
</p>

# 🎭 Face Spoof Detection: The Real vs Fake Face Detective! 🔍

<div align="center">

![Face Spoof Detection](https://img.shields.io/badge/Face-Spoof%20Detection-blue)
![OpenVINO](https://img.shields.io/badge/OpenVINO-Compatible-orange)
![Python](https://img.shields.io/badge/Python-3.7%2B-green)
![AI](https://img.shields.io/badge/AI-Computer%20Vision-purple)

**"Because not everything that smiles at your camera is real! 😄➡️🤖"**

</div>

## 🚀 What's This All About?

Ever worried that someone might trick your face recognition system with a photo? Or that your security camera might be fooled by a printed picture? **Worry no more!** This project is your AI-powered lie detector for faces! 

We've built a sophisticated system that can look at a face and say: **"You're real!"** 🎉 or **"Nice try, fake face!"** 🚫 with impressive accuracy!

### 🎯 The Mission
- **Detect** faces in images and videos
- **Analyze** them for signs of spoofing (photos, videos, masks)
- **Classify** as "REAL" (legitimate person) or "FAKE" (spoof attempt)
- **Track** faces across frames for stable predictions

## 🧠 How Does This Magic Work?

### The AI Dream Team 🤖

| Component | Role | Superpower |
|-----------|------|------------|
| **SCRFD** | Face Detective 🕵️ | Finds all the faces in the frame with lightning speed |
| **OpenVINO Model** | Truth Verifier 🔍 | Analyzes each face and detects spoofing patterns |
| **Stability Tracking** | Consistency Checker 📊 | Makes sure we're not fooled by temporary glitches |

### The Technical Wizardry 🔮

```python
# The secret sauce in a nutshell:
1. 🔍 SCRFD finds faces → "I see you!"
2. 🎯 Crop and preprocess → "Let me look closer"
3. 🧠 AI model analyzes → "Hmm, real or fake?"
4. 📈 History tracking → "Let me watch you for a bit"
5. 🎨 Display results → "VERDICT: REAL/FAKE!"
```

## ⚡ Quick Start: Become a Face Spoof Detective!

### Prerequisites 🛠️

```bash
# Make sure you have these installed:
pip install openvino-dev[onnx]
pip install opencv-python
pip install dlib
pip install numpy
pip install onnxruntime
```

### Let's Run It! 🏃‍♂️

```python
# Want to test on a video? Easy!
from your_module import FaceSpoofDetector

detector = FaceSoopfDetector()
detector.process_video(
    video_path="suspect_video.mp4", 
    output_file="investigation_results.mp4"
)

# Or use your webcam for real-time detective work!
detector.process_video(video_path=0)  # 0 = webcam
```

### Command Line Superpowers 💻

```bash
# Analyze a video file
python detect_faces.py --video my_video.mp4 --output results.mp4

# Real-time webcam analysis (feel like in a spy movie!)
python detect_faces.py --webcam

# Test on a single image
python detect_faces.py --image suspect_photo.jpg
```

## 🎨 What You'll See

When you run the system, you'll get this cool output:

```
🎬 Frame Analysis:
├── 🔵 Real Face: Green bounding box + "REAL" label
├── 🔴 Fake Face: Red bounding box + "FAKE" label  
└── ⚪ Analyzing: Gray box (AI is thinking...)

📊 Confidence: Hidden for cleaner display (we're confident in our confidence!)
```

## 🏗️ Project Architecture Deep Dive

### The Brain: `FaceSpoofDetector` Class 🧠

```python
class FaceSpoofDetector:
    def __init__(self):
        self.detector = SCRFD()          # Face finder
        self.compiled_model = OpenVINO() # Spoof analyzer
        self.face_histories = {}         # Memory for tracking
    
    def process_frame(self, frame):
        # The main investigation happens here!
        # 1. Detect faces
        # 2. Analyze each face
        # 3. Track over time
        # 4. Render results
```

### The Investigation Process 🔍

1. **Face Detection Phase** 🎯
   - SCRFD scans the frame
   - Finds all faces with bounding boxes
   - Prepares them for analysis

2. **Spoof Analysis Phase** 🔬
   - Each face gets normalized and preprocessed
   - Fed to the OpenVINO model
   - Model returns: "Real" or "Fake" probability

3. **Stability Check Phase** 📊
   - Tracks face predictions over multiple frames
   - Only shows stable, consistent results
   - Prevents flickering between labels

4. **Results Display Phase** 🎨
   - Green box = Real person ✅
   - Red box = Spoof attempt ❌
   - Clean, professional overlay

## 📁 Project Structure

```
face-spoof-detector/
├── 🧠 models/
│   ├── scrfd_2.5g_bnkps_shape640x640.onnx  # Face detector
│   └── model.xml + model.bin               # Spoof classifier
├── 🔧 src/
│   └── scrfd.py                           # SCRFD implementation
├── 🎥 videos/                             # Your test videos
├── 📊 outputs/                            # Analysis results
└── 🚀 main_script.py                      # The main controller
```

## 🛠️ Advanced Configuration

### Tuning the Detective 🎛️

```python
# Want to adjust sensitivity?
detector = FaceSpoofDetector(
    model_path="./custom_model.xml",    # Your custom model
    device="CPU",                       # Or "GPU" if available
    history_length=5                    # More frames = more stable
)

# Adjust detection thresholds
detector.det_thresh = 0.6              # Face detection sensitivity
detector.nms_thresh = 0.4              # Overlap removal threshold
```

### Model Training (For AI Wizards) 🧙‍♂️

The system uses pre-trained models, but you can train your own!

```python
# Data preparation magic
transform = create_transform(
    input_size=224,
    is_training=True,
    color_jitter=0.4,
    auto_augment='rand-m9-mstd0.5'
)
```

## 🎯 Performance Tips

| Scenario | Tip |
|----------|-----|
| **High Accuracy** | Use `history_length=5` for stable predictions |
| **Real-time** | Set `history_length=1` for fastest response |
| **Low Light** | Adjust preprocessing normalization |
| **Multiple Faces** | SCRFD handles this automatically! |

## 🐛 Troubleshooting Common Issues

```python
# Problem: No faces detected?
# Solution: Adjust detection threshold
detector.det_thresh = 0.3  # More sensitive

# Problem: Too many false positives?
# Solution: Increase confidence threshold
# In predict_face(): change 0.73 to 0.8

# Problem: Slow performance?
# Solution: Reduce input resolution or use GPU
```

## 🎭 Real-World Applications

- **Banking Security** 🏦: Prevent account takeover with photo attacks
- **Phone Unlock** 📱: Make sure it's really you unlocking your device
- **Access Control** 🚪: Secure entry points against spoofing
- **Exam Proctoring** 📝: Ensure test-taker is really present
- **Social Media** 📸: Detect fake profiles using stolen photos

## 🔮 Future Enhancements

- [ ] **3D Mask Detection** - Because flat faces are suspicious!
- [ ] **Liveness Detection** - Blink twice if you're real!
- [ ] **Thermal Analysis** - Real faces have warmth!
- [ ] **Multi-modal Fusion** - Combining multiple detection methods

## 📊 Results & Demo

Want to see it in action? Check out the `outputs/` folder for sample videos showing the system detecting both real and spoofed faces with impressive accuracy!

## 🤝 Contributing

Found a way to make this better? We'd love your help!
- 🐛 Found a bug? Open an issue!
- 💡 Have an idea? Start a discussion!
- 🔧 Made improvements? Submit a PR!

## 📜 License

This project is available for research and educational purposes. For commercial use, please check the respective model licenses.

---

<div align="center">

**Made with ❤️ by AI Enthusiasts | Because Security Shouldn't Be Spoofed!**

*"In a world of deepfakes, be the detective that finds the truth!"*

</div>

---

### 🚀 Ready to Start Detecting?

Clone this repo, follow the setup instructions, and start protecting your systems from face spoofing attacks today! Your digital bouncer awaits! 🎭🔒

**Remember: In the battle against digital impersonation, this is your secret weapon!**

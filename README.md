# 🎙️ Interview Analyzer
**AI-powered Interview Feedback System (CPU-Optimized, Windows-Friendly)**  
Built with **Faster-Whisper**, **Streamlit**, and **Rule-based NLP Insights**

## 🚀 Overview
Interview Analyzer is a lightweight, fully offline, CPU-optimized application that analyzes interview audio or text transcripts and provides:

- Transcription (via Faster-Whisper)
- Sentiment & tone analysis
- Confidence scoring
- Communication quality metrics
- Filler word analysis
- Key point extraction
- Keyword coverage
- Automatic interview summary
- Actionable improvement suggestions
- Downloadable JSON report

Designed to run smoothly **on any Windows system without a GPU**.

## ✨ Key Features
### 🎧 1. Audio & Transcript Input
- Upload audio (`.wav`, `.mp3`, `.flac`, `.m4a`)
- OR paste a transcript directly
- FFmpeg-based conversion (16 kHz mono)

### 🔊 2. Fast CPU-Only Transcription
- Powered by **Faster-Whisper** (CTranslate2 backend)
- Uses **int8 quantization** for high speed on CPU
- Streamlit progress bar for long audio files

### 🧠 3. Deep Analysis
- **Sentiment timeline** (segment-level)
- **Filler words** (counts + bar chart)
- **Speaking pace (WPM)**
- **Keyword coverage** (found/missing)
- **Overall interview score (0–100)**
- **Confidence score (0–100)**
- **Tone label** (Confident / Neutral / Nervous / Uncertain)

### ✍️ 4. AI-Augmented Feedback
- Automatic interview summary
- Round-type contextual analysis (Tech/HR/Managerial)
- Actionable improvement suggestions
- Key point extraction from transcript

### 📊 5. Visual Dashboard
- Filler word bar chart
- Sentiment-over-time line chart
- Keyword coverage
- Summary + insights panels

### 📁 6. Export Options
- Download JSON report
- Download full transcript (txt)

## 🛠️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/interview_analyzer.git
cd interview_analyzer
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate it  
**Windows**
```bash
venv\Scripts\activate
```

### 4. Install requirements
```bash
pip install -r requirements.txt
```

### 5. Install FFmpeg (required)
Download FFmpeg (Windows static build):  
https://www.gyan.dev/ffmpeg/builds/

Add the `bin/` folder to your **System PATH**.

Verify:
```bash
ffmpeg -version
```

## ▶️ Running the App
```bash
streamlit run app_interview.py
```

Then open:
- **Local URL:** http://localhost:8501
- **Network URL:** http://<your-ip>:8501

## 📂 Project Structure
```
interview_analyzer/
│
├── app_interview.py
├── transcribe.py
├── analysis.py
├── requirements.txt
└── tmp_upload/
```

## 🧠 How It Works

### 1. Audio Processing
- Converts uploaded audio → 16kHz mono WAV using FFmpeg
- Uses Faster-Whisper for transcription

### 2. Text Analysis
- Word count, filler detection
- WPM and sentiment
- Tone classification + confidence estimation
- Key points extraction
- Keyword coverage
- Summary + improvement suggestions

### 3. Visualization
- Streamlit + Plotly charts  
- Sentiment timeline  
- Filler-frequency bar chart

## 🔮 Future Enhancements
- Speaker diarization  
- PDF report export  
- Grammar quality scoring  
- Topic modeling  

## 🤝 Contributing
PRs and feature suggestions are welcome.

## 📜 License
MIT License

## ⭐ Support
If you find this project helpful, please ⭐ the repo!

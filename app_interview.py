# app_interview.py
import streamlit as st
import os
import json
from transcribe import Transcriber
from analysis import (
    count_words, words_per_minute, filler_stats, sentiment_summary,
    per_segment_sentiment, keyword_coverage, compute_overall_score,
    generate_summary, improvement_suggestions, confidence_score,
    tone_label, extract_key_points
)
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Interview Analyzer", layout="wide")
st.title("🎙️ Interview Analyzer — Faster Whisper (CPU, small model)")

# -----------------------------
# Sidebar: settings + interview type
# -----------------------------
with st.sidebar:
    st.markdown("## Settings")
    model_size = st.selectbox("Model size (Faster Whisper)", ["small", "base", "tiny"], index=0)
    compute_type = st.selectbox("Compute type (CPU)", ["int8", "float32"], index=0)
    language = st.text_input("Transcription language (ISO)", value="en")
    user_keywords_text = st.text_area("Keywords to check (comma-separated)", value="data,algorithm,project")
    st.markdown("---")
    round_type = st.selectbox("Interview Type", ["General", "Technical", "Managerial", "HR", "Group Discussion"], index=0)

# -----------------------------
# Allow either paste transcript OR upload audio
# -----------------------------
st.markdown("### OR Paste Transcript (optional)")
manual_text = st.text_area("Paste transcript text here if you don't want to upload audio", height=200)

uploaded = st.file_uploader("Upload interview audio (.wav/.mp3/.m4a/.flac)", type=["wav", "mp3", "m4a", "flac"])

# helper: safe transcribe wrapper (existing flow preserved)
def run_transcription_flow(file_path, model_size, compute_type, language, progress_cb=None):
    transcriber = Transcriber(model_size=model_size, device="cpu", compute_type=compute_type)
    result = transcriber.transcribe_file(file_path, language=language, progress_callback=progress_cb)
    return result

# If user provided neither, prompt
if not uploaded and (not manual_text or manual_text.strip() == ""):
    st.info("Upload an audio file or paste a transcript. Recommended model: small (Faster Whisper) for CPU.")
    st.stop()

# If transcript is pasted, skip transcription (non-destructive)
if manual_text and manual_text.strip():
    transcript = manual_text.strip()
    duration = None
    segments = []  # empty; analysis will handle gracefully
    st.success("Using pasted transcript for analysis.")
else:
    # Save uploaded file and transcribe using existing flow
    tmp_path = os.path.join("tmp_upload")
    os.makedirs(tmp_path, exist_ok=True)
    file_path = os.path.join(tmp_path, uploaded.name)
    with open(file_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.info(f"Saved file: {file_path}")

    st.markdown("### Transcription")
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def progress_cb(p):
        try:
            progress_bar.progress(min(1.0, float(p)))
        except Exception:
            pass

    status_text.text("Transcribing audio — this may take a few moments...")
    result = run_transcription_flow(file_path, model_size, compute_type, language, progress_cb=progress_cb)
    status_text.text("Transcription complete.")
    progress_bar.empty()

    transcript = result.get("text", "")
    duration = result.get("duration", None)
    segments = result.get("segments", [])

# -----------------------------
# Show transcript
# -----------------------------
st.subheader("Full Transcript")
st.text_area("Transcript", value=transcript, height=300)

# -----------------------------
# Core Analysis (existing metrics)
# -----------------------------
st.subheader("Automatic Analysis")
total_words, tokens = count_words(transcript)
wpm = words_per_minute(total_words, duration or 0.0)
filler = filler_stats(transcript)
sentiment = sentiment_summary(transcript)
per_seg_sent = per_segment_sentiment(segments)

st.metric("Duration (s)", f"{duration:.1f}" if duration else "Unknown")
st.metric("Total words", total_words)
st.metric("Words per minute", f"{wpm:.1f}")
st.metric("Filler words (total)", filler["total"])
st.metric("Sentiment (compound)", round(sentiment["compound"], 3))

# Filler words chart
if filler["by_word"]:
    df_f = pd.DataFrame(list(filler["by_word"].items()), columns=["filler", "count"])
    fig_f = px.bar(df_f, x="filler", y="count", title="Filler words frequency")
    st.plotly_chart(fig_f, use_container_width=True)

# Sentiment timeline
df_s = pd.DataFrame([
    {"start": s["start"], "end": s["end"], "compound": s["sentiment"]["compound"], "text": s["text"][:80]}
    for s in per_seg_sent
])
if not df_s.empty:
    fig_s = px.line(df_s, x="start", y="compound", title="Sentiment over time (segment-level)")
    st.plotly_chart(fig_s, use_container_width=True)

# Keyword coverage
user_keywords = [k.strip() for k in user_keywords_text.split(",") if k.strip()]
kwcov = keyword_coverage(user_keywords, transcript)
st.write("**Keyword coverage**")
st.write(kwcov)

# Overall score (existing)
overall = compute_overall_score(wpm, filler["total"], sentiment["compound"])
st.metric("Overall Interview Score (0-100)", overall)

# -----------------------------
# NEW: Summary, Suggestions, Confidence, Tone, Key Points
# (all non-destructive additions)
# -----------------------------
st.subheader("📝 Interview Summary")
summary = generate_summary(transcript, wpm, sentiment, filler["total"], round_type)
st.write(summary)

st.subheader("🔧 Improvement Suggestions")
for s in improvement_suggestions(wpm, filler["total"], sentiment):
    st.write("• " + s)

# Confidence and Tone
conf_score = confidence_score(filler["total"], sentiment, wpm)
st.metric("Confidence Score", conf_score)
st.metric("Tone", tone_label(sentiment, filler["total"]))

# Key points
st.subheader("🧩 Key Points / Highlights")
key_points = extract_key_points(transcript)
if key_points:
    for kp in key_points:
        st.write("• " + kp)
else:
    st.write("No clear key points extracted. Try pasting a longer transcript or enabling speaker prompts during interview.")

# -----------------------------
# Downloadable report (existing JSON export)
# -----------------------------
report = {
    "file": uploaded.name if uploaded else None,
    "duration": duration,
    "total_words": total_words,
    "wpm": wpm,
    "filler": filler,
    "sentiment": sentiment,
    "keywords": kwcov,
    "overall_score": overall,
    "confidence_score": conf_score,
    "tone": tone_label(sentiment, filler["total"]),
    "summary": summary,
    "improvement_suggestions": improvement_suggestions(wpm, filler["total"], sentiment),
    "key_points": key_points,
    "transcript": transcript,
    "segments": segments
}

st.subheader("📥 Download Report")
st.download_button("Download JSON Report", json.dumps(report, indent=2), file_name="interview_report.json", mime="application/json")
st.download_button("Download Transcript (txt)", transcript, file_name="transcript.txt", mime="text/plain")

st.success("Analysis complete. Use the metrics, summary and suggestions to create feedback and improvement actions.")

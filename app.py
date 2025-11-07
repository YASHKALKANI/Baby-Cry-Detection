import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import soundfile as sf


# Define the CNN Model

class CryCNN(nn.Module):
    def __init__(self):
        super(CryCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 10 * 10, 128)
        self.fc2 = nn.Linear(128, 9)  # 9 classes
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load Model Safely

MODEL_PATH = "model_training/baby_cry_model_cpu.pth"

model = CryCNN()
try:
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Model load error: {e}")


# Audio Preprocessing

def preprocess_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64)
    mel = transform(waveform)
    mel = (mel - mel.mean()) / mel.std()

    mel = torch.nn.functional.interpolate(mel.unsqueeze(0), size=(40, 40), mode="bilinear").squeeze(0)
    return mel


# Prediction Function

def predict_audio(file_path):
    mel = preprocess_audio(file_path)
    mel = mel.unsqueeze(0)

    with torch.no_grad():
        output = model(mel)
        _, predicted = torch.max(output, 1)

    classes = ["belly pain", "burping", "cold_hot", "discomfort",
               "hungry", "laugh", "noise", "silence", "tired"]

    return classes[predicted.item()]


# Streamlit UI

st.title("Baby Cry Detection App")
st.write("Record OR Upload a baby cry audio file to detect the reason.")

# Choose option

option = st.radio("Select Input Method:", ["Record Audio", "Upload Audio File"])

audio_path = None


# OPTION 1: MANUAL AUDIO RECORDING

if option == "Record Audio":
    st.info("Click the button to record using your microphone.")

    duration = st.slider("Recording duration (seconds)", 3, 10, 5)
    fs = 16000

    if st.button("Start Recording"):
        st.warning("Recording... make the baby sound!")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()

        write("recorded_audio.wav", fs, audio)
        st.success("Recording complete!")
        st.audio("recorded_audio.wav")
        audio_path = "recorded_audio.wav"



# OPTION 2: AUDIO FILE UPLOAD

if option == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload Baby Cry Audio (.wav)", type=["wav"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            data, samplerate = sf.read(uploaded_file)
            sf.write(tmp.name, data, samplerate)
            audio_path = tmp.name

        st.success("Audio uploaded successfully!")
        st.audio(audio_path)


# FINAL PREDICTION

if audio_path and st.button("Analyze Cry"):
    st.write("Analyzing...")
    try:
        result = predict_audio(audio_path)
        st.success(f"The baby might be: **{result}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

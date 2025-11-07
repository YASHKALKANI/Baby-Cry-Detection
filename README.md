# Baby Cry Detection (AI/ML + Streamlit)

A machine-learning-powered web app that identifies **why a baby is crying** using audio analysis.  
The model classifies cry sounds into **9 categories**:

- belly pain  
- burping  
- cold/hot  
- discomfort  
- hungry  
- laugh  
- noise  
- silence  
- tired  

This project uses **PyTorch**, **Torchaudio**, and **Streamlit** to perform audio preprocessing, Mel spectrogram generation, and CNN-based classification.

---

## Features

- Upload audio (.wav)  
- Record live audio using microphone  
- Convert audio → Mel Spectrogram → CNN prediction  
- Supports CPU (no GPU required)  
- Clean UI built with Streamlit  
- Works on Linux, Windows, Mac  

---

## Project Structure


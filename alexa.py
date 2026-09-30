import streamlit as st # Streamlit: Used to create the web-based user interface for the Alexa application
import sounddevice as sd # SoundDevice: Used to record audio from the microphone
from scipy.io.wavfile import write # SciPy WAVFile: Used to save recorded audio as a WAV file
import whisper # Whisper: Used to convert spoken audio into text (Speech-to-Text)
import pandas as pd # Pandas: Used to load and work with the Alexa training dataset
import webbrowser # WebBrowser: Used to open websites and web pages from the application
import datetime # DateTime: Used to get and work with the current date and time
import random # Random: Used to randomly select a response, such as a joke
import re  # Regular Expression (Regex): Used for pattern matching and text processing
import os # OS: Used to interact with the operating system and manage environment variables
import pyttsx3 # pyttsx3: Used to convert text responses into spoken audio (Text-to-Speech)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier


#streamlit Page Config
st.set_page_config(page_title='Alexa Voice Assistance',layout="centered")
st.title("Alexa Voice Assistance(Mini)")
st.caption("Speech -> Intent ->Action ->Speech")

#Load whisper mode
def load_whisper():
    return whisper.load_model("base")
model = load_whisper() 

#Load and intent model

def load_intent_model():
    df = pd.read_csv("alexa_data.csv")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['prompt'])

    model_intent = MLPClassifier(
        hidden_layer_sizes=(50,25),
        max_iter=500,
        random_state=42
    )

#Audio recording
sd.default.device = (14,12)

def record_audio(filename  = "input2.wav",duration = 5,fs = 48000):
    st.info("Listening")
    recording = st.rec(int(duration * fs),samplerate = fs,channels = 1)
    sd.wait()
    write(filename,fs,recording)
    st.success("Audio Recorded")

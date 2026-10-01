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

import imageio_ffmpeg

ffmpeg_path = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

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
    model_intent.fit(X,df['intent'])
    return vectorizer,model_intent

vectorizer,intent_model = load_intent_model()
#Audio recording
sd.default.device = (1,3)

def record_audio(filename  = "input2.wav",duration = 5,fs = 48000):
    st.info("Listening")
    recording = sd.rec(int(duration * fs),samplerate = fs,channels = 1)
    sd.wait()
    write(filename,fs,recording)
    st.success("Audio Recorded")

# speech to text
def speech_to_text(audio_path):
    result = model.transcribe(audio_path)
    return result["text"].lower() 

#Intent Prediction
def intent_prediction(text):
    X_test = vectorizer.transform([text])
    return intent_model.predict(X_test)[0]


#Action Engine

def perform_action(intent,prompt = ""):
    if intent == "play_music":
        webbrowser.open("https://www.youtube.com/results?search_query=music")
        return "Playing music on YouTube"
    elif  intent == "open_website":
        if "youtube" in prompt:
            webbrowser.open("https://www.youtube.com")
            return "Opening YouTube"
        elif "google" in prompt:
            webbrowser.open("https://www.google.com")
            return "Opening Google"
        return "Which website should I open?"
    elif intent == "news":
        webbrowser.open("https://news.google.com")
        return "Here are today's headlines"
    
    elif intent == "date_time":
        now = datetime.datetime.now()
        return now.strftime("It is %I:%M %p on %B %d, %Y")

    elif intent == "jokes_fun":
        jokes = [
            "Why do programmers hate nature? Too many bugs!",
            "Why did the computer go to the doctor? It caught a virus!",
            "I would tell you a UDP joke, but you might not get it."
        ]
        return random.choice(jokes)
    elif intent == "general_qa":
        webbrowser.open(f"https://www.google.com/search?q={prompt.replace(' ', '+')}")
        return "Here is what I found on Google"
    else:
        return "Sorry, I didn't understand that."

#text to speech

if 'tts_engine' not in st.session_state:
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty("voices")
    engine.setProperty("voice",voices[1].id)
    engine.setProperty("rate",165)
    st.session_state.tts_engine = engine

def speak(text):
    engine = st.session_state.tts_engine
    try:
         if engine._inLoop:
             engine.endLoop()

    except:
        pass
    engine.stop()
    engine.say(text)
    engine.runAndWait()

# UI Control

st.markdown("### talk to Alexa")
duration =  st.slider("Recording Duration (seconds)",3,10,5)

if st.button("Record & Run Alexa"):
    record_audio(duration=duration)

    text = speech_to_text('input2.wav')
    st.success(f"🗣 You said: **{text}**")

    intent = intent_prediction(text)
    st.info(f"Detected intent: **{intent}**")

    response = perform_action(intent,text)
    st.success(f"Alexa: **{response}**")

    speak(response)

st.markdown("-----")
st.caption("Built with Streamlit + Whisper + ML Intent Model")




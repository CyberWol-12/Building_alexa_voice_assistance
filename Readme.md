# 🧠 Alexa Voice Assistant

> Hello! Welcome to my Alexa Voice Assistant project — an end-to-end AI application combining Speech Recognition, NLP, Machine Learning, and Voice Response.

![Alexa Voice Assistant Architecture](system_overflow.png)

##  Overview

This project is a **mini Alexa-style voice assistant** that converts spoken commands into text, identifies the user's intent using Machine Learning, performs an appropriate action, and responds through voice.

### Core Pipeline

```text
🎤 Voice
   ↓
🎙️ Audio Recording
   ↓
🧠 Whisper
   ↓
📝 Speech-to-Text
   ↓
🔤 TF-IDF
   ↓
🤖 MLPClassifier
   ↓
🎯 Intent
   ↓
⚙️ Action Engine
   ↓
💬 Response
   ↓
🔊 Text-to-Speech
````

## 🏗️ System Architecture

![Project Architecture](Project_arch.png)

The system is divided into four main stages:

### 1. Speech Input

* Captures voice through the microphone.
* Uses `sounddevice` for recording.
* Saves audio as a WAV file.

### 2. Speech-to-Text

![Speech to Text](speech_to_text.png)

**OpenAI Whisper** converts recorded speech into text.

```text
Audio → Whisper → Transcribed Text
```

### 3. NLP & Machine Learning

The transcribed command is converted into numerical features using **TF-IDF** and classified using **MLPClassifier**.

```text
Text
 ↓
TF-IDF
 ↓
Feature Vector
 ↓
MLPClassifier
 ↓
Predicted Intent
```

The MLP uses two hidden layers:

```text
Input
 ↓
50 Neurons
 ↓
25 Neurons
 ↓
Intent
```

Training data is stored in `alexa_data.csv` as:

```text
Prompt → Intent
```

### 4. Action Engine

![Action Engine](action_engine.png)

The predicted intent is passed to the Action Engine, which decides what the assistant should do.

| Intent         | Action                   |
| -------------- | ------------------------ |
| `play_music`   | Opens music on YouTube   |
| `open_website` | Opens requested website  |
| `news`         | Opens Google News        |
| `date_time`    | Provides date and time   |
| `jokes_fun`    | Generates a random joke  |
| `general_qa`   | Performs a Google search |

## 🔊 Voice Response

After completing an action, **pyttsx3** converts the response text into speech.

```text
Action
 ↓
Response Text
 ↓
pyttsx3
 ↓
🔊 Voice Output
```

## 🧠 Machine Learning

The project demonstrates a supervised **text classification pipeline**:

```text
alexa_data.csv
      ↓
Prompt + Intent
      ↓
TF-IDF
      ↓
Feature Matrix
      ↓
MLPClassifier
      ↓
Trained Intent Model
```

An important concept is the separation between **feature extraction** and **classification**:

* **TF-IDF** → converts text into numerical features
* **MLPClassifier** → predicts the intent

## 🛠️ Technologies

* **Python**
* **OpenAI Whisper**
* **Pandas**
* **TF-IDF**
* **Scikit-learn**
* **MLPClassifier**
* **SoundDevice**
* **SciPy**
* **FFmpeg**
* **pyttsx3**
* **Streamlit**
* **Webbrowser**
* **PyAutoGUI**

## 🚀 How It Works

1. User gives a voice command.
2. Audio is recorded and saved.
3. Whisper converts speech into text.
4. TF-IDF extracts text features.
5. MLPClassifier predicts the intent.
6. Action Engine performs the required action.
7. A response is generated.
8. pyttsx3 speaks the response.

## 🎯 Learning Focus

This project helped me understand how **Data Science and AI concepts can become part of a real application**, including:

* Speech Recognition
* NLP
* Feature Extraction
* Supervised Learning
* Neural Network Classification
* Model Inference
* Automation
* Text-to-Speech
* End-to-End AI Architecture

## 🔮 Future Improvements

* Expand intent and action coverage
* Add train/test evaluation
* Accuracy, Precision, Recall and F1-Score
* Confusion Matrix
* Error Analysis
* Confidence-based predictions
* Model saving and loading
* Conversational memory
* API integrations
* Multilingual support
* Wake-word detection

## 👩‍💻 Author

**Divya Upadhyay**

Data Science | Machine Learning | NLP | AI

This project is part of my journey of **learning by building practical AI and Data Science projects**.

Thanks for visiting and exploring my project! ⭐



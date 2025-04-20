import pyttsx3
import speech_recognition as sr
import datetime
import webbrowser
import os
import sys
import requests
from configparser import ConfigParser

# Load OpenRouter API Key
config = ConfigParser()
config.read('config.ini')
openrouter_api_key = config.get('openrouter', 'api_key')

# Text-to-Speech setup
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty('rate', 180)

def speak(audio):
    try:
        engine.say(audio)
        engine.runAndWait()
    except Exception as e:
        print(f"Speech error: {e}")

def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour < 12:
        speak("Good Morning!")
    elif hour < 18:
        speak("Good Afternoon!")
    else:
        speak("Good Evening!")
    speak("I am your AI assistant. How may I help you?")

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 1
        try:
            audio = r.listen(source, timeout=5)
            print("Recognizing...")
            query = r.recognize_google(audio, language='en-in')
            print(f"User said: {query}\n")
            return query.lower()
        except:
            speak("Sorry, I didn't catch that.")
            return ""

def get_openrouter_response(prompt):
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "anthropic/claude-3-haiku",  # You can change to e.g. 'mistralai/mixtral-8x7b'
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"OpenRouter API Error: {e}")
        return "I'm having trouble connecting to OpenRouter right now."

if __name__ == "__main__":
    wishMe()
    while True:
        query = takeCommand()
        if not query:
            continue

        if 'open youtube' in query:
            webbrowser.open("https://youtube.com")
            speak("Opening YouTube")

        elif 'open google' in query:
            webbrowser.open("https://google.com")
            speak("Opening Google")

        elif 'time' in query:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")
            speak(f"The time is {strTime}")

        elif 'exit' in query or 'stop' in query:
            speak("Goodbye!")
            sys.exit()

        else:
            speak("Let me think...")
            response = get_openrouter_response(query)
            print(f"Assistant: {response}")
            speak(response)

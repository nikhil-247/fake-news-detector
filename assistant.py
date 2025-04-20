import pyttsx3
import speech_recognition as sr
import datetime
import requests

# Replace with your OpenRouter API key
openrouter_api_key = "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"

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    engine.say(text)
    engine.runAndWait()

def get_openai_response(prompt):
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "openai/gpt-3.5-turbo",  # or "mistralai/mixtral-8x7b", etc.
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']

    except Exception as e:
        print("API Error:", e)
        return "Sorry, I couldn't get a response."

def run_voice_assistant():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5)

    try:
        query = recognizer.recognize_google(audio)
        print(f"You said: {query}")
        response = get_openai_response(query)
        speak(response)
        return response
    except Exception as e:
        print("Recognition Error:", e)

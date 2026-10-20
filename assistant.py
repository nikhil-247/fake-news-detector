import os

import pyttsx3
import requests
import speech_recognition as sr

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")


def speak(text: str) -> None:
    engine = pyttsx3.init()
    engine.setProperty("rate", 180)
    engine.say(text)
    engine.runAndWait()


def get_openai_response(prompt: str) -> str:
    """Call OpenRouter without embedding a credential in source code."""
    if not OPENROUTER_API_KEY:
        return "The assistant is not configured. Set the OPENROUTER_API_KEY environment variable."

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.RequestException:
        return "Sorry, the AI service is temporarily unavailable. Please try again."
    except (KeyError, IndexError, TypeError, ValueError):
        return "Sorry, the AI service returned an unexpected response."


def run_voice_assistant() -> str | None:
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=20)

        query = recognizer.recognize_google(audio)
        print(f"You said: {query}")
        response = get_openai_response(query)
        speak(response)
        return response
    except sr.WaitTimeoutError:
        print("No speech was detected before the timeout.")
    except sr.UnknownValueError:
        print("Speech could not be understood.")
    except sr.RequestError:
        print("The speech-recognition service is unavailable.")
    except OSError:
        print("A microphone or audio device could not be opened.")
    return None


if __name__ == "__main__":
    run_voice_assistant()

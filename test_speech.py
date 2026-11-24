"""Run this file to test Windows text-to-speech independently of the camera."""
import time
import pyttsx3

print("Initialising pyttsx3...")
engine = pyttsx3.init()
engine.setProperty("rate", 160)
engine.setProperty("volume", 1.0)

voices = engine.getProperty("voices")
print(f"Voices found: {len(voices)}")
for index, voice in enumerate(voices):
    print(index, voice.name, voice.id)

print("Speaking test phrase now...")
engine.say("EyeComm text to speech test. Can you hear me?")
engine.runAndWait()
engine.stop()
print("Speech test finished.")

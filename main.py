# ==================================================
# IMPORTS
# ==================================================

import cv2
import requests
import base64
import speech_recognition as sr
import threading
import pyttsx3
import time
from queue import Queue
from ultralytics import YOLO


# ==================================================
# SETTINGS / CONFIGURATION
# ==================================================

MIC_INDEX = 1

WAKE_PHRASES = {
    "hello assistant",
    "hey assistant"
}

THREAT_OBJECTS = {
    "knife",
    "scissors",
    "baseball bat"
}

THREAT_LEVELS = {
    "knife": "HIGH",
    "scissors": "MEDIUM",
    "baseball bat": "HIGH"
}

latest_question = None
vision_busy = False

speech_queue = Queue()


# ==================================================
# MICROPHONE DEBUG LIST
# ==================================================

for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"Microphone {index}: {name}")


# ==================================================
# AI / OBJECT DETECTION MODEL
# ==================================================

model = YOLO("yolo11s.pt")


# ==================================================
# TEXT TO SPEECH SETUP
# ==================================================

tts = pyttsx3.init()

voices = tts.getProperty("voices")

print("\nAvailable voices:")
for index, voice in enumerate(voices):
    print(index, voice.name)

if len(voices) > 1:
    tts.setProperty("voice", voices[1].id)
else:
    tts.setProperty("voice", voices[0].id)

tts.setProperty("rate", 155)
tts.setProperty("volume", 1.0)


# ==================================================
# SPEECH QUEUE / NON-BLOCKING SPEAKING
# ==================================================

def speech_worker():
    while True:
        text = speech_queue.get()
        print("Itzel:", text)
        tts.say(text)
        tts.runAndWait()
        speech_queue.task_done()


def speak(text):
    speech_queue.put(text)


speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()


# ==================================================
# VISION-LANGUAGE MODEL FUNCTION
# ==================================================

def ask_vision_model(image_path, question):
    try:
        with open(image_path, "rb") as img:
            image_base64 = base64.b64encode(img.read()).decode("utf-8")

        response = requests.post(
            "http://192.168.1.236:11434/api/generate",
            json={
                "model": "minicpm-v",
                "prompt": question,
                "images": [image_base64],
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()
        return data.get("response", "I could not find a response from the vision model.")

    except requests.exceptions.ConnectionError:
        return "My vision model is offline, boss. Ollama is not running right now."

    except requests.exceptions.Timeout:
        return "The vision model took too long to respond, boss."

    except Exception as error:
        return f"Something went wrong with my vision system, boss. Error: {error}"


# ==================================================
# BACKGROUND VISION QUESTION WORKER
# ==================================================

def process_vision_question(image_path, question):
    global vision_busy

    print("Processing vision question in background...")
    answer = ask_vision_model(image_path, question)

    print("You asked:", question)
    speak(answer)

    vision_busy = False


# ==================================================
# MICROPHONE LISTENING FUNCTION
# ==================================================

def listen_to_microphone():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 150
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.5

    with sr.Microphone(device_index=MIC_INDEX) as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening...")

        audio = recognizer.listen(source)

    try:
        speech = recognizer.recognize_google(audio)
        print("Heard:", speech)
        return speech.lower()

    except sr.UnknownValueError:
        print("Could not understand audio.")
        return ""

    except sr.RequestError:
        print("Speech recognition service error.")
        return ""


# ==================================================
# BACKGROUND VOICE LISTENER
# ==================================================

def voice_listener():
    global latest_question

    while True:
        speech = listen_to_microphone()

        print("Detected speech:", speech)

        if speech in WAKE_PHRASES:
            speak("Yes boss")

            question = listen_to_microphone()

            if question:
                print("Question detected:", question)
                latest_question = question
            else:
                print("No question detected. Returning to wake phrase mode.")


# ==================================================
# PROGRAM STARTUP
# ==================================================

print("Itzel online.")
print("Say 'hello assistant' to activate.")

listener_thread = threading.Thread(target=voice_listener, daemon=True)
listener_thread.start()


# ==================================================
# CAMERA / WINDOW SETUP
# ==================================================

cap = cv2.VideoCapture(0)

cv2.namedWindow("Itzel Vision", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Itzel Vision", 800, 800)

prev_time = time.time()


# ==================================================
# MAIN PROGRAM LOOP
# ==================================================



frame_count = 0
results = None

while True:
    ret, frame = cap.read()
    frame_count += 1

    if not ret:
        print("Camera error.")
        break


    # ==============================================
    # OBJECT DETECTION
    # ==============================================

    

    if frame_count % 3 == 0 or results is None:
        results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    # ==============================================
    # FPS COUNTER
    # ==============================================

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # ==============================================
    # THREAT DETECTION
    # ==============================================
    detected_threats = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]

        if label in THREAT_OBJECTS:
            threat_level = THREAT_LEVELS[label]

            detected_threats.append(
                f"{label} ({threat_level})"
            )

    if detected_threats:
        warning_text = "THREAT DETECTED: " + ", ".join(set(detected_threats))

        print(warning_text)

        cv2.putText(
            annotated_frame,
            warning_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )


    # ==============================================
    # VISION THINKING STATUS
    # ==============================================

    if vision_busy:
        cv2.putText(
            annotated_frame,
            "ITZEL IS THINKING...",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )


    # ==============================================
    # QUESTION ANSWERING MODE - NOW NON-BLOCKING
    # ==============================================

    if latest_question and not vision_busy:
        question_to_process = latest_question
        latest_question = None
        vision_busy = True

        cv2.imwrite("current_view.jpg", frame)

        speak("Let me look.")

        vision_thread = threading.Thread(
            target=process_vision_question,
            args=("current_view.jpg", question_to_process),
            daemon=True
        )

        vision_thread.start()


    # ==============================================
    # DISPLAY WINDOW
    # ==============================================

    square_frame = cv2.resize(annotated_frame, (800, 800))

    cv2.putText(
        square_frame,
        "LISTENING FOR WAKE WORD",
        (20, 760),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow("Itzel Vision", square_frame)


    # ==============================================
    # QUIT PROGRAM
    # ==============================================

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

    if cv2.getWindowProperty("Itzel Vision", cv2.WND_PROP_VISIBLE) < 1:
        break


# ==================================================
# CLEANUP
# ==================================================

cap.release()
cv2.destroyAllWindows()
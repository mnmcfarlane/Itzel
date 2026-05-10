from openwakeword.model import Model
import pyaudio
import numpy as np

model = Model(
    wakeword_models=["alexa"],
    inference_framework="onnx"
)

# Audio settings
CHUNK = 1280
RATE = 16000

# Start microphone stream
audio = pyaudio.PyAudio()

stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

print("Listening for wake word...")

while True:

    # Read microphone audio
    audio_data = stream.read(CHUNK)

    # Convert audio into numpy array
    audio_array = np.frombuffer(
        audio_data,
        dtype=np.int16
    )

    # Run wake word prediction
    prediction = model.predict(audio_array)

    # Print prediction scores
    print(prediction)

    # Detect wake word
    if prediction["alexa"] > 0.5:
        print("WAKE WORD DETECTED")
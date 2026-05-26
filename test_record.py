"""
test_record.py
--------------
Testa APENAS gravacao + salvamento do WAV, sem TF.
Rode:  python test_record.py
Depois abra recordings/test.wav para verificar o audio.
"""

import wave
from pathlib import Path

import numpy as np
import pyaudio

SAMPLE_RATE = 22050
DURATION = 5
CHUNK = 1024
OUT = Path(__file__).resolve().parent / "recordings" / "test.wav"


def main():
    pa = pyaudio.PyAudio()

    # Lista dispositivos de entrada disponiveis
    print("Dispositivos de entrada:")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"  [{i}] {info['name']}  (canais={info['maxInputChannels']})")

    print(f"\nGravando {DURATION}s no dispositivo padrao...")
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                     input=True, frames_per_buffer=CHUNK)

    frames = []
    n = int(SAMPLE_RATE / CHUNK * DURATION)
    for i in range(n):
        frames.append(stream.read(CHUNK, exception_on_overflow=False))
        p = int((i + 1) / n * 30)
        print(f"\r  [{'#'*p}{'-'*(30-p)}]", end="", flush=True)
    print("  ok")

    stream.stop_stream()
    stream.close()
    pa.terminate()

    raw = b"".join(frames)
    audio = np.frombuffer(raw, dtype=np.int16)
    print(f"\nAmplitude max: {audio.max()}  min: {audio.min()}")
    print(f"RMS: {np.sqrt(np.mean(audio.astype(np.float32)**2)):.1f}")

    OUT.parent.mkdir(exist_ok=True)
    with wave.open(str(OUT), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw)

    print(f"\nSalvo em: {OUT}")
    print("Abra o arquivo para verificar se o audio foi capturado.")


if __name__ == "__main__":
    main()

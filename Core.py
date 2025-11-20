# =====================================================
#                       IMPORTS
# =====================================================

from vosk import Model, KaldiRecognizer
from settings.model_selector import select_vosk_model 
from settings.config import RANDOM_COMMAND_RESPONSES, additional_phrases_response, models_path
from settings.Models.Owner_AI.ai_owner import OwnerDetector

import cv2
import numpy as np
import pygame
import pyaudio
import logging
import os
import random
import json
import threading
import time


# =====================================================
#              ГЛОБАЛЬНИЙ ФЛАГ ВЛАСНИКА
# =====================================================

owner_detected = False
owner_detector = OwnerDetector(
    yolo_path="settings/Models/Owner_AI/yolov8n-face.pt",
    owner_model_path="settings/Models/Owner_AI/owner_cnn.pth"
)




# =====================================================
#                     CAMERA LOOP
# =====================================================

def camera_loop():
    global owner_detected

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Не вдалося відкрити камеру.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # ---- YOLO FACE DETECTOR ----
        boxes = owner_detector.detect_faces(frame)

        if not boxes:
            owner_detected = False
            owner_detector.sound_played = False

            cv2.imshow("Jarvis Camera", frame)
            cv2.waitKey(1)
            continue

        # ---- PROCESS EACH FACE ----
        for box, conf in boxes:
            x1, y1, x2, y2 = box
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # ---- CNN OWNER CHECK ----
            prob = owner_detector.classify_owner(face)
            is_owner = prob > 0.5

            label = f"{'OWNER' if is_owner else 'UNKNOWN'}  {prob:.2f}  conf:{conf:.2f}"
            color = (0, 255, 0) if is_owner else (0, 0, 255)

            frame = owner_detector.draw_label(frame, box, label, color)

            # ---- STATE LOGIC ----
            if is_owner:
                owner_detected = True
                if not owner_detector.sound_played:
                    print("[AI] ✔ Власника впізнано")
                    owner_detector.sound_played = True
            else:
                owner_detected = False
                owner_detector.sound_played = False

        cv2.imshow("Jarvis Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



# =====================================================
#                 MICROPHONE + VOSK
# =====================================================

def microphone_initialization():
    pygame.mixer.init()

    model_path = select_vosk_model(models_path)
    print(f"[VOSK] Завантажую модель: {model_path}")

    model = Model(model_path)
    rec = KaldiRecognizer(model, 8000)

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=8000,
        input=True,
        frames_per_buffer=8000
    )
    stream.start_stream()

    return model, rec, stream


def listen_for_commands(model, rec, stream):
    """Розпізнавання голосу лише коли власник у кадрі."""
    global owner_detected

    while True:

        if not owner_detected:
            print("[WAIT] Власник не в кадрі...", end="\r")
            time.sleep(0.1)
            continue

        data = stream.read(4000, exception_on_overflow=False)

        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "").strip()

            if text:
                print(f"[FULL] {text}")

        else:
            partial = json.loads(rec.PartialResult()).get("partial", "").strip()
            if partial:
                print(f"[LIVE] {partial}", end="\r")



# =====================================================
#                     AUDIO SYSTEM
# =====================================================

def play_audio(file_path):
    if not isinstance(file_path, str):
        logging.error("Неправильний тип даних (має бути рядок!)")
        return

    if not os.path.exists(file_path):
        logging.error(f"Файл не знайдено: {file_path}")
        return

    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except pygame.error as e:
        logging.error(f"Помилка відтворення: {e}")


def play_random_additional_phrases_response():
    random_file = random.choice(list(additional_phrases_response))
    play_audio(random_file)


def play_random_response():
    available = [file for file in RANDOM_COMMAND_RESPONSES if os.path.exists(file)]

    if not available:
        logging.warning("Немає доступних аудіофайлів.")
        return

    play_audio(random.choice(available))

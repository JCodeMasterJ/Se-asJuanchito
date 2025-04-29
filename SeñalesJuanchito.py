# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 14:44:24 2025

@author: juanf
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from tensorflow.keras.models import load_model

# Cargar modelo
modelo = load_model('mejor_modelo_letras.h5')

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Variables de control
detectar_letras = False
palabra = ""
ultima_prediccion = None
contador_prediccion = 0
frames_consecutivos = 5
umbral_confianza = 0.7
tiempo_ultima_letra = 0
delay_letras = 1.5

# Letras estaticas
letras = ['A', 'B', 'C', 'D', 'E', 'F', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y']

# Variables dinámicas
detectando_movimiento = False
punto_inicio_meñique = None
punto_inicio_indice = None
punto_inicio_medio = None
inicio_movimiento = 0
duracion_movimiento = 0.5  # segundos
frame_counter = 0

# Funciones auxiliares
def extraer_landmarks(hand_landmarks):
    puntos = []
    for lm in hand_landmarks.landmark:
        puntos.extend([lm.x, lm.y, lm.z])
    return np.array(puntos)

def detectar_movimiento_j(punto_inicio, punto_fin):
    delta_x = punto_fin[0] - punto_inicio[0]
    delta_y = punto_fin[1] - punto_inicio[1]
    if delta_y > 0.12 and delta_x > 0.06:  # Ligeramente más exigente
        return True
    return False

def detectar_movimiento_h(punto_inicio_indice, punto_fin_indice, punto_inicio_medio, punto_fin_medio):
    delta_y_indice = punto_fin_indice[1] - punto_inicio_indice[1]
    delta_y_medio = punto_fin_medio[1] - punto_inicio_medio[1]
    delta_x_indice = abs(punto_fin_indice[0] - punto_inicio_indice[0])
    delta_x_medio = abs(punto_fin_medio[0] - punto_inicio_medio[0])
    if delta_y_indice > 0.12 and delta_y_medio > 0.12 and delta_x_indice < 0.04 and delta_x_medio < 0.04:
        return True
    return False

# Iniciar cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(rgb)

    frame_counter += 1

    if resultado.multi_hand_landmarks and detectar_letras:
        for hand_landmarks in resultado.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            indice = (landmarks[8].x, landmarks[8].y)
            medio = (landmarks[12].x, landmarks[12].y)
            meñique = (landmarks[20].x, landmarks[20].y)

            if not detectando_movimiento:
                punto_inicio_indice = indice
                punto_inicio_medio = medio
                punto_inicio_meñique = meñique
                inicio_movimiento = time.time()
                detectando_movimiento = True
                frame_counter = 0
            else:
                if frame_counter % 3 == 0:  # Solo cada 3 frames
                    if time.time() - inicio_movimiento >= duracion_movimiento:
                        # Prioridad primero detectar J
                        if detectar_movimiento_j(punto_inicio_meñique, meñique):
                            if time.time() - tiempo_ultima_letra > delay_letras:
                                palabra += "J"
                                print("Letra dinámica detectada: J")
                                tiempo_ultima_letra = time.time()
                        # Luego detectar H
                        elif detectar_movimiento_h(punto_inicio_indice, indice, punto_inicio_medio, medio):
                            if time.time() - tiempo_ultima_letra > delay_letras:
                                palabra += "H"
                                print("Letra dinámica detectada: H")
                                tiempo_ultima_letra = time.time()

                        detectando_movimiento = False  # Reiniciar para próximo movimiento

            # Solo hacer predicción estática si no estoy detectando movimiento
            if not detectando_movimiento:
                vector_landmarks = extraer_landmarks(hand_landmarks)
                vector_landmarks = np.expand_dims(vector_landmarks, axis=0)

                prediccion = modelo.predict(vector_landmarks)[0]
                max_confianza = np.max(prediccion)
                letra_idx = np.argmax(prediccion)
                letra_actual = letras[letra_idx]

                if max_confianza >= umbral_confianza:
                    if letra_actual == ultima_prediccion:
                        contador_prediccion += 1
                    else:
                        ultima_prediccion = letra_actual
                        contador_prediccion = 1

                    if contador_prediccion == frames_consecutivos:
                        if time.time() - tiempo_ultima_letra > delay_letras:
                            palabra += letra_actual
                            tiempo_ultima_letra = time.time()

                        contador_prediccion = 0
                        ultima_prediccion = None

    # Mostrar palabra detectada
    cv2.putText(frame, palabra, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    if detectar_letras:
        cv2.putText(frame, "MODO DETECTANDO (d: iniciar, p: parar, r: resetear)", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "MODO DETENIDO (d: iniciar, p: parar, r: resetear)", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Reconocimiento LSC', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        detectar_letras = True
        ultima_prediccion = None
        contador_prediccion = 0
        tiempo_ultima_letra = 0
    elif key == ord('p'):
        detectar_letras = False
        print(f"Palabra detectada: {palabra}")
    elif key == ord('r'):
        palabra = ""
        ultima_prediccion = None
        contador_prediccion = 0
        tiempo_ultima_letra = 0
        print("Palabra reiniciada. Listo para detectar una nueva palabra.")

cap.release()
cv2.destroyAllWindows()
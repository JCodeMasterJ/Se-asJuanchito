import cv2
import numpy as np
import mediapipe as mp
import time
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
modelo = load_model('mejor_modelo_letras.h5')

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

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

# Variables para detectar movimiento de "J" y "H"
detectando_j = False
detectando_h = False
detectando_s = False
detectando_z = False
inicio_movimiento_j = None
inicio_movimiento_h = None
inicio_movimiento_s = None
inicio_movimiento_z = None

# Función para extraer landmarks
def extraer_landmarks(hand_landmarks):
    puntos = []
    for lm in hand_landmarks.landmark:
        puntos.extend([lm.x, lm.y, lm.z])
    return np.array(puntos)

# Función para detectar movimiento "J" (meñique)
def detectar_movimiento_j(punto_inicio, punto_fin):
    delta_x = punto_fin[0] - punto_inicio[0]
    delta_y = punto_fin[1] - punto_inicio[1]
    if delta_y > 0.1 and delta_x > 0.05:  # Movimientos hacia abajo y derecha
        return True
    return False

# Función para detectar movimiento "H" (índice y medio juntos, movimiento vertical)
def detectar_movimiento_h(punto_inicio_indice, punto_fin_indice, punto_inicio_medio, punto_fin_medio):
    delta_y_indice = punto_fin_indice[1] - punto_inicio_indice[1]
    delta_y_medio = punto_fin_medio[1] - punto_inicio_medio[1]
    delta_x_indice = abs(punto_fin_indice[0] - punto_inicio_indice[0])
    delta_x_medio = abs(punto_fin_medio[0] - punto_inicio_medio[0])
    
    # Queremos movimiento principalmente vertical
    if delta_y_indice > 0.1 and delta_y_medio > 0.1 and delta_x_indice < 0.05 and delta_x_medio < 0.05:
        return True
    return False

# Función corregida para detectar movimiento "S" (solo dedo índice)
def detectar_movimiento_s(punto_inicio_indice, punto_fin_indice):
    delta_x = punto_fin_indice[0] - punto_inicio_indice[0]
    delta_y = punto_fin_indice[1] - punto_inicio_indice[1]

    # Movimiento en S: cambio de x importante + bajada en y
    if abs(delta_x) > 0.05 and delta_y > 0.1:
        return True
    return False

# # Función para detectar movimiento "Z" (índice y medio juntos, trayectoria tipo Z)
# def detectar_movimiento_z(punto_inicio, punto_fin):
#     delta_x = punto_fin[0] - punto_inicio[0]
#     delta_y = punto_fin[1] - punto_inicio[1]

#     # Movimiento Z: cambio horizontal grande (derecha-izquierda-derecha) y algo de bajada
#     if abs(delta_x) > 0.1 and delta_y > 0.05:
#         return True
#     return False
# Función mejorada para detectar movimiento "Z" (índice como referencia)
def detectar_movimiento_z(punto_inicio, punto_fin):
    delta_x = punto_fin[0] - punto_inicio[0]
    delta_y = punto_fin[1] - punto_inicio[1]

    # Reglas mejoradas:
    # - Movimiento horizontal inicial grande (x positivo)
    # - Luego un retroceso leve en x (negativo)
    # - Descenso claro en y (positivo)

    if delta_x > 0.05 and delta_y > 0.05:
        # Movimiento global hacia la derecha y hacia abajo
        return True
    return False




# Abrir cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(rgb)

    if resultado.multi_hand_landmarks and detectar_letras:
        for hand_landmarks in resultado.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            meñique = (landmarks[20].x, landmarks[20].y)
            indice = (landmarks[8].x, landmarks[8].y)
            medio = (landmarks[12].x, landmarks[12].y)

            # Detectar movimiento "J"
            if not detectando_j:
                inicio_movimiento_j = meñique
                detectando_j = True
                tiempo_inicio_j = time.time()
            else:
                tiempo_actual = time.time()
                if tiempo_actual - tiempo_inicio_j > 0.5:
                    if detectar_movimiento_j(inicio_movimiento_j, meñique):
                        tiempo_actual_letra = time.time()
                        if tiempo_actual_letra - tiempo_ultima_letra > delay_letras:
                            palabra += "J"
                            print("Letra dinámica detectada: J")
                            tiempo_ultima_letra = tiempo_actual_letra
                    detectando_j = False

            # Detectar movimiento "H"
            if not detectando_h:
                inicio_movimiento_h_indice = indice
                inicio_movimiento_h_medio = medio
                detectando_h = True
                tiempo_inicio_h = time.time()
            else:
                tiempo_actual = time.time()
                if tiempo_actual - tiempo_inicio_h > 0.5:
                    if detectar_movimiento_h(inicio_movimiento_h_indice, indice, inicio_movimiento_h_medio, medio):
                        tiempo_actual_letra = time.time()
                        if tiempo_actual_letra - tiempo_ultima_letra > delay_letras:
                            palabra += "H"
                            print("Letra dinámica detectada: H")
                            tiempo_ultima_letra = tiempo_actual_letra
                    detectando_h = False
            # Detectar movimiento "S" (solo con el índice)
            if not detectando_s:
                inicio_movimiento_s_indice = indice
                detectando_s = True
                tiempo_inicio_s = time.time()
            else:
                tiempo_actual = time.time()
                if tiempo_actual - tiempo_inicio_s > 0.5:
                    if detectar_movimiento_s(inicio_movimiento_s_indice, indice):
                        tiempo_actual_letra = time.time()
                        if tiempo_actual_letra - tiempo_ultima_letra > delay_letras:
                            palabra += "S"
                            print("Letra dinámica detectada: S")
                            tiempo_ultima_letra = tiempo_actual_letra
                    detectando_s = False
            # Detectar movimiento "Z" (usando el índice como referencia)
            if not detectando_z:
                inicio_movimiento_z = indice
                detectando_z = True
                tiempo_inicio_z = time.time()
            else:
                tiempo_actual = time.time()
                if tiempo_actual - tiempo_inicio_z > 0.5:
                    if detectar_movimiento_z(inicio_movimiento_z, indice):
                        tiempo_actual_letra = time.time()
                        if tiempo_actual_letra - tiempo_ultima_letra > delay_letras:
                            palabra += "Z"
                            print("Letra dinámica detectada: Z")
                            tiempo_ultima_letra = tiempo_actual_letra
                    detectando_z = False
            

            # Solo predecimos con modelo si no estamos detectando movimiento dinámico
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
                    tiempo_actual_letra = time.time()
                    if tiempo_actual_letra - tiempo_ultima_letra > delay_letras:
                        palabra += letra_actual
                        tiempo_ultima_letra = tiempo_actual_letra

                    contador_prediccion = 0
                    ultima_prediccion = None

    # Mostrar palabra en pantalla
    cv2.putText(frame, palabra, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    if detectar_letras:
        cv2.putText(frame, "Modo: DETECTANDO (d: iniciar, p: parar, r: resetear)", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Modo: DETENIDO (d: iniciar, p: parar, r: resetear)", (10, 120),
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
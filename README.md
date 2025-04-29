
# Detección del Lenguaje de Señas Colombiano (LSC)

Este proyecto tiene como objetivo **reconocer palabras formadas por señas del Lenguaje de Señas Colombiano (LSC)** a partir de gestos realizados con la mano frente a una cámara, usando técnicas de Machine Learning (ML) y visión por computador.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8b76c331-2e7c-40f0-a50e-9afc9528edbe" width="600"/>
</p>

---

## Descripción General del Dataset

Se desarrolló un modelo de **clasificación de imágenes entrenado en Google Colab**, utilizando dos fuentes principales de datos:

- Un **dataset propio**, creado por los estudiantes de la asignatura electiva Python de la Universidad Santo Tomás.
- Un **dataset de acceso abierto** proporcionado por la **Universidad del Cauca**, disponible en el siguiente DOI:  
  🔗 [https://doi.org/10.1016/j.dib.2024.111213](https://doi.org/10.1016/j.dib.2024.111213)

El dataset complementario de la Universidad del Cauca cuenta con más de **400 imágenes por cada letra**, lo que permitió **reforzar y mejorar el rendimiento del modelo** entrenado. Cabe resaltar que ambos datasets incluyen únicamente **gestos estáticos** del alfabeto en LSC (por lo tanto, no se incluyen letras como "J" o "H" que requieren movimiento).

### Características del dataset

- Se aprovechó el curso gratuito de **Academy Edutin** para aprender sobre el Lenguaje de Señas Colombiano (LSC) y se incorporaron señas adicionales como **"rr"**, que forma parte del abecedario en LSC.
- Todas las imágenes tienen un tamaño de **640x480 píxeles**.
- Para el **dataset estático**:
  - Se aplicaron técnicas de **data augmentation** para aumentar la variabilidad y robustez de los datos.
  - A cada imagen se le extrajeron **21 puntos landmarks** por mano (coordenadas **x, y, z**), generando un **vector de 63 valores**.

---

### Visualización de los datasets

**Dataset propio de los estudiantes de la electiva Python:**  
<p align="center">
  <img src="https://github.com/user-attachments/assets/64a581f5-67d1-4857-80a4-c45fc87a87bc" width="600"/>
</p>

**Dataset Universidad del Cauca:**  
<p align="center">
  <img src="https://github.com/user-attachments/assets/24083207-d1c2-4437-991a-f8b1d277ba3c" width="600"/>
</p>

---

##  Tecnologías utilizadas

- Google Colab
- Python (Anaconda / Spyder)
- TensorFlow / Keras
- MediaPipe Hands
- OpenCV

---

##  Proceso de desarrollo

1. **Limpieza y organización del dataset.**
2. **Extracción de características (landmarks)** de las manos utilizando MediaPipe, guardando las coordenadas en un archivo `.csv`.
3. **Entrenamiento del modelo de clasificación** en Google Colab a partir de los valores numéricos del `.csv`, representando los puntos clave de la mano en cada imagen.
4. **Exportación del modelo entrenado (.h5)** y carga en el entorno de desarrollo Spyder (Anaconda).
5. Programación del script (python) para la **detección de landmarks en tiempo real** mediante MediaPipe para generar palabras.
6. **Clasificación en tiempo real desde la cámara**, utilizando el modelo previamente entrenado.

---
## Entrenamiento del Modelo

El modelo fue entrenado utilizando un **archivo `.csv`**, el cual contenía la relacion entre:

- Las imágenes del dataset.
- Los **landmarks** (puntos críticos de la mano) extraídos con MediaPipe, representados mediante sus coordenadas **x, y, z**.

### Arquitectura del modelo

Se utilizó un modelo **secuencial** con la siguiente estructura:

- **Capa de entrada**: recibe los vectores de 63 características (21 puntos * 3 coordenadas).
- **Dos capas ocultas**: capas densas (fully connected) con funciones de activación `ReLU`.
- **Capa de salida**: capa densa con función de activación `Softmax`, para la clasificación multiclase de las letras.

### Resultados

- El modelo alcanzó un **70% de precisión** en el conjunto de validación.
- El entrenamiento fue realizado en **Google Colab**, optimizando la función de pérdida `categorical_crossentropy` con el optimizador `Adam`.

---

### Gráfica de Precisión del Modelo


*A continuación se muestra la gráfica de la evolución de la precisión durante el entrenamiento:*

<p align="center">
  <img src="https://github.com/user-attachments/assets/4a2629af-9e0d-442e-8e9b-35563d2bf843" width="500"/>
</p>

## Resultados

En la aplicación programada en Python (utilizando Spyder) se integró el modelo entrenado para el reconocimiento de señas. Es importante destacar que el modelo original fue entrenado únicamente con **letras estáticas** del alfabeto en LSC.

Las **letras dinámicas** (como "J" o "H") fueron implementadas posteriormente mediante **funciones adicionales en Python**, aprovechando el análisis de **secuencias de landmarks** detectados en tiempo real con MediaPipe.

El funcionamiento general de la aplicación se basa en un **sistema de captura de imágenes en secuencia**, que permite:

- Detectar si la seña corresponde a una **letra estática** (clasificada por el modelo) o a una **letra dinámica** (detectada por las funciones personalizadas de análisis de movimiento).
- Escribir palabras letra por letra conforme se reconocen las señas frente a la cámara.
- Utilizar un **modo dinámico** y un **modo estático** de forma independiente, optimizando así la precisión de reconocimiento en cada caso.
- Incorporar **funciones especiales mediante teclas de control** para gestionar la ejecución:
  
  - **d**: iniciar la detección.
  - **r**: reiniciar la palabra escrita.
  - **p**: pausar el reconocimiento.
  - **q**: cerrar la cámara y terminar la aplicación.

---

### Captura de la aplicación en ejecución

*A continuación se muestra una imagen de la aplicación corriendo en Spyder:*

<p align="center">
  <img src="https://github.com/user-attachments/assets/63d16c5f-c1b9-4be5-a2d1-1823d7fa99d6" width="600"/>
</p>

---

##  Recomendaciones de Uso

Para garantizar un funcionamiento óptimo de la aplicación, se sugiere tener en cuenta las siguientes recomendaciones:

-  **Iluminación adecuada**: utiliza el sistema en un entorno bien iluminado y con fondo neutro para mejorar la detección de los landmarks de la mano.
-  **Evitar accesorios**: no usar anillos, relojes u otros objetos que puedan interferir con el reconocimiento de la mano.
-  **Velocidad de captura**: la aplicación opera a **3 FPS** (frames por segundo) para mantener un equilibrio entre precisión y rendimiento. Aumentar esta tasa puede provocar ralentización del video en tiempo real.
-  **Distancia a la cámara**: mantén una distancia estable y visible frente a la cámara; se recomienda entre 40 cm y 80 cm.
-  **Uso separado de modos**: aunque el sistema permite mezclar letras estáticas y dinámicas, se recomienda utilizar **modo dinámico o modo estático por separado** para obtener mejores resultados.
   **No cerrar la app forzadamente**: utilizar la tecla **q** para cerrar la cámara correctamente y evitar conflictos en procesos posteriores.

---

Con estas buenas prácticas, el sistema puede lograr una mayor precisión. Se espera que en caso de ser utilizado pueda ser muy útil para futuros proyectos.


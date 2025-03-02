import re
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import datetime


# recursos necesarios de NLTK
nltk.download("punkt")

# ampliando el dataset: 10 ejemplos por categoría
textos_tecnologia = [
    "La inteligencia artificial está revolucionando la tecnología.",
    "El desarrollo de software mejora la productividad.",
    "Nuevas aplicaciones móviles facilitan la vida diaria.",
    "La computación en la nube permite procesar grandes datos.",
    "Los avances en robótica están cambiando la industria.",
    "Innovaciones en hardware impulsan la tecnología.",
    "La ciberseguridad es clave en el mundo digital.",
    "El internet de las cosas conecta dispositivos inteligentes.",
    "El machine learning optimiza procesos empresariales.",
    "La automatización mejora la eficiencia en las fábricas.",
]

textos_deportes = [
    "El equipo de fútbol ganó el campeonato tras un gran partido.",
    "La final de baloncesto fue espectacular y llena de emoción.",
    "El atleta batió el récord mundial en los 100 metros.",
    "El torneo de tenis reunió a los mejores del mundo.",
    "El entrenamiento diario mejora el rendimiento deportivo.",
    "La estrategia del entrenador llevó al equipo a la victoria.",
    "El estadio se llenó de fans durante el partido decisivo.",
    "El ciclismo de montaña exige resistencia y técnica.",
    "El maratón fue un desafío para todos los participantes.",
    "El campeonato de natación mostró un alto nivel competitivo.",
]


textos = textos_tecnologia + textos_deportes

etiquetas = [0] * len(textos_tecnologia) + [1] * len(textos_deportes)


def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúüñ\s]", "", texto)
    return word_tokenize(texto)


tokens = [limpiar_texto(texto) for texto in textos]
print("Ejemplo de tokens:", tokens[:3])  # Mostrar los primeros 3 para verificar


vector_size = 50
model_w2v = Word2Vec(
    sentences=tokens,
    vector_size=vector_size,
    window=3,
    min_count=1,
    workers=4,
    epochs=100,
)


# Función para obtener la representación vectorial promedio de un documento
def obtener_vector_documento(token_list, model):
    vectores = [model.wv[token] for token in token_list if token in model.wv]
    if vectores:
        return np.mean(vectores, axis=0)
    else:
        return np.zeros(model.vector_size)


X = np.array([obtener_vector_documento(token_list, model_w2v) for token_list in tokens])
y = np.array(etiquetas)

# Convertir las etiquetas a formato one-hot para la red neuronal
num_classes = len(set(etiquetas))
y_categorical = to_categorical(y, num_classes=num_classes)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.25, random_state=42
)

# Construir una arquitectura de red neuronal más simple
model_nn = Sequential(
    [
        Input(shape=(vector_size,)),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ]
)

model_nn.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
model_nn.summary()

# Usar EarlyStopping para detener el entrenamiento si no hay mejora
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Entrena tu modelo incluyendo el callback de TensorBoard
history = model_nn.fit(
    X_train, y_train, epochs=10, validation_split=0.2, callbacks=[tensorboard_callback]
)


# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model_nn.evaluate(X_test, y_test, verbose=0)
print(f"\nAccuracy en el conjunto de prueba: {accuracy:.2f}")

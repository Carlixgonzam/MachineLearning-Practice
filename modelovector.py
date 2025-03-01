import nltk
import re
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.manifold import TSNE
from nltk.corpus import brown

# descargo el dataset

nltk.download("brown")
nltk.download("punkt")


# se preprocesan los datos
corpus = brown.sents()  # se obtiene las oraciones del corpus
print("Ejemplo de datos:", corpus[:3])


# tokens limpios
def limpiar_texto(texto):
    texto = " ".join(texto)
    texto = texto.lower()
    texto = re.sub(r"[^a-zA-Záéíóúñ ]", "", texto)  # elimino caracteres especiales
    return word_tokenize(texto)


# limpio todas las frases
tokens = [limpiar_texto(frase) for frase in corpus]

print("Ejemplo de tokens:", tokens[:3])  # Ver las primeras frases tokenizadas

# modelo Word2Vec
modelo_word2vec = Word2Vec(
    sentences=tokens,  # lista de listas de palabras
    vector_size=200,  # mayor precisión con más dimensiones
    window=5,  # contexto de palabras a considerar
    min_count=2,  # ignorar palabras poco frecuentes
    workers=4,  # número de núcleos CPU
    sg=1,  # Skip-Gram
    epochs=100,
)


modelo_word2vec.save("modelo_grande_word2vec.bin")


print("Vector de 'learning':", modelo_word2vec.wv["learning"])


similares = modelo_word2vec.wv.most_similar("science", topn=10)
print("Palabras más similares a 'science':", similares)


palabras = modelo_word2vec.wv.index_to_key[:100]
vectores = np.array([modelo_word2vec.wv[word] for word in palabras])


tsne = TSNE(n_components=2, random_state=42, perplexity=5)
vectores_reducidos = tsne.fit_transform(vectores)

# Graficamos
plt.figure(figsize=(12, 8))
plt.scatter(vectores_reducidos[:, 0], vectores_reducidos[:, 1], marker="o")

# Anotamos cada palabra en el gráfico
for i, word in enumerate(palabras):
    plt.annotate(word, (vectores_reducidos[i, 0], vectores_reducidos[i, 1]))

plt.title("Representación de palabras en 2D con Word2Vec")
plt.xlabel("Dimensión 1")
plt.ylabel("Dimensión 2")
plt.show()

# se guarda el modelo
modelo_word2vec.save("modelo_grande_word2vec.bin")

# cargar más tarde
modelo_cargado = Word2Vec.load("modelo_grande_word2vec.bin")

# probar después de cargarlo
print("Palabras similares a 'machine':", modelo_cargado.wv.most_similar("machine"))

model = Word2Vec(
    sentences=tokens, vector_size=100, window=5, min_count=1, workers=4, epochs=50
)

# 2. Extraer las palabras y sus vectores
words = list(model.wv.index_to_key)  # Lista de todas las palabras entrenadas
word_vectors = np.array([model.wv[word] for word in words])

# 3. Reducir a 2 dimensiones usando t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
vectors_2d = tsne.fit_transform(word_vectors)

# 4. Graficar los resultados
plt.figure(figsize=(12, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], marker="o")

# Anotar cada punto con la palabra correspondiente
for i, word in enumerate(words):
    plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]))

plt.title("Visualización de Word Embeddings con t-SNE")
plt.xlabel("Dimensión 1")
plt.ylabel("Dimensión 2")
plt.show()

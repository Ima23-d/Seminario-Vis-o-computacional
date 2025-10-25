import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from PIL import Image
import requests
from io import BytesIO

# URL da imagem
imagem_caminho = "https://plus.unsplash.com/premium_photo-1689977968861-9c91dbb16049?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8cGVzc29hJTIwc29ycmluZG98ZW58MHx8MHx8fDA%3D&fm=jpg&q=60&w=3000"

# --- Carregar a imagem ---
try:
    if imagem_caminho.startswith("http"):
        resposta = requests.get(imagem_caminho)
        imagem_original = Image.open(BytesIO(resposta.content))
    else:
        imagem_original = Image.open(imagem_caminho)
except Exception as erro:
    print(f"Erro ao carregar a imagem: {erro}")
    exit()

# --- Pré-processamento ---
imagem_redimensionada = imagem_original.resize((28, 28))
imagem_cinza_array = np.array(imagem_redimensionada.convert('L'))
imagem_normalizada = imagem_cinza_array / 255.0

# Cria o vetor achatado (flatten)
vetor_flat = imagem_normalizada.flatten()

rotulo = np.array([1])
rotulo_categorizado = to_categorical(rotulo, num_classes=2)

# --- Modelo simples ---
modelo = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

modelo.compile(optimizer=SGD(learning_rate=0.01),
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# --- Treinamento ---
modelo.fit(vetor_flat.reshape(1, -1), rotulo_categorizado, epochs=5, batch_size=1, verbose=1)

# --- Mostrar as imagens ---
plt.figure(figsize=(16, 5))

# 1️⃣ Imagem original
plt.subplot(1, 4, 1)
plt.imshow(imagem_original)
plt.title('Imagem Original')
plt.axis('off')

# 2️⃣ Imagem em tons de cinza
plt.subplot(1, 4, 2)
plt.imshow(imagem_normalizada, cmap='gray')
plt.title('Imagem em Tons de Cinza (28x28)')
plt.axis('off')

# 3️⃣ Mapa de valores dos pixels (com números)
plt.subplot(1, 4, 3)
plt.imshow(imagem_normalizada, cmap='gray')
plt.title('Matriz com Valores de Pixels')
plt.axis('off')

linhas, colunas = imagem_normalizada.shape
for i in range(linhas):
    for j in range(colunas):
        valor = imagem_cinza_array[i, j]
        cor_texto = 'white' if valor < 128 else 'black'
        plt.text(j, i, f'{valor:2.0f}', ha='center', va='center', color=cor_texto, fontsize=6)

plt.tight_layout()
plt.show()

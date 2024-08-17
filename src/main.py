import os
from deepface import DeepFace
from rmn import RMN
import cv2

rmn = RMN()
img_path = "../data/test/0/32298.png"  # Substitua pelo caminho da imagem

# Analisar a imagem e detectar emoções
result = DeepFace.analyze(img_path=img_path, actions=['emotion'])

# Exibir o resultado
print("Emoções detectadas:")
print(result[0]["emotion"])

# Detectar emoções na imagem
image = cv2.imread(img_path)
predictions = rmn.detect_emotion_for_single_frame(image)

# Exibir o resultado
print("Emoções detectadas:")
print(predictions)
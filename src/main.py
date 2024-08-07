import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_data_dir = "./data/test"

# Gerador de dados para o conjunto de teste
datagen = ImageDataGenerator(rescale=1./255)
test_generator = datagen.flow_from_directory(
    test_data_dir,
    target_size=(48, 48),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

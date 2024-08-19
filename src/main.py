import os
from deepface import DeepFace
from rmn import RMN
import cv2
import numpy as np
import pandas as pd
import time

dic_emotion = {
    "angry": '0',
    "disgust": '1',
    "fear": '2',
    "happy": '3',
    "sad": '4',
    "surprise": '5',
    "neutral": '6'
}

def load_images_from_folder(folder):
    images = []
    labels = []
    image_ids = []
    class_names = os.listdir(folder)
    class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(folder, class_name)
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))
                img = cv2.convertScaleAbs(img, alpha=1.3, beta=40)
                images.append(img)
                labels.append(class_indices[class_name])
                image_ids.append(filename)

    return np.array(images), np.array(labels), class_indices, image_ids

def evaluate_deepface(images, labels, class_indices, image_ids):
    y_true = []
    y_pred = []
    confidences = []
    times = []
    
    for img, true_label, img_id in zip(images, labels, image_ids):
        start_time = time.time()
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        end_time = time.time()
        dominant_emotion = result[0]['dominant_emotion']
        emotion_index = dic_emotion[dominant_emotion]
        confidence = result[0]['face_confidence']
        
        y_true.append(true_label)
        y_pred.append(class_indices[emotion_index])
        confidences.append(confidence)
        times.append(end_time - start_time)
    
    return y_true, y_pred, confidences, times

def evaluate_rmn(images, labels, class_indices, image_ids):
    rmn = RMN()

    y_true = []
    y_pred = []
    confidences = []
    times = []
    
    for img, true_label, img_id in zip(images, labels, image_ids):
        start_time = time.time()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        detections = rmn.detect_emotion_for_single_frame(img_bgr)
        end_time = time.time()
        if detections:
            emotion = detections[0]['emo_label']
            emotion_index = dic_emotion[emotion]
            confidence = detections[0]['emo_proba']
            
            y_true.append(true_label)
            y_pred.append(class_indices[emotion_index])
            confidences.append(confidence)
            times.append(end_time - start_time)
    
    return y_true, y_pred, confidences, times

def main():

    dataset_path = "../data/test"  

    val_images, val_labels, class_indices, image_ids = load_images_from_folder(dataset_path)
    
    print(len(val_images))    
    
    # Avaliação com DeepFace
    y_true_deepface, y_pred_deepface, confidences_deepface, times_deepface = evaluate_deepface(val_images, val_labels, class_indices, image_ids)
    
    # Avaliação com RMN
    y_true_rmn, y_pred_rmn, confidences_rmn, times_rmn = evaluate_rmn(val_images, val_labels, class_indices, image_ids)

    # Preparar dados para salvar no CSV
    data = {
        "ImageID": image_ids,
        "TrueLabel": y_true_rmn,  # y_true_rmn e y_true_deepface são os mesmos
        "PredictedLabel_RMN": y_pred_rmn,
        "Confidence_RMN": confidences_rmn,
        "ProcessingTime_RMN": times_rmn,
        "PredictedLabel_DeepFace": y_pred_deepface,
        "Confidence_DeepFace": confidences_deepface,
        "ProcessingTime_DeepFace": times_deepface
    }

    df = pd.DataFrame(data)
    df.to_csv('resultados_modelos.csv', index=False)

if __name__ == "__main__":
    main()
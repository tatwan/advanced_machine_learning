import warnings
warnings.filterwarnings('ignore')

import numpy as np
import sys
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import logging
import os

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

from PIL import Image
import argparse
import json


def predict(image_path, model, n):
    img = Image.open(image_path)
    test_image = np.asarray(img)
    processed_img = process_image(test_image)
    processed_img = np.expand_dims(processed_img, axis=0)
    prediction = model.predict(processed_img)
    idx = prediction[0].argsort()[-n:][::-1]
    labels = [str(i) for i in idx]
    probs = (prediction[0][idx])
    
    return probs, labels


def process_image(image):
    ts_img = tf.convert_to_tensor(image)
    ts_img = tf.image.resize(ts_img, (224,224))
    ts_img /= 255
    return ts_img.numpy()


def load_model(model, image_path, k=5):
    reloaded_model = tf.keras.models.load_model(model, custom_objects={'KerasLayer':hub.KerasLayer})
    #print(reloaded_model.summary())
    print("Model Loaded ....")
    return predict(image_path, reloaded_model, k)
  
#     return reloaded_model

def check_parameters(model, image_path):
        if os.path.exists(model):
            if os.path.exists(image_path):
                return True
            else:
                print("can't find image")
        else:
            print("can't find model")
        return False

def check_gpu():
    if tf.test.is_gpu_available():
        print("Running on GPU")
        return True
    else:
        print("GPU Not dedicted, please enable GPU")
    return False
        


if __name__=='__main__':
    if check_gpu():
        
        parser = argparse.ArgumentParser()
        parser.add_argument("image_path", help="Path to the image file")
        parser.add_argument("model", help="Predictive Model")
        parser.add_argument("--top_k", help="Return the top K most likely classes", action="store", type=int)
        parser.add_argument("--category_names", help="Path to a JSON file mapping labels to flower names", action="store")
        args = parser.parse_args()
        print(args)
        image_path, model = args.image_path, args.model
    #             print(class_names)
        try:
            if check_parameters(model, image_path):
                print("files found .... ")
                if args.top_k:
                       k = args.top_k
                       probs, classes = load_model(model, image_path, k)
                else:
                    print('=======here=======')
                    probs, classes = load_model(model, image_path)
            else:
                print("files not found, check and try again")
        except:
            print("Check the model path or filename")
    #     reloaded_model = load_model(model)
        if args.category_names:
            with open(args.category_names) as f:
                class_names = json.load(f)
                guessed_classes = [class_names[str(int(i)+1)] for i in classes ]
                print('=======================')
                print('class names:', guessed_classes)
                print('probabilities:',probs)
                print('classes:',classes)
                print('=======================')
        else:
            print('=======================')
            print('probabilities:',probs)
            print('classes:',classes)
            print('=======================')
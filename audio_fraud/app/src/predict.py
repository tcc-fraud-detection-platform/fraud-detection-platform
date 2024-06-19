import pandas as pd
import numpy as np
import librosa
import os
from tensorflow.keras.models import load_model
import librosa.display, os
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img

class_names = [1, 0]

model = tf.keras.models.load_model('src/saved_model/model')

class Predict:
    @staticmethod
    def create_spectrogram(sound):
        audio_file = os.path.join('src/audio_files/', sound)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        y, sr = librosa.load(audio_file)
        ms = librosa.feature.melspectrogram(y=y, sr=sr)
        log_ms = librosa.power_to_db(ms, ref=np.max)
        librosa.display.specshow(log_ms, sr=sr)
        # st.pyplot(fig)
        plt.savefig('melspectrogram.png')
        image_data = load_img('melspectrogram.png',target_size=(224,224))
        plt.close(fig)
        del fig
        return(image_data)

    @staticmethod
    def predictions(image_data,model):
        img_array = np.array(image_data)
        img_array1 = img_array / 255
        img_batch = np.expand_dims(img_array1, axis=0)

        prediction = model.predict(img_batch)
        class_label = np.argmax(prediction)
        return class_label,prediction

    @staticmethod
    def run_predict(audio_path):
        sound = audio_path

        spec = Predict.create_spectrogram(sound)

        class_label, prediction = Predict.predictions(spec,model)

        print("#### " + audio_path)
        # print("#### The uploaded audio file is " + class_names[class_label])
        # print(class_names[class_label], prediction)
        return class_names[class_label], prediction[0][class_label]

    @staticmethod
    def test(test):
        print(test)
        pass
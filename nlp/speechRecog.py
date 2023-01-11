import librosa
import IPython.display as ipd
import  numpy as np
from keras.models import load_model
import sounddevice as sd
import soundfile as sf
import os

train_audio_path = '../data'
classes = os.listdir(train_audio_path)
model = load_model('best_model1.hdf5')

def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]

samplerate = 16000
duration = 1
filename = 'recording/fg6.wav'
print('start')
mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
                channels=1, blocking=True)
print("end")
sd.wait()
sf.write(filename, mydata, samplerate)

samples , sample_rate = librosa.load( filename, sr= 16000)
samples = librosa.resample(samples, sample_rate , 8000)
ipd.Audio(samples , rate=8000)
print(predict(samples))
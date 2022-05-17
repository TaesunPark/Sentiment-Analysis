import sklearn
from flask import Flask
import joblib
import librosa
import numpy as np
import os
import test
import question

app = Flask(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 모든 음성파일의 길이가 같도록 후위에 padding 처리
pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/audio_sentiment')
def audio_test():  # put application's code here
    return load_audio("./test/happy.wav")


@app.route('/analysis')
def analyze_self_introduction():
    text = "리눅스의 기본명령어를 이해 합니다.\nAWS EC2 서버를 호스팅 하여 소프트웨어 설치 및 config파일을 수정 및 서비스를 구동 할 수 있습니다.\n도커를 이용하여 Flask, Spring-boot 를 배포한 경험이 있습니다.\nNginx를 이용하여 포트 포워딩 및 로드 밸런싱, ssl 인증서를 적용한 경험이 있습니다.\n인프라를 구축하고 무중단 배포한 경험이 있습니다."
    result = question.get_keyword_summarizer(text)
    return str(question.chat(result))

if __name__ == '__main__':
    app.run()


def load_audio(audio_name):
    audio, sr = librosa.load(audio_name)
    mfcc = librosa.feature.mfcc(audio, sr=16000, n_mfcc=100, n_fft=400, hop_length=160)
    mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
    # 모든 음성파일의 길이가 같도록 후위에 padding 처리
    padded_mfcc = pad2d(mfcc, 300)
    padded_mfcc = np.expand_dims(padded_mfcc, 0)
    # 파일로 저장된 모델 불러와서 예측
    clf_from_joblib = joblib.load('model/mfcc.pkl')
    result = clf_from_joblib.predict(padded_mfcc)
    if result[0][0] >= result[0][1]:
        return "긍정"
    else:
        return "부정"

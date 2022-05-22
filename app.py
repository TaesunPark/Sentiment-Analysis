import sklearn
from flask import Flask
from flask import request
import joblib
import librosa
import numpy as np
import os
import base64
import tensorflow
import json
import test
import question
import sentiment_text
import produce_wordcloud

app = Flask(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 모든 음성파일의 길이가 같도록 후위에 padding 처리
pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/analysis')
def analyze_self_introduction():
    text = "Android로 직접 화면 구성을 하고, MVP 디자인 패턴을 활용해 앱을 만들어봤습니다.시뮬레이터를 통한 테스트를 진행하였고 출시를 위한 프로세스를 이해하고 있습니다.html, css, js를 사용해 스마트 워치 기반 Tizen 웹 어플리케이션을 만들어봤습니다."
    file_dir = "./data/test.csv"
    result = question.get_keyword_summarizer(text, "text")
    return str(question.chat(result, file_dir, "text"))


@app.route('/wordcloud')
def get_wordcloud():
    text = "안녕하세요 저는 박태순이고요 아니 그 그렇지만 그런 방식을 좋아합니다."
    file_dir = "./data/qs.csv"
    first_list = question.get_keyword_summarizer(text, "wordcloud")
    result = question.chat(first_list, file_dir, "wordcloud")
    print(result)
    tags = produce_wordcloud.get_wordcloud_list(result)
    print(tags)
    produce_wordcloud.create_wordcloud(tags)
    return "생성 성공"

@app.route('/bertmodel')
def get_bert_model():
    return sentiment_text.get_bert_model("상벽아 안녕")

@app.route('/audio-sentiment', methods=['POST'])
def audio_test():  # put application's code here

    # return load_audio("./test/happy.wav")
    base64data = request.get_json().get('base64data').split(',')[1]

    try:
        decode_string = base64.b64decode(base64data)

        wav_file = open("./test/data1.wav", "wb")
        wav_file.write(decode_string)

        return load_audio("./test/data1.wav")

    except Exception as e:
        print(str(e))
        return "error-1"

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

    l_list = {'p': str(int(result[0][0] * 100)), 'n': str(int(result[0][1] * 100))}
    jsonStr = json.dumps(l_list)
    return jsonStr

if __name__ == '__main__':
        app.run(host='0.0.0.0', port='5000', debug=True)






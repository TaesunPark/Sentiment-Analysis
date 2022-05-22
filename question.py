from textrank.textrank import KeywordSummarizer
from pororo import Pororo
import pandas as pd
from sentence_transformers import util
import torch
import numpy as np
from tqdm import tqdm
from setting import komoran1


def get_komoran():
    return


def komoran_tokenizer(sent):
    words = komoran1.pos(sent, join=True)
    words = [w for w in words if ('/NN' in w or '/NP' in w or '/SL' in w)]
    return words


def get_keyword_summarizer(text):
    text = text.split("\n")
    results = []
    keyword_summarizer = KeywordSummarizer(tokenize=komoran_tokenizer, min_count=1, min_cooccurrence=2)
    texts = keyword_summarizer.summarize(text, topk=30)
    for i in texts:
        results.append(i[0].split('/')[0])
    return results


def get_csv_data(dir):
    Chatbot_Data = pd.read_csv(dir)
    return Chatbot_Data


def chat(results, sent="0"):
    result = []
    tqdm.pandas()
    Chatbot_Data = get_csv_data("./data/test.csv")
    sTe = Pororo(task="sentence_embedding", lang="ko")
    Chatbot_Data['EmbVector'] = Chatbot_Data['Q'].progress_map(lambda x: sTe(x))
    EmbData = torch.tensor(Chatbot_Data['EmbVector'].tolist())
    for i in results:
        a = ""
        # Pororo Sentense Embedding으로 텍스트 유사도를 구합니다.
        q = sTe(i)
        # 질문을 Tensor로 바꿉니다.
        q = torch.tensor(q)
        # 코사인 유사도
        cos_sim = util.pytorch_cos_sim(q, EmbData)
        # 유사도가 가장 비슷한 질문 인덱스를 구합니다.
        best_sim_idx = np.where(cos_sim >= 0.73)
        for i in best_sim_idx[1]:
            result.append(Chatbot_Data['A'][i])
    return set(result)

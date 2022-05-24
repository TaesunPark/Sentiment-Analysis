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


def komoran_tokenizer_wordcloud(sent):
    words = komoran1.pos(sent, join=True)
    words = [w for w in words if ('/NN' in w or '/NP' in w or '/SL' in w or '/VV' in w or '/MM' in w or '/MA' in w)]
    return words


def get_keyword_summarizer(text, type):
    text = text.split("\n")
    results = []

    if type == 2:
        keyword_summarizer = KeywordSummarizer(tokenize=komoran_tokenizer_wordcloud, min_count=1, min_cooccurrence=2)
    else:
        keyword_summarizer = KeywordSummarizer(tokenize=komoran_tokenizer, min_count=1, min_cooccurrence=2)

    texts = keyword_summarizer.summarize(text, topk=30)

    for i in texts:
        results.append(i[0].split('/')[0])
    return results


def get_csv_data(dir):
    Chatbot_Data = pd.read_csv(dir)
    return Chatbot_Data


def chat(result_list, dir, type, sent="0"):
    print(result_list)
    qa_percent = 0
    if type == 2:
        qa_percent = 0.75
    else:
        qa_percent = 0.73

    print(qa_percent)


    result = []
    tqdm.pandas()
    Chatbot_Data = get_csv_data(dir)
    sTe = Pororo(task="sentence_embedding", lang="ko")
    Chatbot_Data['EmbVector'] = Chatbot_Data['Q'].progress_map(lambda x: sTe(x))
    EmbData = torch.tensor(Chatbot_Data['EmbVector'].tolist())

    for i in result_list:
        a = ""
        # Pororo Sentense Embedding으로 텍스트 유사도를 구합니다.
        q = sTe(i)
        # 질문을 Tensor로 바꿉니다.
        q = torch.tensor(q)
        # 코사인 유사도
        cos_sim = util.pytorch_cos_sim(q, EmbData)
        # 유사도가 가장 비슷한 질문 인덱스를 구합니다.
        best_sim_idx = np.where(cos_sim >= qa_percent)

        for j in best_sim_idx[1]:
            result.append(Chatbot_Data['A'][j])

    if type == 2:
        return result

    return set(result)
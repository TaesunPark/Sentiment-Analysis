import torch
from konlpy.tag import Komoran
from kobert.pytorch_kobert import get_pytorch_kobert_model
from gluonnlp.data import SentencepieceTokenizer
from kobert import get_tokenizer

tok_path = get_tokenizer()
sp = SentencepieceTokenizer(tok_path)
bertmodel, vocab = get_pytorch_kobert_model()

device = "cuda" if torch.cuda.is_available() else "cpu"
komoran1 = Komoran()
import torch
from torch import nn
import numpy as np
import gluonnlp as nlp
from setting import bertmodel
from setting import vocab
from setting import device
from setting import sp
from torch.utils.data import Dataset, DataLoader

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return (len(self.labels))


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=7,  ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device), return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

    def predict(predict_sentence, model):
        global out_2
        data = [predict_sentence, '0']
        dataset_another = [data]

        another_test = BERTDataset(dataset_another, 0, 1, sp, vocab, 64, True, False)
        test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=64, num_workers=5)
        model.eval()

        #for문 요소들 받는 과정이 느림
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out_2 = model(token_ids, valid_length, segment_ids)
            test_eval = []
            for i in out_2:
                logits = i
                logits = logits.detach().cpu().numpy()

                if np.argmax(logits) == 0:
                    test_eval.append("부정적")
                elif np.argmax(logits) == 1:
                    test_eval.append("부정적")
                elif np.argmax(logits) == 2:
                    test_eval.append("부정적")
                elif np.argmax(logits) == 3:
                    test_eval.append("부정적")
                elif np.argmax(logits) == 4:
                    test_eval.append("중립이")
                elif np.argmax(logits) == 5:
                    test_eval.append("긍정적")
                elif np.argmax(logits) == 6:
                    test_eval.append("부정적")

            return test_eval[0]


def get_bert_model(text):
    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    model_state_dict = torch.load("./model/model_state_dict.pt", map_location=device)
    model.load_state_dict(model_state_dict)
    return BERTClassifier.predict(text, model)


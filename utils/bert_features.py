# 提取bert特征

#import numpy as np
#from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array

config_path = '../../nlp/nlp_model/chinese_bert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../nlp/nlp_model/chinese_bert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../nlp/nlp_model/chinese_bert_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True, 
    token_start=None, token_end=None) # 不要起始标记 [CLS] [SEP]
model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重


def bert_features(s):
    token_ids, segment_ids = tokenizer.encode(s)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    return model.predict([token_ids, segment_ids])

def avg_featrues(s):
    features = bert_features(s)
    return features.sum(axis=1) / features.shape[1]

def sum_featrues(s):
    features = bert_features(s)
    return features.sum(axis=1)

if __name__ == '__main__':
    f = "语言模型"
    print(bert_features(f).shape)
    print(sum_featrues(f).shape)
    print(avg_featrues(f).shape)

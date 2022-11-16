#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.preprocessing import LabelEncoder
from transformers import DataCollatorWithPadding
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_excel('test_drive_data.xlsx')

model_path = 'bert_chinese/'
tokenizer = BertTokenizer.from_pretrained(model_path)
max_length = 64
batch_size = 32


class TestdriveDataset(Dataset):
    def __init__(self, df, max_length, tokenizer):
        self.text = df['评论内容'].values
        self.max_len = max_length
        self.tokenizer = tokenizer

    def __getitem__(self, ids):
        text = self.text[ids]
        inputs = self.tokenizer.encode_plus(text,
                                            truncation=True,
                                            add_special_tokens=True,
                                            max_length=self.max_len
                                            )

        data_dict = {'input_ids': inputs['input_ids'],
                     'attention_mask': inputs['attention_mask']
                     }

        return data_dict

    def __len__(self):
        return len(self.text)


collate_fn = DataCollatorWithPadding(tokenizer)

train_dataset = TestdriveDataset(df, max_length, tokenizer)
train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)


def mean_pool(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask

    return mean_embeddings


one_batch_embedding = []
all_embedding = []
bert = BertModel.from_pretrained(model_path)
for data in tqdm(train_loader):
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']

    out = bert(input_ids=input_ids,
               attention_mask=attention_mask)

    output = mean_pool(out.last_hidden_state, attention_mask)
    one_batch_embedding.extend(output)
all_embedding = torch.stack(one_batch_embedding)
all_embedding = all_embedding.cpu().detach().numpy()

# 选择合适的k值
d = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++')
    km.fit(all_embedding)
    d.append(km.inertia_)

plt.plot(range(1, 11), d, marker='o')
plt.xlabel('number of clusters')
plt.ylabel('distortions')
plt.show()

km = KMeans(n_clusters=5)
result = km.fit_predict(all_embedding)
df['prediction'] = result

df.to_excel('kmeans_test_V2.xlsx')

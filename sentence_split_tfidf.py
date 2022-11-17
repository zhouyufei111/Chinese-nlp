#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import jieba
from collections import Counter
import pickle
import re

file_path = 'test_drive training set.xlsx'

df = pd.read_excel('/s/yzhou180/HSMS/TrainingSet/test_drive/' + file_path)

stop_words = pd.read_csv('cn_stopwords.txt', index_col=False, encoding='UTF-8', names=['stopwords'])
stop_words = stop_words.stopwords.values.tolist()
stop_words.append(',')
stop_words.append(' ')
stop_words.append('[')
stop_words.append(']')
stop_words.append("'")


class TFIDF():
    def __init__(self, df, text, by, aspect_num=19):
        self.df = df
        self.text = text
        self.by = by
        self.aspect_num = aspect_num
        self.wc_list, self.wc_all = self.word_count()
        for n in range(len(self.wc_list)):
            self.filter_key(n)

        self.wc_dict = dict(zip(self.df.Aspect.unique(), self.wc_list))
        self.tf()
        self.idf_dict = self.idf()
        self.tfidf = self.get_tfidf()

    # 按类别统计词频和总词数，返回一个列表和一个字典。
    # 第一个返回值，列表中包含多个字典，字典数=类别数，每个字典中包含这个分类中的词和对应的词频
    # 第二个返回值，字典中包含每个分类中总词数
    def word_count(self):
        wc_word = []
        wc_all = {}
        for x in self.df[self.by].unique().tolist():
            string_all = ''
            df_asp = self.df.loc[self.df[self.by] == x, :]
            for i in df_asp[self.text].tolist():
                string_all += str(i)
            seg = jieba.lcut(string_all)
            seg = [x for x in seg if x not in stop_words]
            wc = Counter(seg)
            wc_word.append(wc)
            wc_all[x] = len(seg)
        return wc_word, wc_all

    # 如果一个词在这类出现小于3次，认为它不包含这个类别中的信息，删除
    def filter_key(self, n):
        remove_key = []
        for i in self.wc_list[n].keys():
            if self.wc_list[n][i] < 3:
                remove_key.append(i)
        for i in remove_key:
            self.wc_list[n].pop(i)

    # 计算tf：单个词出现的个数/总词数
    # 返回一个字典，类别名对应这个类中每个词的词频
    def tf(self):
        # get term frequency
        for i in self.wc_dict.keys():
            for key in self.wc_dict[i].keys():
                self.wc_dict[i][key] = self.wc_dict[i][key] / self.wc_all[i]

    # 计算idf：log(总文档数/(出现这个词的文档数+1))
    # 返回idf字典，每个词的对应idf值
    def idf(self):
        # 词表，所有出现的词去重
        word_list = []
        for i in self.wc_list:
            word_list.extend(i.keys())
        word_list = list(set(word_list))

        # 每个词的idf
        idf_dict = {}
        for i in word_list:
            w = 0
            for aspect in self.wc_list:
                if i in aspect.keys():
                    w += 1
            idf_dict[i] = w

        for i in idf_dict.keys():
            idf_dict[i] = np.log2(self.aspect_num / (idf_dict[i] + 1))

        return idf_dict

    # 计算tfidf
    def get_tfidf(self):
        tfidf = {}
        for i in self.wc_dict.keys():
            tfidf[i] = {}
            for key in self.wc_dict[i].keys():
                tfidf[i][key] = self.wc_dict[i][key] * self.idf_dict[key]

        return tfidf


class cutSentence():
    def __init__(self, tfidf):
        self.tfidf = tfidf

    # 把句子切分到最小单位

    def cut_sentences(self, content):
        # 结束符号，包含中文和英文的
        end_flag = ['?', '!', '.', '？', '！', '。', '…', '，', ',', ' ', ';', '；']

        content_len = len(content)
        sentences = []
        tmp_char = ''
        for idx, char in enumerate(content):
            # 拼接字符
            tmp_char += char

            # 判断是否已经到了最后一位
            if (idx + 1) == content_len:
                sentences.append(tmp_char)
                break

            # 判断此字符是否为结束符号
            if char in end_flag:
                # 再判断下一个字符是否为结束符号，如果不是结束符号，则切分句子
                next_idx = idx + 1
                if not content[next_idx] in end_flag:
                    sentences.append(tmp_char)
                    tmp_char = ''

        return sentences

    # 通过tfidf把句子分类，如果一句话中有一个分类的两个关键词，两个词的tfidf相加，输出tfidf最大的类
    def get_aspect(self):
        key_dict = {}
        for key in self.tfidf.keys():
            for i in self.tfidf[key]:
                if i in self.sentences:
                    if key not in key_dict:
                        key_dict[key] = self.tfidf[key][i]
                    else:
                        key_dict[key] += self.tfidf[key][i]

        return max(key_dict.items(), key=lambda x: x[1])

    # 根据规则切分句子
    def get_cut_sentence(self, sentences):
        # 如果文本的格式类似'1，2，'，直接按这个格式切句
        sentence_ = re.split(r'\d,\D|\d、\D|\d\.\D', sentences)
        if len(sentence_) > 1:
            try:
                sentence_.remove('')
            except:
                pass
            return sentence_
        sentence = self.cut_sentences(sentences)

        ite = len(sentence)
        # 每次完成合并后，跳出内部的loop，从头开始遍历
        for _ in range(ite):
            # 从第二句开始遍历
            for i in range(1, len(sentence)):
                # 长度小于6的句子合并
                if (len(sentence[i]) < 6) & (len(sentence[i - 1]) < 6):
                    sentence[i - 1] = sentence[i - 1] + sentence[i]
                    sentence.remove(sentence[i])
                    break

                # 获取当前句和前一句的分类和tfidf，如果前一句没有词库中的词，合并
                try:
                    aspect, score = self.get_aspect(sentence[i])[0], get_aspect(sentence[i])[1]
                    last_aspect, last_score = self.get_aspect(sentence[i - 1])[0], get_aspect(sentence[i - 1])[1]
                except:
                    sentence[i - 1] = sentence[i - 1] + sentence[i]
                    sentence.remove(sentence[i])
                    break

                # 如果这一句的分类和前一句的分类一样，合并
                if (aspect == last_aspect):
                    sentence[i - 1] = sentence[i - 1] + sentence[i]
                    sentence.remove(sentence[i])
                    break

                # 如果这一句或者上一句的tfidf小于0.1，认为这句没有明显分类特征，合并
                elif (score < 0.1) | (last_score < 0.1):
                    sentence[i - 1] = sentence[i - 1] + sentence[i]
                    # print(sentence[i-1])
                    sentence.remove(sentence[i])
                    break

        return sentence

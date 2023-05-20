from pathlib import Path
import os
import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import TomekLinks
import sys
sys.path.append('../../../bert-utils')
from extract_feature import BertVector

dimens = ['空间', '动力', '操控', '能耗', '舒适性', '外观', '内饰', '性价比', '配置', '续航', '安全性', '环保', '质量与可靠性', '充电', '服务', '品牌', '智能驾驶', '其它', '总的来说']


def read_data_from_excel():
    type_list = ['宝马1系', '宝马2系', '宝马3系（1）','宝马3系（2）', '宝马3系（3）', '宝马3系（4）','宝马3系（5）', '宝马3系（6）', '宝马3系（7）','宝马3系（8）', '宝马3系（9）', '宝马3系（10)','宝马3系（11）', '宝马3系（12）', '宝马4系','宝马5系','宝马X1（1）', '宝马X1（2）', '宝马X1(5)', '宝马X2', '宝马X3']
    df = pd.read_excel('./my_data/bmw_all.xlsx', sheet_name=type_list, dtype=str)
    contents, labels = [], []
    for type in type_list:
        df_drop = df[type].dropna(subset=[dimens[-1], '具体评价'])
        contents += df_drop["具体评价"].tolist()
        labels += df_drop[dimens[-1]].tolist()
    real_contents = []
    encode_labels = []
    for idx, label in enumerate(labels):
        if label == '1':
            # 积极的
            encode_labels.append('POS')
            real_contents.append(contents[idx])
        elif label == '0':
            # 中性的
            encode_labels.append('NEU')
            real_contents.append(contents[idx])
        elif label == '-1':
            # 消极的
            encode_labels.append('NEG')
            real_contents.append(contents[idx])
    train_words, test_words, train_labels, test_labels = train_test_split(real_contents, encode_labels, test_size=0.2)
    return train_words, test_words, train_labels, test_labels

def save_words_to_text(mode, save_words):
    for word in save_words:
        if word.strip() != '':
            cut_word = ' '.join(jieba.cut(word.strip(), cut_all=False, HMM=True))
            with Path('{}.words.txt'.format(mode)).open('a', encoding='utf-8') as g:
                g.write('{}\n'.format(cut_word))

def save_labels_to_text(mode, save_labels):
    for word in save_labels:
        if word.strip() != '':
            with Path('{}.labels.txt'.format(mode)).open('a', encoding='utf-8') as g:
                g.write('{}\n'.format(word))

def balance_data_by_tomek_links(X_train, y_train):
    tl = TomekLinks(sampling_strategy='not minority')
    X_train_resampled, y_train_resampled = tl.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

if __name__ == '__main__':
    # train_words, test_words, train_labels, test_labels = read_data_from_excel()
    # print(len(train_words), len(train_labels))
    # print(len(test_words), len(test_labels))
    bv = BertVector()
    bert_vec = bv.encode(["测试一下"])
    print(str(bert_vec))
    # X_train_resampled, y_train_resampled = balance_data_by_tomek_links(bert_vec, train_labels)
    # print(X_train_resampled, y_train_resampled)
    # save_words_to_text('train', train_words)
    # save_words_to_text('eval', test_words)
    # save_labels_to_text('train', train_labels)
    # save_labels_to_text('eval', test_labels)

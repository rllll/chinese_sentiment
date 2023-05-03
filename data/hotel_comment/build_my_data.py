from pathlib import Path
import os
import jieba
import pandas as pd
from sklearn.model_selection import train_test_split


def build_data_file(directory, samples_path, label, mode_str):
    for sample_path in samples_path:
        with Path('{}/{}'.format(directory, sample_path)).open() as f:
            words = [' '.join(jieba.cut(line.strip(), cut_all=False, HMM=True)) for line in f if line.strip() != '']
            with Path('{}.words.txt'.format(mode_str)).open('a') as g:
                g.write('{}\n'.format(' '.join(words)))
            with Path('{}.labels.txt'.format(mode_str)).open('a') as h:
                h.write('{}\n'.format(label))

def read_data_from_excel():
    train_data_file, test_data_file = './my_data/train.csv', './my_data/test.csv'
    if os.path.exists(train_data_file) and os.path.exists(test_data_file):
        return
    # type_list = ['宝马1系', '宝马2系', '宝马3系（1）','宝马3系（2）', '宝马3系（3）', '宝马3系（4）','宝马3系（5）', '宝马3系（6）', '宝马3系（7）','宝马3系（8）', '宝马3系（9）', '宝马3系（10)','宝马3系（11）', '宝马3系（12）', '宝马4系','宝马5系','宝马X1（1）', '宝马X1（2）', '宝马X1(5)', '宝马X2', '宝马X3']
    type_list = ['宝马3系（3）']
    df = pd.read_excel('./my_data/bmw_all.xlsx', sheet_name=type_list, usecols=[2, 21], dtype=str)
    contents, labels = [], []
    for type in type_list:
        df_drop = df[type].dropna(subset=['总的来说'])
        contents += df_drop["具体评价"].tolist()
        labels += df_drop['总的来说'].tolist()
    encode_labels = []
    for label in labels:
        if label == '1':
            # 积极的
            encode_labels.append('POS')
        elif label == '0':
            # 中性的
            encode_labels.append('NEU')
        elif label == '-1':
            # 消极的
            encode_labels.append('NEG')
    print(encode_labels)
    train_words, test_words, train_labels, test_labels = train_test_split(contents, encode_labels, test_size=0.2)
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

if __name__ == '__main__':
    train_words, test_words, train_labels, test_labels = read_data_from_excel()
    print(len(train_words), len(train_labels))
    print(len(test_words), len(test_labels))
    save_words_to_text('train', train_words)
    save_words_to_text('eval', test_words)
    save_labels_to_text('train', train_labels)
    save_labels_to_text('eval', test_labels)
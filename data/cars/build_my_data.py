from pathlib import Path
import json
import jieba
import pandas as pd
from sklearn.model_selection import train_test_split


transform_label = {
    "1": "POS",
    "0": "NEU",
    "-1": "NEG"
}

def read_data_from_excel():
    case_type = ['宝马1系', '宝马2系', '宝马3系（1）','宝马3系（2）', '宝马3系（3）', '宝马3系（4）','宝马3系（5）', '宝马3系（6）', '宝马3系（7）','宝马3系（8）', '宝马3系（9）', '宝马3系（10)','宝马3系（11）', '宝马3系（12）', '宝马4系','宝马5系','宝马X1（1）', '宝马X1（2）', '宝马X1(5)', '宝马X2', '宝马X3']
    df = pd.read_excel('./my_data/bmw_all.xlsx', sheet_name=case_type, dtype=str)
    all_data = df[case_type[0]]
    for idx in range(1, len(case_type)):
        all_data = pd.concat([all_data, df[case_type[idx]]], ignore_index=True)
    
    all_data = all_data.dropna(subset=["具体评价"])

    dimens = ['空间', '动力', '操控', '能耗', '舒适性', '外观', '内饰', '性价比', '配置', '续航', '安全性', '环保', '质量与可靠性', '充电', '服务', '品牌', '智能驾驶', '其它', '总的来说']

    detail_text = {}

    for dim in dimens:
        drop_df = all_data.dropna(subset=[dim])
        contents = drop_df["具体评价"].tolist()
        labels = drop_df[dim].tolist()

        valid_contents = []
        valid_labels = []

        for idx, text in enumerate(contents):
            if labels[idx] != '1' and labels[idx] != '0' and labels[idx] != '-1':
                continue
            valid_contents.append(text)
            valid_labels.append(transform_label[labels[idx]])

        train_words, test_words, train_labels, test_labels = train_test_split(valid_contents, valid_labels, test_size=0.2)


        cut_train_words = []
        cut_test_words = []
        
        for word in train_words:
            if word.strip() != '':
                cut_word = ' '.join(jieba.cut(word.strip(), cut_all=False, HMM=True))
                cut_train_words.append(cut_word)
        for word in test_words:
            if word.strip() != '':
                cut_word = ' '.join(jieba.cut(word.strip(), cut_all=False, HMM=True))
                cut_test_words.append(cut_word)

        detail_text[dim] = {
            "train_words": cut_train_words,
            "test_words": cut_test_words,
            "train_labels": train_labels,
            "test_labels": test_labels
        }
    
    detail_text = json.dumps(detail_text, indent=2, ensure_ascii=False)

    with open('data.json', 'w', encoding='utf-8-sig') as f:
        f.write(detail_text)

if __name__ == '__main__':
    read_data_from_excel()

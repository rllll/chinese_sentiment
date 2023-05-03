from pathlib import Path
import numpy as np
import json

def load_vocab_data():
    with open('./vocab.json', 'r', encoding='utf-8-sig') as f:
        vocab_data = json.loads(f.read())
    return vocab_data


if __name__ == '__main__':
    vocab_data = load_vocab_data()

    Path('w2v_dir').mkdir(exist_ok=True)

    dimens = ['空间', '动力', '操控', '能耗', '舒适性', '外观', '内饰', '性价比', '配置', '续航', '安全性', '环保', '质量与可靠性', '充电', '服务', '品牌', '智能驾驶', '其它', '总的来说']
    dim_tag = {}
    for idx, dim in enumerate(dimens):
        dim_tag[dim] = 'i' + str(idx)

    for dim in dimens:
        vocab_words = vocab_data[dim]["words"]
        word_to_idx = {line.strip(): idx for idx, line in enumerate(vocab_words)}
        word_to_found = {line.strip(): False for line in vocab_words}

        size_vocab = len(word_to_idx)

        embeddings = np.zeros((size_vocab, 300))

        found = 0
        print('Reading W2V file (may take a while)')
        with Path('../sgns/sgns.zhihu.bigram').open(encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if line_idx % 100000 == 0:
                    print('- At line {}'.format(line_idx))
                line = line.strip().split()
                if len(line) != 300 + 1:
                    continue
                word = line[0]
                embedding = line[1:]
                if (word in word_to_idx) and (not word_to_found[word]):
                    word_to_found[word] = True
                    found += 1
                    word_idx = word_to_idx[word]
                    embeddings[word_idx] = embedding
        print('- done. Found {} vectors for {} words'.format(found, size_vocab))

        # 保存 np.array
        np.savez_compressed('./w2v_dir/{}_w2v.npz'.format(dim_tag[dim]), embeddings=embeddings)

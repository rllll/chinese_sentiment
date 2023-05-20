from pathlib import Path
import numpy as np
from imblearn.under_sampling import TomekLinks

def balance_data_by_tomek_links(X_train, y_train):
    tl = TomekLinks(sampling_strategy='not minority')
    X_train_resampled, y_train_resampled = tl.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

if __name__ == '__main__':
    with Path('vocab.words.txt').open(encoding='utf-8') as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    with Path('vocab.words.txt').open(encoding='utf-8') as f:
        word_to_found = {line.strip(): False for line in f}
    with Path('vocab.labels.txt').open(encoding='utf-8') as f:
        labels = [line.strip() for line in f]

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

    print(embeddings)
    X_train_resampled, y_train_resampled = balance_data_by_tomek_links(embeddings, labels)
    print(X_train_resampled, y_train_resampled)
    # 保存 np.array
    # np.savez_compressed('w2v.npz', embeddings=embeddings)

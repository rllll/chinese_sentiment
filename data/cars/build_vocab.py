from collections import Counter
from pathlib import Path
import json

MIN_COUNT = 1

def load_data():
    with open('./data.json', 'r', encoding='utf-8-sig') as f:
        res_data = json.loads(f.read())
    return res_data

if __name__ == '__main__':

    data = load_data()

    Path('vocab_dir').mkdir(exist_ok=True)

    dimens = ['空间', '动力', '操控', '能耗', '舒适性', '外观', '内饰', '性价比', '配置', '续航', '安全性', '环保', '质量与可靠性', '充电', '服务', '品牌', '智能驾驶', '其它', '总的来说']
    dim_tag = {}
    for idx, dim in enumerate(dimens):
        dim_tag[dim] = 'i' + str(idx)

    vocab_cache = {}

    for dim in dimens:
        print('Build vocab words')
        counter_words = Counter()
        all_words = data[dim]['train_words'] + data[dim]['test_words']
        for w in all_words:
             counter_words.update(str(w).strip().split())
        vocab_words = {w for w, c in counter_words.items() if c >= MIN_COUNT}
        print('Done. Kept {} out of {}'.format(len(vocab_words), len(counter_words)))
        
        print('Build labels')
        doc_tags = set()
        train_labels = data[dim]['train_labels']
        for lb in train_labels:
            doc_tags.add(str(lb).strip())
        print('- done. Found {} labels.'.format(len(doc_tags)))

        vocab_cache[dim] = {
            "words": sorted(list(vocab_words)),
            "labels": sorted(list(doc_tags))
        }
        with open('./vocab_dir/{}_words.txt'.format(dim_tag[dim]), 'w', encoding='utf-8-sig') as f:
            for v in sorted(list(vocab_words)):
                f.write('{}\n'.format(v))
        with open('./vocab_dir/{}_labels.txt'.format(dim_tag[dim]), 'w', encoding='utf-8-sig') as f:
            for d in sorted(list(doc_tags)):
                f.write('{}\n'.format(d))
    
    vocab_cache = json.dumps(vocab_cache, indent=2, ensure_ascii=False)

    with open('vocab.json', 'w', encoding='utf-8-sig') as f:
        f.write(vocab_cache)

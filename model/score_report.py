import argparse
from sklearn import metrics
from pathlib import Path
import json

parser = argparse.ArgumentParser()
parser.add_argument('-f', help='specify the path of the results file')
parser.add_argument('-o', help='specify the path of the output results file')
args = parser.parse_args()

dimens = ['空间', '动力', '操控', '能耗', '舒适性', '外观', '内饰', '性价比', '配置', '续航', '安全性', '环保', '质量与可靠性', '充电', '服务', '品牌', '智能驾驶', '其它', '总的来说']


if __name__ == '__main__':
    label_true = []
    label_pred = []
    target_names = []
    origin = {}
    with Path(args.o).open('r', encoding='utf-8-sig') as f:
        origin = json.loads(f.read())
    with Path(args.f).open('r', encoding='utf-8') as f:
        for line in f:
            tag_name = line.strip().split()[0]
            if tag_name not in target_names:
                target_names.append(tag_name)
            label_true.append(tag_name)
            label_pred.append(line.strip().split()[1])
    res = metrics.classification_report(y_pred=label_pred, y_true=label_true, output_dict=True, target_names=target_names)
    origin[dimens[0]] = res
    with open(args.o, 'w', encoding='utf-8-sig') as f:
        f.write(json.dumps(origin, ensure_ascii=False, indent=2))


import numpy as np
from fairseq.models.roberta import RobertaModel
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", default="/fsx/zhuofeng/checkpoint/", help="the finetuned directory")
parser.add_argument("-o", "--output_file", default="", help="the result file")
parser.add_argument("-d", "--data_file", default="data/STS-B-dev.tsv", help="the data")

args = parser.parse_args()

gold, pred = [], []

preds = [[] for i in range(5)]
for number in range(5):
    with open(args.input_dir + str(number + 1) + '/' + str(number + 1), 'r') as fin:
        for line in fin.readlines():
            preds[number].append(float(line.strip()))
        
with open(args.data_file, encoding="utf-8") as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        target = float(tokens[-1])

        prediction_label = 0.0
        for number in range(5):
            prediction_label += preds[number][index]
        prediction_label /= 5

        gold.append(target)
        pred.append(prediction_label)

print('| Pearson: ', pearsonr(gold, pred)[0])
print('| Spearmanr: ', spearmanr(gold, pred)[0])
print('---------------------')
print('| bagging: ', str((pearsonr(gold, pred)[0] + spearmanr(gold, pred)[0]) / 2))

with open(args.output_file, 'a') as outf:
    outf.write('sts-b bagging' + '| Pearson: ' + str(pearsonr(gold, pred)) + ' | Spearmanr: ' + str(spearmanr(gold, pred)) + ' | avg: ' + str((pearsonr(gold, pred)[0] + spearmanr(gold, pred)[0]) / 2) + '\n')
    #count = 0
    #results = [0.0 for i in range(5)]
    #with open(args.output_file, 'r') as inf:
    #    for line in inf.readlines():
    #        results[count % 5] = float(line.split(':')[-1].strip())
    #        count += 1
    #outf.write('avg | Accuracy: ' + str(sum(results) / 5.0) + '\n')
    #print('| avg : ' + str(sum(results) / 5.0))
    #print('| std : ' + str(np.std(results)))
    #print('---------------------')

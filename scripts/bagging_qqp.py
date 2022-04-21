import numpy as np
from fairseq.models.roberta import RobertaModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", default="/fsx/zhuofeng/checkpoint/", help="the finetuned directory")
parser.add_argument("-o", "--output_file", default="", help="the result file")
parser.add_argument("-d", "--data_file", default="data/QQP-dev.tsv", help="the data")

args = parser.parse_args()

ncorrect, nsamples = 0, 0
tp, fn, fp, tn = 0.0, 0.0, 0.0, 0.0

pred = [[] for i in range(5)]
for number in range(5):
    with open(args.input_dir + str(number + 1) + '/' + str(number + 1), 'r') as fin:
        for line in fin.readlines():
            pred[number].append(int(line.strip()))
        
with open(args.data_file, encoding="utf-8") as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        target = tokens[5]

        prediction_label = 0.0
        for number in range(5):
            prediction_label += pred[number][index]
        prediction_label /= 5
        if prediction_label > 0.5:
            prediction_label = '1'
        else:
            prediction_label = '0'

        ncorrect += int(prediction_label == target)
        nsamples += 1
        if target == '1' and prediction_label == '1':
            tp += 1
        if target == '1' and prediction_label == '0':
            fn += 1
        if target == '0' and prediction_label == '1':
            fp += 1
        if target == '0' and prediction_label == '0':
            tn += 1

print('f1 = ' + str(tp / (tp + 0.5 * (fp + fn))))
print('| Accuracy: ', float(ncorrect)/float(nsamples))
print('---------------------')
print('| bagging: '+ str((tp / (tp + 0.5 * (fp + fn)) + float(ncorrect)/float(nsamples)) / 2))

count = 0
results = [0.0 for i in range(5)]
with open(args.output_file, 'r') as inf:
    for line in inf.readlines():
        if ':' in line:
            results[count % 5] = float(line.split(':')[-1].strip())
            count += 1

with open(args.output_file, 'a') as outf:
    outf.write('qqp bagging' + '| f1 = ' + str(tp / (tp + 0.5 * (fp + fn))) + ' | Accuracy: ' + str(float(ncorrect)/float(nsamples)) + ' | avg: '+ str((tp / (tp + 0.5 * (fp + fn)) + float(ncorrect)/float(nsamples)) / 2) + '\n')
    outf.write('avg | %.1f'%(sum(results) * 20.0) + '(%.1f'%(np.std(results)*100.0) + ')\n')
    #outf.write('avg | Accuracy: ' + str(sum(results) / 5.0) + '\n')
    #print('| avg : ' + str(sum(results) / 5.0))
    #print('| std : ' + str(np.std(results)))
    #print('---------------------')

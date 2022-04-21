import numpy as np
from fairseq.models.roberta import RobertaModel
from sklearn.metrics import matthews_corrcoef
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", default="/fsx/zhuofeng/checkpoint/", help="the finetuned directory")
parser.add_argument("-o", "--output_file", default="", help="the result file")
parser.add_argument("-d", "--data_file", default="data/CoLA-dev.tsv", help="the data")

args = parser.parse_args()

ncorrect, nsamples = 0, 0

gold, pred = [], []

preds = [[] for i in range(5)]
for number in range(5):
    with open(args.input_dir + str(number + 1) + '/' + str(number + 1), 'r') as fin:
        for line in fin.readlines():
            preds[number].append(float(line.strip()))
        
with open(args.data_file, encoding="utf-8") as fin:
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        target = int(tokens[1])

        prediction_label = 0.0
        for number in range(5):
            prediction_label += preds[number][index]
        prediction_label /= 5
        if prediction_label >= 0.5:
            prediction_label = 1
        else:
            prediction_label = 0

        gold.append(target)
        pred.append(prediction_label)

        ncorrect += int(prediction_label == target)
        nsamples += 1

print('---------------------')
print('| bagging: ', float(ncorrect)/float(nsamples))
#print('| bagging: ' + str(matthews_corrcoef(gold, pred)))

with open(args.output_file, 'w') as outf:
    #outf.write('cola ' + '| matthews_corrcoef: ' + str(matthews_corrcoef(gold, pred)) + '\n')
    outf.write('cola bagging' + '| accuracy: ' + str(float(ncorrect)/float(nsamples)) + '\n')
    #count = 0
    #results = [0.0 for i in range(5)]
    #with open(args.output_file, 'r') as inf:
    #    for line in inf.readlines():
    #        results[count % 5] = float(line.split(':')[-1].strip())
    #        count += 1
    ##outf.write('avg | matthews_corrcoef: ' + str(sum(results) / 5.0) + '\n')
    #outf.write('avg | accuracy: ' + str(sum(results) / 5.0) + '\n')
    #print('| avg : ' + str(sum(results) / 5.0))
    #print('| std : ' + str(np.std(results)))
    #print('---------------------')

import numpy as np
from fairseq.models.roberta import RobertaModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", default="/fsx/zhuofeng/checkpoint/", help="the finetuned directory")
parser.add_argument("-o", "--output_file", default="", help="the result file")
parser.add_argument("-d", "--data_file", default="data/RTE-dev.tsv", help="the data")

args = parser.parse_args()

ncorrect, nsamples = 0, 0

pred = [[] for i in range(5)]
for number in range(5):
    with open(args.input_dir + str(number + 1) + '/' + str(number + 1), 'r') as fin:
        for line in fin.readlines():
            pred[number].append(int(line.strip()))
        
with open(args.data_file, encoding="utf-8") as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        if tokens[3] == 'not_entailment':
            target = '0'
        else:
            target = '1'

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
print('ncorrect / nsamples = ' + str(ncorrect) + ' / ' + str(nsamples))
print('---------------------')
print('| bagging: ', float(ncorrect)/float(nsamples))

with open(args.output_file, 'a') as outf:
    outf.write('rte bagging' + '| Accuracy: ' + str(float(ncorrect)/float(nsamples)) + '\n')
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

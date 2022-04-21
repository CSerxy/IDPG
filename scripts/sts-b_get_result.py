from fairseq.models.roberta import RobertaModel
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from os import path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", default="", help="the finetuned directory")
parser.add_argument("-o", "--output_file", default="", help="the result file")
parser.add_argument("-n", "--number", default="1", help="the seed number")
parser.add_argument("-f", "--froms", default="0")
parser.add_argument("-t", "--to", default="0")
parser.add_argument("-d", "--data_file", default="data/STS-B-dev.tsv", help="the data")
parser.add_argument("-l", "--klr", default="1e-2")

args = parser.parse_args()

print(args.input_dir)

roberta = RobertaModel.from_pretrained(
    args.input_dir,
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='STS-B-bin'
)

roberta.cuda()
roberta.eval()
gold, pred = [], []

with open(args.input_dir + args.number, 'w') as fout:
    with open(args.data_file, encoding="utf-8") as fin:
        fin.readline()
        for index, line in enumerate(fin):
            tokens = line.strip().split('\t')
            sent1, sent2, target = tokens[-3], tokens[-2], float(tokens[-1])
            tokens = roberta.encode(sent1, sent2)
            features = roberta.extract_features(tokens)
            prediction = roberta.predict('sentence_classification_head', tokens)
            predictions = 5.0 * roberta.model.classification_heads['sentence_classification_head'](features)
            gold.append(target)
            pred.append(predictions.item())
            fout.write(str(predictions.item()) + '\n')

print('| Pearson: ', pearsonr(gold, pred)[0])
print('| Spearmanr: ', spearmanr(gold, pred)[0])
print('| avg: ', (pearsonr(gold, pred)[0] + spearmanr(gold, pred)[0]) / 2)

with open(args.output_file, 'a') as outf:
    outf.write('sts-b: ' + args.froms + '-' + args.to + '-' + args.number + '-' + args.klr + '| Pearson: ' + str(pearsonr(gold, pred)) + ' | Spearmanr: ' + str(spearmanr(gold, pred)) + ' | avg: ' + str((pearsonr(gold, pred)[0] + spearmanr(gold, pred)[0]) / 2) + '\n')

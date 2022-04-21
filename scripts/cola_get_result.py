from fairseq.models.roberta import RobertaModel
from sklearn.metrics import matthews_corrcoef
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", default="", help="the finetuned directory")
parser.add_argument("-o", "--output_file", default="", help="the result file")
parser.add_argument("-n", "--number", default="1", help="the seed number")
parser.add_argument("-f", "--froms", default="0")
parser.add_argument("-t", "--to", default="0")
parser.add_argument("-d", "--data_file", default="data/CoLA-dev.tsv", help="the data")

args = parser.parse_args()

print(args.input_dir)

roberta = RobertaModel.from_pretrained(
    args.input_dir,
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='CoLA-bin'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()

y_true, y_pred = [], []
with open(args.input_dir + args.number, 'w') as fout:
    with open(args.data_file, encoding="utf-8") as fin:
        for index, line in enumerate(fin):
            tokens = line.strip().split('\t')
            sent, target = tokens[-1], tokens[1]
            tokens = roberta.encode(sent)
            prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
            prediction_label = label_fn(prediction)
            ncorrect += int(prediction_label == target)

            fout.write(prediction_label + '\n')
            y_true.append(int(target))
            y_pred.append(int(prediction_label))
            nsamples += 1
#print('matthews_corrcoef: ' + str(matthews_corrcoef(y_true, y_pred)))
#print('ncorrect / nsamples = ' + str(ncorrect) + ' / ' + str(nsamples))
print('| Accuracy: ', float(ncorrect)/float(nsamples))

with open(args.output_file, 'a') as outf:
    outf.write('cola: ' + args.froms + '-' + args.to + '-' + args.number + ' | Accuracy: ' + str(float(ncorrect)/float(nsamples)) + '\n')

from fairseq.models.roberta import RobertaModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", default="", help="the finetuned directory")
parser.add_argument("-o", "--output_file", default="", help="the result file")
parser.add_argument("-n", "--number", default="1", help="the seed number")
parser.add_argument("-f", "--froms", default="0")
parser.add_argument("-t", "--to", default="0")
parser.add_argument("-d", "--data_file", default="data/QQP-dev.tsv", help="the data")
parser.add_argument("-l", "--klr", default="1e-2")

args = parser.parse_args()

print(args.input_dir)

roberta = RobertaModel.from_pretrained(
    args.input_dir,
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='QQP-bin',
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
tp, fn, fp, tn = 0.0, 0.0, 0.0, 0.0
roberta.cuda()
roberta.eval()

with open(args.input_dir + args.number, 'w') as fout:
    with open(args.data_file, encoding="utf-8") as fin:
        fin.readline()
        for index, line in enumerate(fin):
            tokens = line.strip().split('\t')
            sent1, sent2, target = tokens[3], tokens[4], tokens[5]
            tokens = roberta.encode(sent1, sent2)
            prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
            prediction_label = label_fn(prediction)
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
            fout.write(str(prediction_label) + '\n')
print('f1 = ' + str(tp / (tp + 0.5 * (fp + fn))))
print('| Accuracy: ', float(ncorrect)/float(nsamples))
print('| avg: '+ str((tp / (tp + 0.5 * (fp + fn)) + float(ncorrect)/float(nsamples)) / 2))

with open(args.output_file, 'a') as outf:
    outf.write('qqp: ' + args.froms + '-' + args.to + '-' + args.number + '-' + args.klr + '| f1 = ' + str(tp / (tp + 0.5 * (fp + fn))) + ' | Accuracy: ' + str(float(ncorrect)/float(nsamples)) + ' | avg: '+ str((tp / (tp + 0.5 * (fp + fn)) + float(ncorrect)/float(nsamples)) / 2) + '\n')

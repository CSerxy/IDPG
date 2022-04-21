from fairseq.models.roberta import RobertaModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", default="/fsx/zhuofeng/checkpoint/", help="the finetuned directory")
parser.add_argument("-o", "--output_file", default="/fsx/zhuofeng/result/efl/qnli.txt", help="the result file")
parser.add_argument("-n", "--number", default="1", help="the seed number")
parser.add_argument("-f", "--froms", default="0")
parser.add_argument("-t", "--to", default="0")
parser.add_argument("-k", "--kn", default="")
parser.add_argument("-d", "--data_file", default="data/QNLI-dev.tsv", help="the data")
parser.add_argument("-l", "--klr", default="1e-2")

args = parser.parse_args()

print(args.input_dir)

roberta = RobertaModel.from_pretrained(
    args.input_dir,
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='QNLI-bin'+args.kn,
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()

with open(args.input_dir + args.number, 'w') as fout:
    with open(args.data_file, encoding="utf-8") as fin:
        fin.readline()
        for index, line in enumerate(fin):
            tokens = line.strip().split('\t')
            sent1, sent2, target = tokens[1], tokens[2], tokens[3]
            tokens = roberta.encode(sent1, sent2)
            prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
            prediction_label = label_fn(prediction)
            ncorrect += int(prediction_label == target)
            nsamples += 1

            if prediction_label == 'not_entailment':
                fout.write('0\n')
            else:
                fout.write('1\n')
print('ncorrect / nsamples = ' + str(ncorrect) + ' / ' + str(nsamples))
print('| Accuracy: ', float(ncorrect)/float(nsamples))

with open(args.output_file, 'a') as outf:
    outf.write('qnli: ' + args.froms + '-' + args.to + '-' + args.number + '-' + args.klr + ' | Accuracy: ' + str(float(ncorrect)/float(nsamples)) + '\n')

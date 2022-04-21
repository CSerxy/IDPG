#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import sys
import string
import json

from collections import Counter
from multiprocessing import Pool
from fairseq import tasks

from fairseq.data.encoders.gpt2_bpe import get_encoder


def main():
    """
    Helper script to encode raw text with the GPT-2 BPE using multiple processes.

    The encoder.json and vocab.bpe files can be obtained here:
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-json",
        help='path to encoder.json',
    )
    parser.add_argument(
        "--vocab-bpe",
        type=str,
        help='path to vocab.bpe',
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=['-'],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=['-'],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    assert len(args.inputs) == len(args.outputs), \
        "number of input and output paths should match"

    taskname = "translation"
    srcdictname = "gpt2_bpe/dict.txt"
    task = tasks.get_task(taskname)
    src_dict = task.load_dictionary(srcdictname)

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-" else sys.stdin
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout
            for output in args.outputs
        ]
        with open(args.inputs[0]) as f:
            neighbors = json.load(f)

        encoder = MultiprocessingEncoder(args)
        encoder.initializer()
        mm = {}

        def get_idx(src_dict, encoder, i):
            bpe_key = encoder.encode(i)
            key_ids = src_dict.encode_line(
                line=bpe_key,
                line_tokenizer=lambda t: t,
                add_if_not_exist=False,
                consumer=None,
                append_eos=False,
                reverse_order=False,
            )
            tmp = []
            for key_id in key_ids:
                tmp.append(str(int(key_id)))
            return tmp

        with open(args.outputs[0], 'w') as outf:
            for i, j in neighbors.items():
                if len(j) > 0:
                    tmp = get_idx(src_dict, encoder, i)
                    outf.write(" ".join(tmp))

                    if len(tmp) not in mm:
                        mm[len(tmp)] = 0
                    mm[len(tmp)] += 1
                    
                    for k in j:
                        tmp = get_idx(src_dict, encoder, k)
                        outf.write(", " + " ".join(tmp))
                    outf.write('\n')

                    tmp = get_idx(src_dict, encoder, ' '+i)
                    outf.write(" ".join(tmp))

                    if len(tmp) not in mm:
                        mm[len(tmp)] = 0
                    mm[len(tmp)] += 1
                    
                    for k in j:
                        tmp = get_idx(src_dict, encoder, ' '+k)
                        outf.write(", " + " ".join(tmp))
                    outf.write('\n')

                    tmp = get_idx(src_dict, encoder, i[0].upper() + i[1:])
                    outf.write(" ".join(tmp))

                    if len(tmp) not in mm:
                        mm[len(tmp)] = 0
                    mm[len(tmp)] += 1
                    
                    for k in j:
                        tmp = get_idx(src_dict, encoder, k[0].upper() + k[1:])
                        outf.write(", " + " ".join(tmp))
                    outf.write('\n')

                    tmp = get_idx(src_dict, encoder, ' ' + i[0].upper() + i[1:])
                    outf.write(" ".join(tmp))

                    if len(tmp) not in mm:
                        mm[len(tmp)] = 0
                    mm[len(tmp)] += 1
                    
                    for k in j:
                        tmp = get_idx(src_dict, encoder, ' ' + k[0].upper() + k[1:])
                        outf.write(", " + " ".join(tmp))
                    outf.write('\n')

                    #tmp = get_idx(src_dict, encoder, i.upper())
                    #outf.write(" ".join(tmp))

                    #if len(tmp) not in mm:
                    #    mm[len(tmp)] = 0
                    #mm[len(tmp)] += 1
                    #
                    #for k in j:
                    #    tmp = get_idx(src_dict, encoder, k.upper())
                    #    outf.write(", " + " ".join(tmp))
                    #outf.write('\n')

                    #tmp = get_idx(src_dict, encoder, ' '+i.upper())
                    #outf.write(" ".join(tmp))

                    #if len(tmp) not in mm:
                    #    mm[len(tmp)] = 0
                    #mm[len(tmp)] += 1
                    #
                    #for k in j:
                    #    tmp = get_idx(src_dict, encoder, ' '+k.upper())
                    #    outf.write(", " + " ".join(tmp))
                    #outf.write('\n')
                    #exit(0)

        print(mm)

        #stats = Counter()
        #mm = {}
        #for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
        #    if filt == "PASS":
        #        for enc_line, output_h in zip(enc_lines, outputs):
        #            if len(enc_line) not in mm:
        #                mm[len(enc_line)] = 0
        #            mm[len(enc_line)] += 1
        #            print(enc_line, file=output_h)
        #    else:
        #        stats["num_filtered_" + filt] += 1
        #    if i % 10000 == 0:
        #        print("processed {} lines".format(i), file=sys.stderr)

        #for k, v in stats.most_common():
        #    print("[{}] filtered {} lines".format(k, v), file=sys.stderr)
        #print(mm)


class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        bpe = get_encoder(self.args.encoder_json, self.args.vocab_bpe)

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        key = self.encode(lines[0])
        enc_lines.append(" ".join(key))

        for line in lines[1]:
            tokens = self.encode(line)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]


if __name__ == "__main__":
    main()

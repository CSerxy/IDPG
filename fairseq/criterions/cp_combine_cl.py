# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
#from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.losses import GenericPairLoss

def compute_cross_entropy_loss(logits, targets, ignore_index=-100):
    """
    Function to compute the cross entropy loss. The default value of
    ignore_index is the same as the default value for F.cross_entropy in
    pytorch.
    """
    print(logits.size())
    print(targets.size())
    assert logits.size(0) == targets.size(-1), \
        "Logits and Targets tensor shapes don't match up"

    loss = F.nll_loss(
        F.log_softmax(logits, -1, dtype=torch.float32),
        targets,
        reduction="sum",
        ignore_index=ignore_index,
    )
    return loss

class NTXentLoss(GenericPairLoss):

    def __init__(self, temperature, **kwargs):
        super().__init__(use_similarity=True, mat_based_loss=False, **kwargs)
        self.temperature = temperature

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, _ = indices_tuple

        if len(a1) > 0 and len(a2) > 0:
            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = (a2.unsqueeze(0) == a1.unsqueeze(1)).float()
            neg_pairs = neg_pairs*n_per_p
            neg_pairs[n_per_p==0] = float('-inf')

            max_val = torch.max(pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0].half()) ###This is the line change
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)
            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator/denominator) + 1e-20)
            return {"loss": {"losses": -log_exp, "indices": (a1, p), "reduction_type": "pos_pair"}}
        return self.zero_losses()

def jaccard(arg1, arg2):
    f = []
    for i in arg1:
        tmp = set()
        for j in i:
            tmp.add(int(j))
        f.append(tmp)
    for i in arg2:
        tmp = set()
        for j in i:
            tmp.add(int(j))
        f.append(tmp)

    record = 0
    score1, score2 = 0, 0
    for i in range(len(f)):
        res = []
        pos = (i + len(arg1)) % (2 * len(arg1))
        for j in range(len(f)):
            if j != i:
                res.append((float(len(f[i].intersection(f[j]))) / len(f[i].union(f[j])), j == pos))
        res = sorted(res,key=lambda x:x[0], reverse = True)
        for j in range(len(res)):
            if res[j][1]:
                record += j + 1
                score1 += res[j][0]
                if j == 0:
                    score2 += res[1][0]
                else:
                    score2 += res[0][0]
                break
    return record / len(f), score1 / len(f), score2 / len(f)

@register_criterion('cp_combine_cl')
class Cp_Combine_CLLoss(FairseqCriterion):
    """
    Implementation for the loss used in contrastive learning model training.
    """

    def __init__(self, task, tpu):
        super().__init__(task)
        self.args = task.args
        self.tpu = tpu
        self.count = 0
        self.rank = 0
        self.score1 = 0
        self.score2 = 0

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        masked_tokens = sample['target'].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum()

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if self.tpu:
            masked_tokens = None  # always project all tokens on TPU
        elif masked_tokens.device == torch.device('cpu'):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )

        lm_logits = model(**sample['net_input'], masked_tokens=masked_tokens)[0]
        lm_targets = model.get_targets(sample, [lm_logits]) 
        if masked_tokens is not None:
            lm_targets = lm_targets[masked_tokens]

        lm_loss = modules.cross_entropy(
            lm_logits.view(-1, lm_logits.size(-1)),
            lm_targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        ) 

        # compute the number of tokens for which loss is computed. This is used
        # to normalize the loss
        ntokens = utils.strip_pad(lm_targets, self.padding_idx).numel()
        loss = lm_loss

        if self.args.cl_loss_weight != 0:
            # Size: B * C, B: batch size, C: output representation dim
            logits1, _ = model(sample['arg1'], masked_tokens=None)
            logits2, _ = model(sample['arg2'], masked_tokens=None)        
            logits_ = torch.cat((logits1, logits2))
            indices_ = torch.arange(0, logits1.size(0), device=logits1.device)
            labels_ = torch.cat((indices_, indices_))

            cl_loss = NTXentLoss(temperature=0.10)
            sentence_loss = cl_loss(logits_, labels_)
            loss += self.args.cl_loss_weight * sentence_loss * sample_size
        #print(lm_loss / sample_size, sentence_loss, loss)
        #print(lm_loss / ntokens)
        
        logging_output = {
            'loss': loss if self.tpu else loss.data,
            #'lm_loss': lm_loss if self.tpu else lm_loss.data,
            #'sentence_loss': sentence_loss if self.tpu else sentence_loss.data,
            'ntokens': ntokens,
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        #print(loss, sample_size, sample['ntokens'])
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

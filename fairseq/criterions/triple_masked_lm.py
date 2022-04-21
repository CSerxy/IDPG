# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('triple_masked_lm')
class TripleMaskedLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, task, tpu):
        super().__init__(task)
        self.tpu = tpu

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        def get_logits_and_targets(sample, model, target, net_input):
            masked_tokens = sample[target].ne(self.padding_idx)
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

            logits = model(**sample[net_input], masked_tokens=masked_tokens)[0]
            targets = sample[target]
            if masked_tokens is not None:
                targets = targets[masked_tokens]
            return logits, targets, sample_size
        logits1, targets1, sample_size1 = get_logits_and_targets(sample, model, 'target', 'net_input')
        logits2, targets2, sample_size2 = get_logits_and_targets(sample, model, 'target2', 'net_input2')
        #logits3, targets3, sample_size3 = get_logits_and_targets(sample, model, 'target3', 'net_input3')
        #sample_size = sample_size1 + sample_size2 + sample_size3
        sample_size = sample_size1 + sample_size2
        #logits = torch.cat((logits1, logits2, logits3), 0)
        #targets = torch.cat((targets1, targets2, targets3), 0)
        logits = torch.cat((logits1, logits2), 0)
        targets = torch.cat((targets1, targets2), 0)

        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )

        logging_output = {
            'loss': loss if self.tpu else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
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

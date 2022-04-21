# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    TransformerSentenceEncoder,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.models.lstm import LSTMModel
from fairseq.models.lstm import LSTMEncoder, LSTMDecoder

from .hub_interface import RobertaHubInterface


logger = logging.getLogger(__name__)


@register_model('roberta')
class RobertaModel(FairseqEncoderModel):

    @classmethod
    def hub_models(cls):
        return {
            'roberta.base': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz',
            'roberta.large': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz',
            'roberta.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz',
            'roberta.large.wsc': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.wsc.tar.gz',
        }

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-layers', type=int, metavar='L',
                            help='num encoder layers')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='H',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='F',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='A',
                            help='num encoder attention heads')
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use for pooler layer')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN')
        parser.add_argument('--pooler-dropout', type=float, metavar='D',
                            help='dropout probability in the masked_lm pooler layers')
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')
        parser.add_argument('--load-checkpoint-heads', action='store_true',
                            help='(re-)register and load heads when loading checkpoints')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        parser.add_argument('--untie-weights-roberta', action='store_true',
                            help='Untie weights between embeddings and classifiers in RoBERTa')
        # args for prefix tuning
        parser.add_argument('--add-prefix', default=False, action='store_true',
                            help='whether adding prefix tuning')
        parser.add_argument('--prefix-len', type=int, default=0,
                            help='the prefix token length')
        parser.add_argument('--prefix-prompt', type=str, default=None,
                            help='the initilized prefix prompt')
        #parser.add_argument('--prefix-MLP-mode', type=str, default='none',
        #                    help='either none or shared or separate')
        #parser.add_argument('--prefix-pos', type=str, default='input',
        #                    help='either input or layers')
        parser.add_argument('--add-suffix', default=False, action='store_true',
                            help='whether adding prefix tuning')
        parser.add_argument('--suffix-len', type=int, default=0,
                            help='the prefix token length')
        parser.add_argument('--suffix-prompt', type=str, default=None,
                            help='the initilized suffix prompt')
        parser.add_argument('--prompt-generation', default=False, action='store_true',
                            help='whether generates prompt from example')
        parser.add_argument('--glove', default=None, type=str,
                            help='glove file path, such as .word_vectors_cache/glove.6B.50d.txt')
        parser.add_argument('--generation-net', default='dnn', type=str,
                            help='the decoder network type, dnn or rnn')
        parser.add_argument('--generation-layer', default=1, type=int,
                            help='the number of decoder layer')
        parser.add_argument('--generation-freeze', default=False, action='store_true',
                            help='whether freezes the first generater update')
        parser.add_argument('--freeze-encoder', action='store_true', default=False)
        parser.add_argument('--generation-quaternions', default=None, type=int)
        parser.add_argument('--lphm', default=None, type=int)
        parser.add_argument('--middle-prompt-insert-layer', type=int, default=25,
                            help='the start layer to insert middle prompt')
        parser.add_argument('--middle-prompt-mode', type=str, default='none',
                            help='either none or shared or separate')
        parser.add_argument('--middle-previous', default=False, action='store_true',
                            help='weather middle idpg uses previous layer cls token, if false it uses first layer cls')
        parser.add_argument('--adapter-insert-layer', type=int, default=25,
                            help='the start layer to insert middle prompt')
        parser.add_argument('--adapter-arch', type=str, default='none',
                            help='either none or houlsby or compacter or pfeiffer')
        parser.add_argument('--adapter-bottleneck-dim', type=int, default=256, 
                            help='the botteleneck dim in adapter')
        parser.add_argument('--compacter-n', type=int, default=4,
                            help='the n in compacter')
        parser.add_argument('--generator-layer-norm', default=False, action='store_true')
        parser.add_argument('--generator-residual', default=False, action='store_true')
        parser.add_argument('--sbert', action='store_true', default=False)
        parser.add_argument('--sbert-mode', default=1, type=int)
        parser.add_argument('--prompt-insert-mode', default=1, type=int)

        parser.add_argument('--reparameterization', default='None', type=str)
        parser.add_argument('--phm-bottleneck-dim', type=int, default=16, 
                            help='the botteleneck dim in phm')

        parser.add_argument('--insert-position', type=int, default=4, 
                            help='the insert prompt position, 1: before the first [sep], 2: after the second [sep], 3: before the third [sep], None: for single sentence task, need to add two more [sep] for the inserted prompt.')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, classification_head_name=None, **kwargs):
        if classification_head_name is not None:
            features_only = True

        #bpe_sentence = self.bpe.encode('This is a good movie.')
        #print(self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False, add_if_not_exist=False))
        #print(src_tokens[:1])
        #print(src_tokens.size())
        #for i in range(src_tokens.size()[0]):
        #    for j in range()

        x, extra = self.encoder(src_tokens, features_only, return_all_hiddens, **kwargs)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        if self.args.lstm:
            self.classification_heads[name] = LSTMClassificationHead(
                self.args.encoder_embed_dim,
                inner_dim or self.args.encoder_embed_dim,
                num_classes,
                self.args.pooler_activation_fn,
                self.args.pooler_dropout,
                self.args.quant_noise_pq,
                self.args.quant_noise_pq_block_size,
            )
        else:
            self.classification_heads[name] = RobertaClassificationHead(
                self.args.encoder_embed_dim * self.args.sbert_mode,
                inner_dim or self.args.encoder_embed_dim,
                num_classes,
                self.args.pooler_activation_fn,
                self.args.pooler_dropout,
                self.args.quant_noise_pq,
                self.args.quant_noise_pq_block_size,
            )

    @property
    def supported_targets(self):
        return {'self'}

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', bpe='gpt2', **kwargs):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return RobertaHubInterface(x['args'], x['task'], x['models'][0])

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''

        # rename decoder -> encoder before upgrading children modules
        for k in list(state_dict.keys()):
            if k.startswith(prefix + 'decoder'):
                new_k = prefix + 'encoder' + k[len(prefix + 'decoder'):]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            [] if not hasattr(self, 'classification_heads')
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            if prefix + 'classification_heads.' + head_name + '.dense_after_lstm.weight' in state_dict:
                inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense_after_lstm.weight'].size(0)
            else:
                inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    #or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    logger.info('Overwriting ' + prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout, q_noise=0, qn_block_size=8):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class LSTMClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout, q_noise=0, qn_block_size=8):
        super().__init__()
        #self.lstmencoder = LSTMEncoder(
        #    dictionary=dictionary, 
        #    embed_dim=input_dim, 
        #    hidden_size=inner_dim,
        #)
        #self.lstmdecoder = LSTMDecoder(
        #    dictionary=dictionary,
        #    embed_dim=inner_dim,
        #    hidden_size=inner_dim,
        #    out_embed_dim=inner_dim,
        #)
        #self.lstm = LSTMModel(self.lstmencoder, self.lstmdecoder)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=inner_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.tmp_dim = inner_dim
        self.dense_after_lstm = nn.Linear(2 * inner_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )

    def forward(self, features, **kwargs):
        x = features[:, :, :]  # take <s> token (equiv. to [CLS])
        x, _ = self.lstm(x)
        x = torch.cat(
            [
                x[:, :, :self.tmp_dim],
                torch.flip(x[:, :, self.tmp_dim:], [1]),
            ],
            -1,
        )
        x = x[:, 0, :]
        x = self.dropout(x)
        x = self.dense_after_lstm(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaEncoder(FairseqEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        if args.prompt_generation:
            print('generating prompt for each example...')
            if args.generation_freeze:
                n_trans_layers_to_freeze = 24
            else:
                n_trans_layers_to_freeze = 0
            if args.sbert:
                if args.generation_quaternions is None:
                    if args.sbert_mode == 3:
                        self.generation_dense = nn.Linear(args.encoder_embed_dim, 256)
                    elif args.sbert_mode == 6:
                        self.generation_dense = nn.Linear(2 * args.encoder_embed_dim, 256)
                    self.generation_out_proj = nn.Linear(256, args.suffix_len * args.encoder_embed_dim)


                    self.generation_dense.apply(init_bert_params)
                    self.generation_out_proj.apply(init_bert_params)
                else:
                    if args.sbert_mode == 3:
                        self.generation_dense_w = [torch.empty(args.encoder_embed_dim // args.generation_quaternions, 256 // args.generation_quaternions, dtype=torch.half) for i in range(args.generation_quaternions)]
                        self.generation_dense_b = torch.empty(256, dtype=torch.half)
                    elif args.sbert_mode == 6:
                        self.generation_dense_w = [torch.empty(2 * args.encoder_embed_dim // args.generation_quaternions, 256 // args.generation_quaternions, dtype=torch.half) for i in range(args.generation_quaternions)]
                        self.generation_dense_b = torch.empty(256, dtype=torch.half)

                    self.generation_out_proj_w = [torch.empty(256 // args.generation_quaternions, args.suffix_len * args.encoder_embed_dim // args.generation_quaternions, dtype=torch.half) for i in range(args.generation_quaternions)]
                    self.generation_out_proj_b = torch.empty(args.suffix_len * args.encoder_embed_dim, dtype=torch.half)
                    self.generation_shared_w = [torch.empty(args.generation_quaternions, args.generation_quaternions, dtype=torch.half) for i in range(args.generation_quaternions)]
                    for i in range(args.generation_quaternions):
                        self.generation_dense_w[i].normal_(mean=0.0, std=0.02)
                        self.generation_out_proj_w[i].normal_(mean=0.0, std=0.02)
                        self.generation_shared_w[i].normal_(mean=0.0, std=0.02)
                    self.generation_dense_b.zero_()
                    self.generation_out_proj_b.zero_()

                self.generation_activation_fn = utils.get_activation_fn("gelu")
                self.generation_dropout = nn.Dropout(p=0.1)

                self.sentence_encoder = TransformerSentenceEncoder(
                    padding_idx=dictionary.pad(),
                    vocab_size=len(dictionary),
                    num_encoder_layers=args.encoder_layers,
                    embedding_dim=args.encoder_embed_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=args.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    layerdrop=args.encoder_layerdrop,
                    max_seq_len=args.max_positions,
                    num_segments=0,
                    encoder_normalize_before=True,
                    apply_bert_init=True,
                    activation_fn=args.activation_fn,
                    q_noise=args.quant_noise_pq,
                    qn_block_size=args.quant_noise_pq_block_size,
                    freeze_embeddings=args.generation_freeze,
                    n_trans_layers_to_freeze=n_trans_layers_to_freeze,
                    sentence_generation=None,
                    insert_position=args.insert_position,
                    sbert=args.sbert,
                )
            else:
                if args.glove is not None:
                    self.sentence_generation = 'glove'
                    from fairseq.data import encoders
                    args.bpe = 'gpt2'
                    bpe = encoders.build_bpe(args)
                else:
                    self.sentence_generation = 'roberta'
                #self.sentence_generation = TransformerSentenceEncoder(
                #    padding_idx=dictionary.pad(),
                #    vocab_size=len(dictionary),
                #    num_encoder_layers=args.encoder_layers,
                #    embedding_dim=args.encoder_embed_dim,
                #    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                #    num_attention_heads=args.encoder_attention_heads,
                #    dropout=args.dropout,
                #    attention_dropout=args.attention_dropout,
                #    activation_dropout=args.activation_dropout,
                #    layerdrop=args.encoder_layerdrop,
                #    max_seq_len=args.max_positions,
                #    num_segments=0,
                #    encoder_normalize_before=True,
                #    apply_bert_init=True,
                #    activation_fn=args.activation_fn,
                #    q_noise=args.quant_noise_pq,
                #    qn_block_size=args.quant_noise_pq_block_size,
                #    freeze_embeddings=args.generation_freeze,
                #    n_trans_layers_to_freeze=n_trans_layers_to_freeze,
                #    sentence_generation=None,
                #    insert_position=args.insert_position,
                #    sbert=args.sbert,
                #)
        else:
            self.sentence_generation = None

        if args.freeze_encoder:
            n_trans_layers_to_freeze = 24
        else:

            n_trans_layers_to_freeze = 0

        if not args.sbert:
            self.sentence_encoder = TransformerSentenceEncoder(
                dictionary=dictionary,
                bpe=bpe if self.sentence_generation == 'glove' else None,
                padding_idx=dictionary.pad(),
                vocab_size=len(dictionary),
                num_encoder_layers=args.encoder_layers,
                embedding_dim=args.encoder_embed_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                layerdrop=args.encoder_layerdrop,
                max_seq_len=args.max_positions,
                num_segments=0,
                encoder_normalize_before=True,
                apply_bert_init=True,
                activation_fn=args.activation_fn,
                q_noise=args.quant_noise_pq,
                qn_block_size=args.quant_noise_pq_block_size,
                add_prefix=args.add_prefix,
                prefix_len=args.prefix_len,
                prefix_prompt=args.prefix_prompt,
                add_suffix=args.add_suffix,
                suffix_len=args.suffix_len,
                suffix_prompt=args.suffix_prompt,
                sentence_generation=self.sentence_generation,
                freeze_embeddings=args.freeze_encoder,
                n_trans_layers_to_freeze=n_trans_layers_to_freeze,
                insert_position=args.insert_position,
                generation_net=args.generation_net,
                generation_layer=args.generation_layer,
                sbert=args.sbert,
                generation_quaternions=args.generation_quaternions,
                lphm=args.lphm,
                middle_prompt_mode=args.middle_prompt_mode,
                middle_previous=args.middle_previous,
                middle_prompt_insert_layer=args.middle_prompt_insert_layer,
                adapter_insert_layer=args.adapter_insert_layer,
                adapter_arch=args.adapter_arch,
                adapter_bottleneck_dim=args.adapter_bottleneck_dim,
                compacter_n=args.compacter_n,
                generator_residual=args.generator_residual,
                generator_layer_norm=args.generator_layer_norm,
                reparameterization=args.reparameterization,
                phm_bottleneck_dim=args.phm_bottleneck_dim,
                prompt_insert_mode=args.prompt_insert_mode,
                glove_path=args.glove,
                #prefix_MLP_mode=args.prefix_MLP_mode,
                #prefix_pos=args.prefix_pos,
            )
        args.untie_weights_roberta = getattr(args, 'untie_weights_roberta', False)

        self.lm_head = RobertaLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=self.sentence_encoder.embed_tokens.weight if not args.untie_weights_roberta else None,
        )

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, masked_tokens=None, **unused):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        if not self.args.sbert:
            x, extra = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)
            if not features_only:
                x = self.output_layer(x, masked_tokens=masked_tokens)
        else:
            x, extra = self.extract_sbert_features(src_tokens)

        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **unused):
        inner_states, _ = self.sentence_encoder(
            src_tokens,
            last_state_only=not return_all_hiddens,
        )

        features = inner_states[-1].transpose(0, 1)  # T x B x C -> B x T x C
        return features, {'inner_states': inner_states if return_all_hiddens else None}

    def extract_sbert_features(self, src_tokens, return_all_hiddens=False, **unused):
        if self.args.sbert_mode == 3:
            inner_states, _ = self.sentence_encoder(
                src_tokens,
                last_state_only=not return_all_hiddens,
            )

            sent_features = inner_states[-1].transpose(0, 1)  # T x B x C -> B x T x C
            sent_features = sent_features[:, 0, :] # B x 1 x C
            x = self.generation_dropout(sent_features)
            if self.args.generation_quaternions is not None:
                tmp_x = self.generation_dense_b.repeat(x.size()[0], 1).to('cuda:0')
                for i in range(self.args.generation_quaternions):
                    tmp_x += torch.mm(x, torch.kron(self.generation_shared_w[i], self.generation_dense_w[i]).to('cuda:0'))
                x = self.generation_activation_fn(tmp_x)
                x = self.generation_dropout(x)

                tmp_x = self.generation_out_proj_b.repeat(x.size()[0], 1).to('cuda:0')
                for i in range(self.args.generation_quaternions):
                    tmp_x += torch.mm(x, torch.kron(self.generation_shared_w[i], self.generation_out_proj_w[i]).to('cuda:0'))
                x = tmp_x
            else:
                x = self.generation_dense(x)
                x = self.generation_activation_fn(x)
                x = self.generation_dropout(x)
                x = self.generation_out_proj(x)
            x = x.view(src_tokens.size()[0], -1, self.args.encoder_embed_dim) # T x 5 x C

            prompt_states, _ = self.sentence_encoder.prompt_forward(
                x,
                last_state_only=not return_all_hiddens,
            )
            prompt_features = prompt_states[-1].transpose(0, 1) # 7 x B x C -> B x 7 x C
            prompt_features = prompt_features[:, 0, :] # B x 1 x C

            features = torch.cat([sent_features, prompt_features, torch.abs(torch.sub(sent_features, prompt_features))], dim=1)
            features = features.view(features.size()[0], -1, features.size()[-1])
        elif self.args.sbert_mode == 6:
            src_tokens1, src_tokens2 = [], []
            len1, len2 = [], []
            max1, max2 = 0, 0
            for i in range(src_tokens.size()[0]):
                for stop_idx in range(src_tokens.size()[1]):
                    if src_tokens[i][stop_idx] == 2:
                        break
                len1.append(stop_idx)
                if stop_idx + 1 > max1:
                    max1 = stop_idx + 1
                for stop_idx in range(len1[-1] + 2, src_tokens.size()[1]):
                    if src_tokens[i][stop_idx] == 2:
                        break
                len2.append(stop_idx)
                if len2[-1] - len1[-1] > max2:
                    max2 = len2[-1] - len1[-1]
            for i in range(src_tokens.size()[0]):
                src_tokens1.append(torch.cat((src_tokens[i][:len1[i] + 1], torch.LongTensor([1 for _ in range(max1 - len1[i] - 1)]).type_as(src_tokens)), dim=0))
                src_tokens2.append(torch.cat((torch.LongTensor([0]).type_as(src_tokens), src_tokens[i][len1[i] + 2: len2[i] + 1], torch.LongTensor([1 for _ in range(max2 - len2[i] + len1[i])]).type_as(src_tokens)), dim=0))

            src_tokens1 = torch.stack(src_tokens1)
            src_tokens2 = torch.stack(src_tokens2)
            
            inner_states1, _ = self.sentence_encoder(
                src_tokens1,
                last_state_only=not return_all_hiddens,
            )
            sent_features1 = inner_states1[-1].transpose(0, 1)  # T x B x C -> B x T x C
            sent_features1 = sent_features1[:, 0, :] # B x C

            inner_states2, _ = self.sentence_encoder(
                src_tokens2,
                last_state_only=not return_all_hiddens,
            )
            sent_features2 = inner_states2[-1].transpose(0, 1)  # T x B x C -> B x T x C
            sent_features2 = sent_features2[:, 0, :] # B x C
            sent_features = torch.cat([sent_features1, sent_features2], dim=1)

            x = self.generation_dropout(sent_features)

            if self.args.generation_quaternions is not None:
                tmp_x = self.generation_dense_b.repeat(x.size()[0], 1).to('cuda:0')
                for i in range(self.args.generation_quaternions):
                    tmp_x += torch.mm(x, torch.kron(self.generation_shared_w[i], self.generation_dense_w[i]).to('cuda:0'))
                x = self.generation_activation_fn(tmp_x)
                x = self.generation_dropout(x)

                tmp_x = self.generation_out_proj_b.repeat(x.size()[0], 1).to('cuda:0')
                for i in range(self.args.generation_quaternions):
                    tmp_x += torch.mm(x, torch.kron(self.generation_shared_w[i], self.generation_out_proj_w[i]).to('cuda:0'))
                x = tmp_x
            else:
                x = self.generation_dense(x)
                x = self.generation_activation_fn(x)
                x = self.generation_dropout(x)
                x = self.generation_out_proj(x)

            x = x.view(src_tokens.size()[0], -1, self.args.encoder_embed_dim) # T x 5 x C

            prompt_states, _ = self.sentence_encoder.prompt_forward(
                x,
                last_state_only=not return_all_hiddens,
            )
            prompt_features = prompt_states[-1].transpose(0, 1) # 7 x B x C -> B x 7 x C
            prompt_features = prompt_features[:, 0, :] # B x C

            features = torch.cat([sent_features1, sent_features2, prompt_features, torch.abs(torch.sub(sent_features1, prompt_features)), torch.abs(torch.sub(sent_features2, prompt_features)), torch.abs(torch.sub(sent_features1, sent_features2))], dim=1)
            features = features.view(features.size()[0], -1, features.size()[-1])

        return features, {'inner_states': inner_states if return_all_hiddens else None}

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


@register_model_architecture('roberta', 'roberta')
def base_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)
    args.encoder_layers_to_keep = getattr(args, 'encoder_layers_to_keep', None)
    args.encoder_layerdrop = getattr(args, 'encoder_layerdrop', 0.0)


@register_model_architecture('roberta', 'roberta_base')
def roberta_base_architecture(args):
    base_architecture(args)


@register_model_architecture('roberta', 'roberta_large')
def roberta_large_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    base_architecture(args)


@register_model_architecture('roberta', 'xlm')
def xlm_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 16)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1280)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1280*4)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    base_architecture(args)

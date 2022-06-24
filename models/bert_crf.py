from turtle import forward
from typing import List
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    BertModel,
    get_linear_schedule_with_warmup,)
from torch import nn
from torchcrf import CRF
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert.modeling_bert import BertPreTrainedModel
import pytorch_lightning as pl
import itertools
from utils import compute_metrics


class RuleProcessor:
    def process(self, crf: CRF, labels_list, imp_value=-1e4):
        num_labels = len(labels_list)
        for i in range(num_labels):
            tag_name = labels_list[i]
            # Rule 1 : I label is impossible to be start label
            if tag_name.startswith("I-"):
                nn.init.constant_(crf.start_transitions[i], imp_value)

        # Rule 2 : I label is only followed by I label or B label of same type
        for i in range(num_labels):
            i_tag_name = labels_list[i]
            for j in range(num_labels):
                j_tag_name = labels_list[j]
                if j_tag_name.startswith("I-") and (i_tag_name[2:] != j_tag_name[2:]):
                    nn.init.constant_(crf.transitions[i, j], imp_value)


class CustomNERCRF(pl.LightningModule):
    def __init__(self,
                 model_name_or_path: str,
                 num_labels: int,
                 learning_rate: float = 1e-5,
                 adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0,
                 weight_decay: float = 0.0,):
        super().__init__()
        self.save_hyperparameters()
        # self.num_labels = num_labels
        # self.ignore_labels = config.ignore_labels
        self.hparams.ignore_labels = -100

        # Load MOdel and get it body

        self.config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=num_labels)
        self.config.output_hidden_states = True
        self.config.output_attentions = True
        self.model = AutoModel.from_pretrained(
            model_name_or_path, config=self.config)
        # self.model = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            self.model.config.hidden_size, self.hparams.num_labels)
        self.crf = CRF(num_tags=self.hparams.num_labels, batch_first=True)

    def _get_bert_features(self, input_ids, attention_mask):
        """get features from body of bert model

        Args:
            input_ids (tensor.FloatTensor): input_ids after tokenize and convert to id
            attention_mask (_type_): input attention_mask
        Returns:
            (tensor.FloatTensor, tensor.FloatTensor) : features of bert model shape (batch_size, seq_len, num_labels)
                                                        and bert sequence output tuple (batch_size, seq_len, hidden_size)
        """

        bert_seq_out = self.model(
            input_ids=input_ids, attention_mask=attention_mask)

        bert_seq_out_last = bert_seq_out[0]
        bert_seq_out_last = self.dropout(bert_seq_out_last)
        bert_feats = self.classifier(bert_seq_out_last)
        return bert_feats, bert_seq_out

    def init_crf_transitions(self, rule_processor: RuleProcessor, labels_list, imp_value=-1e4):
        """
        :param tag_list: ['O', "B-LOCATION', ...]
        :param imp_value: value that we assign for impossible transition, ex : B-LOCATION -> I-PERSON
        """
        rule_processor.process(self.crf, labels_list, imp_value)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                prediction_mask=None,
                ):
        """Performs the forward pass of the network.

        If `labels` is not `None`, it will calculate and return the the loss,
        that is the negative log-likelihood of the batch.
        Otherwise, it will calculate the most probable sequence outputs using
        Viterbi decoding and return a list of sequences (List[List[int]]) of
        variable lengths.

        Args:
            input_ids: tensor of input token ids.
            attention_mask: mask tensor that should have value 0 for [PAD]
                tokens and 1 for other tokens.
            labels: tensor of gold NER tag label ids. Values should be ints in
                the range [0, config.num_labels - 1].
            prediction_mask: mask tensor should have value 0 for tokens that do
                not have an associated prediction, such as [CLS] and WordPìece
                subtoken continuations (that start with ##).

        Returns a dict with calculated tensors:
          - "logits"
          - "loss" (if `labels` is not `None`)
          - "y_pred" (if `labels` is `None`)
        """

        bert_feats, bert_out = self._get_bert_features(
            input_ids, attention_mask)

        outputs = {}
        outputs['logits'] = bert_feats

        mask = prediction_mask
        batch_size = bert_feats.shape[0]

        if labels is not None:
            # Negative of the log likelihood.
            # Loop through the batch here because of 2 reasons:
            # 1- the CRF package assumes the mask tensor cannot have interleaved
            # zeros and ones. In other words, the mask should start with True
            # values, transition to False at some moment and never transition
            # back to True. That can only happen for simple padded sequences.
            # 2- The first column of mask tensor should be all True, and we
            # cannot guarantee that because we have to mask all non-first
            # subtokens of the WordPiece tokenization.
            loss = 0
            for seq_logits, seq_labels, seq_mask in zip(bert_feats, labels, mask):
                # Index logits and labels using prediction mask to pass only the
                # first subtoken of each word to CRF.
                assert sum(seq_mask) != 0
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                seq_labels = seq_labels[seq_mask].unsqueeze(0)
                loss -= self.crf(seq_logits, seq_labels,
                                 reduction='token_mean')

            loss /= batch_size
            outputs['loss'] = loss
        else:
            output_tags = []
            for seq_logits, seq_mask in zip(bert_feats, mask):
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                tags = self.crf.decode(seq_logits)
                # Unpack "batch" results
                output_tags.append(tags[0])
            outputs['y_pred'] = output_tags

        return outputs

    @property
    def label_name_list(self):
        return ('B-AGE',
                'B-DATE',
                'B-GENDER',
                'B-JOB',
                'B-LOCATION',
                'B-NAME',
                'B-ORGANIZATION',
                'B-PATIENT_ID',
                'B-SYMPTOM_AND_DISEASE',
                'B-TRANSPORTATION',
                'I-AGE',
                'I-DATE',
                'I-GENDER',
                'I-JOB',
                'I-LOCATION',
                'I-NAME',
                'I-ORGANIZATION',
                'I-PATIENT_ID',
                'I-SYMPTOM_AND_DISEASE',
                'I-TRANSPORTATION',
                'O')

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        model = self.model
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer=optimizer,
        #     num_warmup_steps=self.hparams.warmup_steps,
        #     num_training_steps=200
        # )
        # scheduler = {"scheduler": scheduler,
        #              "interval": "step", "frequency": 1}
        # return [optimizer], [scheduler]
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x = train_batch
        outputs = self.forward(**x)
        loss = outputs['loss']

        # Log
        self.log('train_loss', loss.item())

        return {"loss": loss}

    def post_process(self, labels: torch.Tensor):
        """ Process label to remove ingnored labels"""
        true_labels_step = [
            [label for label in sent_label[1:]
                if label != self.hparams.ignore_labels]
            for sent_label in labels.tolist()
        ]
        return true_labels_step

    def validation_step(self, val_batch, batch_idx):
        x = val_batch
        input_ids = x['input_ids']
        labels = x['labels']
        prediction_mask = x['prediction_mask']
        attention_mask = x['attention_mask']
        outputs = self(
            input_ids,
            attention_mask,
            prediction_mask=prediction_mask
        )

        # loss = outputs['loss']
        y_pred = outputs['y_pred']
        true_labels_step = self.post_process(labels)

        for index, sent in enumerate(y_pred):
            assert len(sent) == sum(prediction_mask[index])

        return {"preds": y_pred, "labels": true_labels_step}

    def validation_epoch_end(self, outputs):
        # preds = np.concatenate([o['preds'] for o in outputs], axis=0)
        # labels = np.concatenate([o['labels'] for o in outputs], axis=0)
        preds = list(itertools.chain(*[o['preds'] for o in outputs]))
        labels = list(itertools.chain(*[o['labels'] for o in outputs]))

        results = compute_metrics((preds, labels), self.label_name_list,
                                  return_logits=False)

        self.log('precision', results['precision'])
        self.log('recall', results['recall'])
        self.log('f1', results['f1'])
        self.log('accuracy', results['accuracy'])

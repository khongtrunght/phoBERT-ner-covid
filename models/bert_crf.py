from turtle import forward
from typing import List
import numpy as np
from sklearn.metrics import classification_report
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

import wandb
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
                 weight_decay: float = 0.0,
                 steps_per_epoch: int = None,
                 n_epochs: int = None,
                 freeze_bert: bool = False,):
        super().__init__()
        self.save_hyperparameters()
        # self.num_labels = num_labels
        # self.ignore_labels = config.ignore_labels
        self.hparams.ignore_labels = -100
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs

        # Load MOdel and get it body

        self.config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=num_labels)
        self.config.output_hidden_states = True
        self.config.output_attentions = True
        self.model = AutoModel.from_pretrained(
            model_name_or_path, config=self.config)
        if freeze_bert:
            for param in self.model.parameters():
                param.requires_grad = False
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
          - "loss"  get rid of (if `labels` is not `None`)
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
            output_tags = []
            for seq_logits, seq_labels, seq_mask in zip(bert_feats, labels, mask):
                # Index logits and labels using prediction mask to pass only the
                # first subtoken of each word to CRF.
                assert sum(seq_mask) != 0
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                seq_labels = seq_labels[seq_mask].unsqueeze(0)
                loss -= self.crf(seq_logits, seq_labels,
                                 reduction='token_mean')
                tags = self.crf.decode(seq_logits)
                output_tags.append(tags[0])

            loss /= batch_size
            outputs['loss'] = loss
            outputs['y_pred'] = output_tags
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

        if self.steps_per_epoch is not None and self.n_epochs is not None:
            # if we get info to build a scheduler, do it
            num_warmup_steps = (self.steps_per_epoch * self.n_epochs) // 100
            num_train_steps = self.steps_per_epoch * self.n_epochs - num_warmup_steps
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_steps
            )
            scheduler = {"scheduler": scheduler,
                         "interval": "step", "frequency": 1}
            return [optimizer], [scheduler]
        else:
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
            prediction_mask=prediction_mask,
            labels=labels
        )

        loss = outputs['loss']
        y_pred = outputs['y_pred']
        true_labels_step = self.post_process(labels)

        for index, sent in enumerate(y_pred):
            assert len(sent) == sum(prediction_mask[index])

        return {"preds": y_pred, "labels": true_labels_step, 'val_loss': loss}

    def test_step(self, test_batch, batch_idx):
        x = test_batch
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

        # TODO : add text input and print prediction in test epoch end
        return {"preds": y_pred, "labels": true_labels_step}

    def test_epoch_end(self, outputs):
        preds = list(itertools.chain(*[o['preds'] for o in outputs]))
        labels = list(itertools.chain(*[o['labels'] for o in outputs]))

        # results = compute_metrics((preds, labels), self.label_name_list,
        #                           return_logits=False)

        predictions, labels = preds, labels

        y_pred = [p for prediction, label in zip(predictions, labels) for (
            p, l) in zip(prediction, label) if l not in (-100, 20)]
        y_true = [l for prediction, label in zip(predictions, labels) for (
            p, l) in zip(prediction, label) if l not in (-100, 20)]

        report = classification_report(y_true, y_pred, labels=range(
            len(self.label_name_list) - 1), target_names=self.label_name_list[:-1], zero_division=0.0)
        liststr = report.split('\n')
        listtag = liststr[2:-5]
        listavg = liststr[-4:-1]
        item_names, precision, recall, f1, count = [], [], [], [], []
        for val in listtag:
            items = val.split()
            item_names.append(items[0])
            precision.append(items[1])
            recall.append(items[2])
            f1.append(items[3])
            count.append(items[4])

        for val in listavg:
            items = val.split()
            item_names.append('#'+items[0]+'_'+items[1])
            precision.append(items[2])
            recall.append(items[3])
            f1.append(items[4])
            count.append(items[5])

        metrics = [item_names, precision, recall, f1, count]
        self.logger.experiment.log({"Test Metrics": wandb.Table(
            columns=['Name', 'Precision', 'Recall', 'F1', 'Count'],
            data=list(map(list, zip(*metrics))))})

    def validation_epoch_end(self, outputs):
        loss = torch.stack([o['val_loss'] for o in outputs]).mean()
        preds = list(itertools.chain(*[o['preds'] for o in outputs]))
        labels = list(itertools.chain(*[o['labels'] for o in outputs]))

        results = compute_metrics((preds, labels), self.label_name_list,
                                  return_logits=False)

        predictions, labels = preds, labels

        y_pred = [p for prediction, label in zip(predictions, labels) for (
            p, l) in zip(prediction, label) if l not in (-100, 20)]
        y_true = [l for prediction, label in zip(predictions, labels) for (
            p, l) in zip(prediction, label) if l not in (-100, 20)]

        report = classification_report(y_true, y_pred, labels=range(
            len(self.label_name_list) - 1), target_names=self.label_name_list[:-1], zero_division=0.0)
        liststr = report.split('\n')
        listtag = liststr[2:-5]
        listavg = liststr[-4:-1]
        item_names, precision, recall, f1, count = [], [], [], [], []
        for val in listtag:
            items = val.split()
            item_names.append(items[0])
            precision.append(items[1])
            recall.append(items[2])
            f1.append(items[3])
            count.append(items[4])

        for val in listavg:
            items = val.split()
            item_names.append('#'+items[0]+'_'+items[1])
            precision.append(items[2])
            recall.append(items[3])
            f1.append(items[4])
            count.append(items[5])

        metrics = [item_names, precision, recall, f1, count]
        self.logger.experiment.log({"Metrics": wandb.Table(
            columns=['Name', 'Precision', 'Recall', 'F1', 'Count'],
            data=list(map(list, zip(*metrics))))})

        # # Precision Recall
        # self.logger.experiment.log({"PR": wandb.plots.precision_recall(y_true, y_pred, labels=range(
        #     len(self.label_name_list) - 1))})

        self.log('precision', results['precision'])
        self.log('recall', results['recall'])
        self.log('f1', results['f1'])
        self.log('accuracy', results['accuracy'], prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)

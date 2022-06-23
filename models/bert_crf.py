from turtle import forward
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

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """forward function of model

        Args:
            input_ids (tensor.IntegerTensor, optional): input_ids after tokenize and convert to ids. Defaults to None.
            attention_mask (tensor.ByteTensor, optional): inpout attention_mask. Defaults to None.
            labels (tensor.IntegerTensor, optional): labels when training. Defaults to None.

        Returns:
            TokenClassifierOuput: output of model
        """
        labels_clone = labels.clone()
        bert_feats, bert_out = self._get_bert_features(
            input_ids, attention_mask)

        # For training
        if labels is not None:
            # ignore_labels = -100
            padding_mask = labels.ne(self.hparams.ignore_labels)
            # padding_mask[:, 0] = 1  # first token must be in the mask
            padding_mask = padding_mask[:, 1:]

            # can not use -100 in tags, must convert to 0 and ignore
            # labels = labels.clamp(min=0)

            # ------------------------------------- label O is 20 --------------------------------------------------
            labels_clone[labels == self.hparams.ignore_labels] = 20
            # try to remove mask
            # crf_out = self.crf.decode(bert_feats)
            crf_out = self.crf.decode(bert_feats[:, 1:, :], mask=padding_mask)
            # crf_out = [torch.tensor(i) for i in crf_out]
            # crf_loss = self.crf(bert_feats, labels)
            crf_loss = - \
                self.crf(bert_feats[:, 1:, :],
                         tags=labels_clone[:, 1:], mask=padding_mask)

        #  For testing
        else:
            padding_mask = attention_mask.ne(0)
            # crf_out = self.crf.decode(bert_feats)
            # crf_out = [torch.tensor(i) for i in crf_out]
            crf_out = self.crf.decode(bert_feats, mask=padding_mask)
            crf_loss = None

        return TokenClassifierOutput(
            loss=crf_loss,
            logits=crf_out,
            hidden_states=bert_out.hidden_states,
            attentions=bert_out.attentions,
        )

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

        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=200
        )
        scheduler = {"scheduler": scheduler,
                     "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        x = train_batch
        outputs = self.forward(**x)
        loss = outputs['loss']
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        x = val_batch
        outputs = self.forward(**x)
        loss = outputs['loss']
        true_labels_step = [
            [label for label in sent_label[1:]
                if label != self.hparams.ignore_labels]
            for sent_label in x['labels'].tolist()
        ]
        return {'val_loss': loss, "preds": outputs['logits'], "labels": true_labels_step, "attentions": outputs['attentions']}

    def validation_epoch_end(self, outputs):
        # preds = np.concatenate([o['preds'] for o in outputs], axis=0)
        # labels = np.concatenate([o['labels'] for o in outputs], axis=0)
        preds = list(itertools.chain(*[o['preds'] for o in outputs]))
        labels = list(itertools.chain(*[o['labels'] for o in outputs]))

        compute_metrics((preds, labels), self.label_name_list,
                        return_logits=False)

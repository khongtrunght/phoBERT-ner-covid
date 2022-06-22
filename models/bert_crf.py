from turtle import forward
from transformers import AutoConfig, AutoModel
from torch import nn
from torchcrf import CRF
from transformers.modeling_outputs import TokenClassifierOutput


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


class CustomNERCRF(nn.Module):
    def __init__(self, checkpoint, num_labels, ignore_labels=-100):
        super().__init__()
        self.num_labels = num_labels
        self.ignore_labels = ignore_labels

        # Load MOdel and get it body
        self.model = AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(
            checkpoint, output_attentions=True, output_hidden_states=True))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

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

        bert_feats, bert_out = self._get_bert_features(
            input_ids, attention_mask)

        # For training
        if labels is not None:
            # ignore_labels = -100
            padding_mask = labels.eq(self.ignore_labels)

            # can not use -100 in tags, must convert to 0 and ignore
            labels = labels.clamp(min=0)
            crf_out = self.crf.decode(bert_feats, mask=padding_mask)

            crf_loss = - self.crf(bert_feats, tags=labels, mask=padding_mask)

        #  For testing
        else:
            padding_mask =  attention_mask.eq(0)
            crf_out = self.crf.decode(bert_feats, mask=padding_mask)
            crf_loss = None

        return TokenClassifierOutput(
            loss=crf_loss,
            logits=crf_out,
            hidden_states=bert_out.hidden_states,
            attentions=bert_out.attentions,
        )


from dataclasses import dataclass
from pyparsing import col
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments
import wandb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import yaml

from dataset.covid19_dataset import COVID19Dataset
from dataset.hf_tokenize import HFTokenizer
from models.bert_crf import CustomNERCRF, RuleProcessor
import pytorch_lightning as pl

from utils import compute_metrics

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

parser = ArgumentParser()
parser.add_argument('--config', '-c',
                    dest='filename',
                    metavar='FILE',
                    default='configs/bert_crf.yaml')

parser.add_argument('--name', '-n', type=str, default=None)

args = parser.parse_args()
with open(args.filename, 'r') as f:
    try:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        exc.print_exc()


model_n_version = config['model_params']['model_n_version']
max_epochs = config['trainer_params']['max_epochs']
learning_rate = config['trainer_params']['learning_rate']
batch_size = config['trainer_params']['batch_size']
model_root_dir = config['model_params']['model_root_dir']


hf_pretrained_tokenizer_checkpoint = config['model_params']['model_pretrain_path']
hf_dataset = COVID19Dataset()

hf_preprocessor = HFTokenizer.init_vf(
    hf_pretrained_tokenizer_checkpoint=hf_pretrained_tokenizer_checkpoint)

tokenized_datasets = hf_dataset.dataset.map(
    hf_preprocessor.tokenize_and_align_labels, batched=True)


@dataclass
class CustomDataCollator(DataCollatorForTokenClassification):
    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        prediction_mask_name = "prediction_mask" if "prediction_mask" in features[0].keys(
        ) else "prediction_masks"
        labels = [feature[label_name]
                  for feature in features] if label_name in features[0].keys() else None
        prediction_mask = [feature[prediction_mask_name]
                           for feature in features] if prediction_mask_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
            batch[prediction_mask_name] = [
                list(prediction_mask) + [False] * (sequence_length - len(prediction_mask)) for prediction_mask in prediction_mask
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]
            batch[prediction_mask_name] = [
                 [False] * (sequence_length - len(prediction_mask)) + list(prediction_mask) for prediction_mask in prediction_mask
            ]

        batch = {k: torch.tensor(v, dtype=torch.int64)
                 for k, v in batch.items()}
        batch[prediction_mask_name] = batch[prediction_mask_name].to(
            torch.bool)
        return batch


# data_collator = DataCollatorForTokenClassification(hf_preprocessor.tokenizer)
data_collator = CustomDataCollator(hf_preprocessor.tokenizer)


# train_dataloader = DataLoader(
#     tokenized_datasets['train'].select(range(4)),
#     shuffle=True,
#     batch_size=config['trainer_params']['batch_size'],
#     collate_fn=data_collator
# )

# eval_dataloader = DataLoader(
#     tokenized_datasets['validation'].select(range(4)),
#     shuffle=False,
#     batch_size=config['trainer_params']['batch_size'],
#     collate_fn=data_collator
# )


model = CustomNERCRF(config['model_params']['model_pretrain_path'],
                     num_labels=len(hf_dataset.labels),
                     learning_rate=config['trainer_params']['learning_rate'],
                     warmup_steps=0,
                     weight_decay=0.01,
                     steps_per_epoch=len(
                         tokenized_datasets['train']) // config['trainer_params']['max_epochs'],  # dang test
                     n_epochs=config['trainer_params']['max_epochs'],
                     freeze_bert=config['model_params']['freeze_bert'],
                     )

rule_processor = RuleProcessor()
model.init_crf_transitions(rule_processor, labels_list=hf_dataset.labels)

args_hf = TrainingArguments(
    f"test-ner",
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=max_epochs,
    weight_decay=0.01,
)


trainer_hf = Trainer(
    model,
    args_hf,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=hf_preprocessor.tokenizer,
    compute_metrics=lambda p: compute_metrics(
        p=p, label_list=hf_dataset.labels,
        return_logits=config['model_params']['model_n_version'] == 'bert-normal')
)


train_dataloader = trainer_hf.get_train_dataloader()

eval_dataloader = trainer_hf.get_eval_dataloader()

test_dataloader = trainer_hf.get_test_dataloader(
    test_dataset=tokenized_datasets["test"])


wandb_logger = WandbLogger(project='NLP NER-CRF COVID-19',
                           name=args.name)

lr_monitor = LearningRateMonitor()

trainer = pl.Trainer(
    max_epochs=config['trainer_params']['max_epochs'],
    logger=wandb_logger,
    accelerator='gpu',
    devices=1,
    callbacks=[EarlyStopping(monitor="val_loss", mode="min"),
               lr_monitor],
)

trainer.fit(model, train_dataloader, eval_dataloader)

trainer.test(model, dataloaders=test_dataloader)

trainer.save_checkpoint(f'BestModel{args.name}.pth')
wandb.save(f'BestModel{args.name}.pth')

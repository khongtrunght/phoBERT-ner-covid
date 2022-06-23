
from pyparsing import col
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

import yaml

from dataset.covid19_dataset import COVID19Dataset
from dataset.hf_tokenize import HFTokenizer
from models.bert_crf import CustomNERCRF, RuleProcessor
import pytorch_lightning as pl

from utils import compute_metrics

parser = ArgumentParser()
parser.add_argument('--config', '-c',
                    dest='filename',
                    metavar='FILE',
                    default='configs/bert_crf.yaml')

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


data_collator = DataCollatorForTokenClassification(hf_preprocessor.tokenizer)


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
                     )


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
    train_dataset=tokenized_datasets["train"].select(range(40)),
    eval_dataset=tokenized_datasets["validation"].select(range(40)),
    data_collator=data_collator,
    tokenizer=hf_preprocessor.tokenizer,
    compute_metrics=lambda p: compute_metrics(
        p=p, label_list=hf_dataset.labels,
        return_logits=config['model_params']['model_n_version'] == 'bert-normal')
)


class ComputeMetricCallback(pl.Callback):
    def on_validation_batch_end(trainer, module, outputs):
        vacc = outputs['']


compute_metric_hook = pl.Callback()


train_dataloader = trainer_hf.get_train_dataloader()

eval_dataloader = trainer_hf.get_eval_dataloader()


trainer = pl.Trainer(
    max_epochs=config['trainer_params']['max_epochs'],
)

trainer.fit(model, train_dataloader, eval_dataloader)

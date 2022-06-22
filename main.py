from argparse import ArgumentParser
import numpy as np
from dataset.hf_tokenize import HFTokenizer
from transformers import AutoConfig, AutoModelForTokenClassification, DataCollatorForTokenClassification, AutoTokenizer, TrainingArguments, Trainer
from dataset.covid19_dataset import COVID19Dataset
import os
import yaml
from models.bert_crf import CustomNERCRF, RuleProcessor

from utils import compute_metrics, metric

parser = ArgumentParser()
parser.add_argument('--config', '-c',
                    dest='filename',
                    metavar='FILE',
                    default='configs/bert_normal.yaml')

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


# dataset
hf_pretrained_tokenizer_checkpoint = config['model_params']['model_pretrain_path']
hf_dataset = COVID19Dataset()

hf_preprocessor = HFTokenizer.init_vf(
    hf_pretrained_tokenizer_checkpoint=hf_pretrained_tokenizer_checkpoint)

tokenized_datasets = hf_dataset.dataset.map(
    hf_preprocessor.tokenize_and_align_labels, batched=True)


data_collator = DataCollatorForTokenClassification(hf_preprocessor.tokenizer)


# model
if config['model_params']['model_n_version'] == 'bert-normal':
    hf_model = AutoModelForTokenClassification.from_pretrained(
        config['model_params']['model_pretrain_path'],
        num_labels=len(hf_dataset.labels))
    hf_model.config.id2label = hf_dataset.id2label
    hf_model.config.label2id = hf_dataset.label2id
elif config['model_params']['model_n_version'] == 'bert-crf':
    config_bert = AutoConfig.from_pretrained(
        config['model_params']['model_pretrain_path'],
        num_labels=len(hf_dataset.labels),
        id2label=hf_dataset.id2label,
        label2id=hf_dataset.label2id)

    # hf_model = CustomNERCRF(checkpoint=config['model_params']['model_pretrain_path'],
    #                         num_labels=len(hf_dataset.labels),

    config.device = "cuda"

    hf_model = CustomNERCRF.from_pretrained(
        config['model_params']['model_pretrain_path'],
        config=config_bert,
        cache_dir=config['model_params']['cache_dir'])

    rule_processor = RuleProcessor()

    hf_model.init_crf_transitions(rule_processor,
                                  labels_list=hf_dataset.labels,
                                  )


args = TrainingArguments(
    f"test-ner",
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=max_epochs,
    weight_decay=0.01,
)


trainer = Trainer(
    hf_model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=hf_preprocessor.tokenizer,
    compute_metrics=lambda p: compute_metrics(
        p=p, label_list=hf_dataset.labels,
        return_logits=config['model_params']['model_n_version'] == 'bert-normal')
)


trainer.train()
trainer.evaluate()

# Predictions on test dataset and evaluation

predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [hf_dataset.labels[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

true_labels = [
    [hf_dataset.labels[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
print(results)

out_dir = os.path.expanduser(model_root_dir) + "/" + model_n_version
trainer.save_model(out_dir)

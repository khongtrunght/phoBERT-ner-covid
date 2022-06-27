import numpy as np
from sklearn.metrics import classification_report
from dataset.hf_tokenize import HFTokenizer
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification, AutoTokenizer, TrainingArguments, Trainer
from dataset.covid19_dataset import COVID19Dataset
import os
from datasets import load_metric


metric = load_metric("seqeval")


def compute_metrics(p, label_list, return_logits=True):
    """Compute metric in traning phase
    Args:
        p (tuple): (predictions, labels) in idx type
        label_list (list[str]): label name list in str type
    Returns:
        dict: dictionary of metric
    """
    predictions, labels = p
    if return_logits:
        predictions = np.argmax(predictions, axis=2)

    y_pred = [p for prediction, label in zip(predictions, labels) for (
        p, l) in zip(prediction, label) if l not in (-100, 20)]
    y_true = [l for prediction, label in zip(predictions, labels) for (
        p, l) in zip(prediction, label) if l not in (-100, 20)]

    report = classification_report(y_true, y_pred, labels=range(
        len(label_list) - 1), target_names=label_list[:-1], zero_division=0.0)
    print(report)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(
        predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


model_n_version = "sroie2019v1"
max_epochs = 50
learning_rate = 2e-5
batch_size = 16
model_root_dir = "~/.phoner_covid19/models/hf/"

hf_pretrained_tokenizer_checkpoint = "vinai/phobert-base"
hf_dataset = COVID19Dataset()

hf_model = AutoModelForTokenClassification.from_pretrained(
    "vinai/phobert-base",
    num_labels=len(hf_dataset.labels))

hf_model.config.id2label = hf_dataset.id2label
hf_model.config.label2id = hf_dataset.label2id

hf_preprocessor = HFTokenizer.init_vf(
    hf_pretrained_tokenizer_checkpoint=hf_pretrained_tokenizer_checkpoint)

tokenized_datasets = hf_dataset.dataset.map(
    hf_preprocessor.tokenize_and_align_labels, batched=True)


data_collator = DataCollatorForTokenClassification(hf_preprocessor.tokenizer)


args = TrainingArguments(
    f"test-ner",
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    report_to='wandb',
)

trainer = Trainer(
    hf_model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=hf_preprocessor.tokenizer,
    compute_metrics=lambda p: compute_metrics(
        p=p, label_list=hf_dataset.labels)
)

trainer.train()
trainer.evaluate()

# test_dataloader = trainer.get_test_dataloader(tokenized_datasets["test"])
# trainer.predict(test_dataloader)

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

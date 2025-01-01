from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from utils import load_and_preprocess_data, split_dataset, create_dataset_dict, tokenize_data
from sklearn.metrics import accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

def train_model(file_path, checkpoint):
    df, label_encoder = load_and_preprocess_data(file_path)
    train_df, val_df, test_df = split_dataset(df)
    dataset_dict = create_dataset_dict(train_df, val_df, test_df)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenized_datasets = tokenize_data(dataset_dict, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=len(label_encoder.classes_))

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("banglanews_headline_classifier")
    return trainer, model, tokenizer, label_encoder

if __name__ == "__main__":
    file_path = "banglanewsheadline.csv"
    checkpoint = "csebuetnlp/banglabert"
    trainer, model, tokenizer, label_encoder = train_model(file_path, checkpoint)
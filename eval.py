from train import train_model
from sklearn.metrics import classification_report
import numpy as np

def evaluate_model(file_path, checkpoint):
    trainer, model, tokenizer, label_encoder = train_model(file_path, checkpoint)
    tokenized_datasets = trainer.train_dataset

    predictions = trainer.predict(tokenized_datasets["test"])
    logits = predictions.predictions
    labels = predictions.label_ids
    predicted_labels = np.argmax(logits, axis=-1)

    target_names = label_encoder.classes_
    report = classification_report(labels, predicted_labels, target_names=target_names)
    print(report)

if __name__ == "__main__":
    file_path = "banglanewsheadline.csv"
    checkpoint = "banglanews_headline_classifier"
    evaluate_model(file_path, checkpoint)

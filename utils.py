import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    label_encoder = LabelEncoder()
    df['NewsType'] = label_encoder.fit_transform(df['NewsType'])
    return df, label_encoder

def split_dataset(df):
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['NewsType'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['NewsType'], random_state=42)
    return train_df, val_df, test_df

def create_dataset_dict(train_df, val_df, test_df):
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    return DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})

def tokenize_data(dataset_dict, tokenizer):
    def tokenize_function(example):
        return tokenizer(
            example["Headline"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("NewsType", "labels")
    tokenized_datasets = tokenized_datasets.remove_columns(["Headline", "__index_level_0__"])
    tokenized_datasets.set_format("torch")
    return tokenized_datasets
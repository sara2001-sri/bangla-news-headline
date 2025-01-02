# bangla-news-headline
Bangla News Headline Classification using a fine-tuned Bangla-BERT model. We utilized a labeled dataset of Bangla news headlines to train the model for categorizing headlines into predefined classes. The project includes data preprocessing, model fine-tuning with Hugging Face Transformers.
Table of Contents



Features

BanglaBERT Fine-Tuning: The pre-trained BanglaBERT model is fine-tuned on a labeled dataset of Bangla news headlines.

Normalization: Bangla text normalization ensures consistency in text preprocessing.

Training and Evaluation: Split data into training, validation, and test sets for robust evaluation.

Custom Headline Prediction: Predict the category of any custom Bangla news headline.

Visualization: Analyze loss curves and class distributions through visual graphs.

How It Works

Data Preprocessing:

Headlines are normalized using bkit to clean and standardize Bangla text.

News categories are label-encoded to prepare them for classification.

Tokenization:

Headlines are tokenized using the pre-trained BanglaBERT tokenizer.

Model Training:

The BanglaBERT model is fine-tuned using the labeled dataset.

Evaluation:

Model performance is evaluated on a test set with metrics such as accuracy.

Custom Predictions:

The trained model predicts the category of a user-provided Bangla news headline.

Installation

Prerequisites:

Python 3.8+

Pip package manager

Steps:

Clone the repository:

Install the required packages:

Download the dataset and place it in the root directory.

Dataset Name: Bengali_News_Headline.csv

Usage

Training the Model:
Run the training script to fine-tune BanglaBERT on your dataset:

python train.py

Evaluating the Model:
Evaluate the model performance:

python eval.py

Predicting Custom Headlines:
Use the following command to predict a news category:

python predict.py

Enter a Bangla headline when prompted, and the model will output its category.

Project Structure

.
├── data/
│   └── Bengali_News_Headline.csv    # Dataset
├── utils.py                           # Utility functions for normalization and preprocessing
├── train.py                           # Script for training the model
├── eval.py                            # Script for evaluating the model
├── predict.py                         # Script for predicting custom headlines
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation

Sample Predictions

Run the predict.py script and input a sample Bangla news headline, such as:

Input:

বাংলাদেশ ক্রিকেট দলের নতুন কোচ ঘোষণা।

Output:

Predicted News Type: খেলাধুলা

Acknowledgments

This project uses the following tools and datasets:

BanglaBERT: Pre-trained language model for Bangla.

Hugging Face Transformers Library: Model and tokenizer utilities.

bkit: Bangla text normalization toolkit.

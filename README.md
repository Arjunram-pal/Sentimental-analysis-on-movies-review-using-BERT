# Sentiment Analysis on Movie Reviews using BERT

![Sentiment Analysis](https://img.shields.io/badge/Sentiment%20Analysis-BERT-blue)

This project aims to perform sentiment analysis on movie reviews using the BERT (Bidirectional Encoder Representations from Transformers) model. The goal is to build a machine learning model that can accurately predict the sentiment (positive or negative) of a movie review text.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Model](#model)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Sentiment analysis is the process of determining the sentiment expressed in a piece of text. In this project, we utilize the powerful BERT model, which is a transformer-based model known for its exceptional performance on various natural language processing tasks, including sentiment analysis.

## Prerequisites

Before running this project, ensure you have the following dependencies:

- Python 3.x
- PyTorch
- Transformers Library (Hugging Face)
- pandas
- numpy
- scikit-learn

You can install the required packages using the `requirements.txt` file provided in this repository.

```bash
pip install -r requirements.txt
```

Getting Started
To get started with the project, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies as mentioned in the Prerequisites section.
3. Download the dataset and place it in the data/ directory.
4. Run the train.py script to train the sentiment analysis model on the movie reviews dataset.
5. Once the model is trained, you can use the predict.py script to predict the sentiment of custom movie reviews.

Dataset

We used the IMDB movie reviews dataset for training and testing our model. The dataset contains movie reviews labeled as positive or negative based on their sentiment. The dataset is divided into training and testing sets.

Download the dataset from here and place it in the data/ directory before running the training script.

Model

We used the BERT model with a pre-trained language model and fine-tuned it on our movie reviews dataset. The BERT model is known for its ability to capture contextual relationships between words, making it well-suited for sentiment analysis tasks.

Usage

To train the model, run the following command:

# Sentimental-analysis-on-movies-review-using-BERT

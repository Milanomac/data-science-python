# Natural-Language-Processing-with-Disaster-Tweets

## Abstract

The project hosted by Kaggle, is aimed at analyzing data present in individuals' online footprints, particularly in Twitter. The goal is to discern tweets depicting real disasters from others. Natural Language Processing (NLP) is used for text and sentiment analysis. By identifying emergency-related messages promptly through NLP models, the potential benefit of this project is to expedite response times in critical situations, potentially saving lives. Hosted on Kaggle, the competition relies on a training set of 10,000 hand-classified tweets. Various models were explored, including basic count vectorizer with Ridge regression (RidgeClassifier), Logistic regression + TF-IDF vectorization, multinomial naïve-bayes method, as well as Google Universal Sentence Encoder. Among these, Logistic Regression model performed the best, with a training f1 score of roughly 0.8255.

## Dataset

https://www.kaggle.com/c/nlp-getting-started/data

## Data Pre-processing and Visulaization

The following were the steps we performed for understanding the data better

- Visualizations (Histograms, Bar charts , etc.)
- Embeddings/ Vectorisation (count vectors, TF-IDF vectorization)
- Scatter text (a special type of interactive visualization)

## Implemenation Methods

- Simple Linear Classification - Ridge Regression
- Logistic Regression
- A multinomial naïve-bayes
- Google Universal Sentence Encoder

## Acknowledgements

Notebooks in this project were used with help of:
https://github.com/MahalavanyaSriram/Natural-Language-Processing-with-Disaster-Tweets/blob/master/Jupyter%20Notebooks/google-universal-sentence-encoder.ipynb

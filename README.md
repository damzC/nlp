# nlp
This repository contains work on Natural Language Processing using both Machine Learning and Deep Learning techniques

## 1. Count based word representation and document similarity
* **Implementation File**: _Count_Based_Word_Representation_for_Document_Similarity.ipynb_

This notebook is an implementation of **Count Based** Word Representations and Text/Document Similarity. It demonstrates the use of **Bag of Words**, **Count Vectorizer** and **TF-IDF** concepts for word representation, which lie at the core of NLP. It also showcases how these concepts can be leveraged to find *similarity between documents*

## 2. Knowledge Graph Construction for NLP Applications
* **Implementation File**: _KnowledgeGraphsInNLP.ipynb_

In the implementation below we will learn the following:

* Extracting meaningful ***information*** from unstructured text data
* ***Constructing Knowledge Graphs*** from the extracted information

First, we will understand the Knowledge Graph construction process on unstructured text data, with a simple example

Consequently, we will employ the same process on ***unstructured wikipedia data***

## 3. Sentiment Analysis
Sentiment Analysis using **Machine Learning**: _Sentiment_Analyzer_Machine_Learning.ipynb_

Sentiment Analysis using **Deep Learning**: _Sentiment_Analyzer_Deep_Learning.ipynb_

These notebook explain the problem of Sentiment Analysis, some of the popular datasets available, approaches to solve this problem and finally a step-by-step guide to solve this problem using both Machine Learning and Deep Learning.

**Definition**: Sentiment Analysis is the NLP task of computationally identifying the opinion or sentiment (*positive*, *negative*, or *neutral*) expressed in a text.

**Popular Data sets for Sentiment Analysis:**
* kaggle: Movie reviews on IMDB data set - 50K entries: *Longer text SA*
* Stanford data set for sentiment analysis:*5 classes*: Very positive, Positive, Neutral, Negative, Very Negative
* Amazon review data set (kaggle) - Pre-trained models available: *Short text SA* (4 million entries) - Extract the headings only

**Approaches for Sentiment Analysis:**
* Lexicon based: Senti WordNet
* NLP Tools: TextBlob, spaCy, NLTK
* Machine Learning: NB Classifier, SVM, XGB
* Deep learning: LSTMs, GRUs, seq2seq
* Sentiment Embeddings - Embeddings of words based on sentiments
* Fine-tuning over Large Language Models (like BERT, RoBERTa *etc.*)

**Sentiment Analyser using Machine Learning**

This notebook exemplifies a sample implementation of a Sentiment Analyzer using a machine learning model (Naive Bayes Calssifier in this case). The implementation shows how to:

1. Import movie review data from NLTK
2. Extract data in (X,Y) pairs for training
3. Vectorize text
4. Train Sentiment Analyser using NB Classifier
5. Show results using Confusion Matrix and Classification Report

**Sentiment Analysis using Deep Learning:**

**Pre-processing:**
1. Download the dataset (*imdb_reviews*)
2. word to index | index to word
3. Train your word embeddings | Use pre-trained embeddings
4. Embedding Matrix: Index -> Word Embedding
5. Padding (Post/Pre) - Fixed length arrays of the input (Size is the max/avg length of the reviews)

**Architecture:**
1. Input Layer (200)
2. Embedding Layer (inbuilt in keras) - Embedding Matrix / Train embeddings using keras (200X300 - Emb Dimension)
3. RNN Layer (LSTM, Bi-LSTM etc)/ Attention Layer
4. Dense Layer /  Fully Connected Layer
5. Dropout Layer
6. Optional Dense Layers
7. Softmax - Final Output

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

First, we understand the Knowledge Graph construction process on unstructured text data, with a simple example. Consequently, the same process is used on ***unstructured wikipedia data***

## 3. Sentiment Analysis
Sentiment Analysis using **Machine Learning**: _Sentiment_Analyzer_Machine_Learning.ipynb_

Sentiment Analysis using **Deep Learning**: _Sentiment_Analyzer_Deep_Learning.ipynb_

These notebook explain the problem of Sentiment Analysis, some of the popular datasets available, approaches to solve this problem and finally a step-by-step guide to solve this problem using both Machine Learning and Deep Learning.

**Definition**: Sentiment Analysis is the NLP task of computationally identifying the opinion or sentiment (*positive*, *negative*, or *neutral*) expressed in a text.

**Popular Data sets for Sentiment Analysis:**
* [IMDB movie reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews): 50K entries for *Longer text SA*
* [Stanford data set for sentiment analysis](https://huggingface.co/datasets/SetFit/sst5): *5 classes*: Very positive, Positive, Neutral, Negative, Very Negative
* [Amazon reviews data set](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews) - Pre-trained models available: *Short text SA* (4 million entries) - Extract the headings only

**Approaches for Sentiment Analysis:**
* Lexicon based: Senti WordNet
* NLP Tools: TextBlob, spaCy, NLTK
* Machine Learning: NB Classifier, SVM, XGB
* Deep learning: LSTMs, GRUs, seq2seq
* Sentiment Embeddings - Embeddings of words based on sentiments
* Fine-tuning over Large Language Models (like BERT, RoBERTa *etc.*)

**Sentiment Analyser using Machine Learning**

This notebook exemplifies a sample implementation of a Sentiment Analyzer using a machine learning model (Naive Bayes Calssifier in this case). The implementation shows how to:

* Import movie review data from NLTK
* Extract data in (X,Y) pairs for training
* Vectorize text
* Train Sentiment Analyser using NB Classifier
* Show results using Confusion Matrix and Classification Report


**Sentiment Analysis using Deep Learning:**

**Pre-processing:**
* Download the dataset (*imdb_reviews*)
* word to index | index to word
* Train your word embeddings | Use pre-trained embeddings
* Embedding Matrix: Index -> Word Embedding
* Padding (Post/Pre) - Fixed length arrays of the input (Size is the max/avg length of the reviews)

**Architecture:**
* Input Layer (200)
* Embedding Layer (inbuilt in keras) - Embedding Matrix / Train embeddings using keras (200X300 - Emb Dimension)
* RNN Layer (LSTM, Bi-LSTM etc)/ Attention Layer
* Dense Layer /  Fully Connected Layer
* Dropout Layer
* Optional Dense Layers
* Softmax - Final Output

## 4. Building a basic conversational chatbot
Implementation file: _Conversational_Chatbot_RASA.ipynb_

**Problem**: In many cases, clients do not want to share their data and since  majority of the avialable tools are cloud-based and provide software as a service so you can not run them internally in your environment and you need to send your data to the third party. 

**Solution**: With **RASA stack**, an open-source customizable AI tool there is no such issue. You can build, deploy or host Rasa internally in your server or environment with complete control over it.

Rasa comes up with 2 components:

**Rasa NLU:** NLU deals with teaching a chatbot on how to understand user inputs.

**Rasa Core:** Deals with teaching a chatbot on how to respond to user’s query.

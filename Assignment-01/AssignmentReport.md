# Text Feature Engineering Report

## Overview

This assignment builds a text processing pipeline on a real-world dataset of Flipkart product reviews for Apple iPhone 15. The dataset was scraped from public Flipkart review pages and stored in `flipkart_reviews.csv`. A total of 110 reviews were collected, with 55 reviews from the positive-first sort order and 55 reviews from the negative-first sort order, to support a balanced sentiment classification use case.

## Preprocessing

The preprocessing pipeline applied the following steps:

- converted all review text to lowercase
- removed punctuation
- tokenized review text into word-level tokens
- removed standard English stopwords while preserving negation words such as `not`, `no`, and `nor`
- lemmatized tokens using WordNet lemmatization

This preprocessing step reduced noise and improved consistency across the review corpus. Preserving negation was important because phrases such as `not good` would otherwise lose their polarity signal during stopword removal.

## Vocabulary And Features

After preprocessing, the vocabulary size was 349 unique tokens. The most frequent terms included `camera`, `battery`, `not`, `iphone`, and `phone`, showing that the review set focused heavily on hardware experience and purchase satisfaction.

Three feature engineering methods were implemented:

- One Hot Encoding using document-level binary term presence
- Bag of Words using `CountVectorizer`
- TF-IDF using `TfidfVectorizer`

The resulting feature shapes were:

- One Hot Encoding: `(110, 349)`
- Bag of Words: `(110, 344)`
- TF-IDF: `(110, 344)`

The highest mean TF-IDF terms included `camera`, `not`, `battery`, `awesome`, and `bad`. This reflects the expected TF-IDF behavior: terms that are more discriminative across reviews receive higher weight, while very common words across many reviews receive lower weight because of inverse document frequency.

## Sparse Matrix Analysis

All three text representations produced highly sparse matrices:

- One Hot Encoding sparsity: `98.27%`
- Bag of Words sparsity: `98.26%`
- TF-IDF sparsity: `98.26%`

This happens because each review contains only a small subset of the full vocabulary. Sparse matrices are memory-efficient when stored in sparse formats, but they can still become expensive in large-scale systems because the dimensionality grows quickly with vocabulary size, making storage, indexing, and model training more expensive.

## Mini Use Case: Sentiment Classification

A Logistic Regression classifier was trained twice: once using Bag of Words features and once using TF-IDF features. The dataset was split into train and test sets using stratified sampling.

Results:

- Bag of Words accuracy: `0.9643`
- Bag of Words F1 score: `0.9655`
- TF-IDF accuracy: `0.9643`
- TF-IDF F1 score: `0.9655`

On this small balanced review sample, both feature types performed equally well. This indicates that the positive and negative reviews contain strong lexical cues, so both raw counts and IDF-weighted features were sufficient for the classification task.

## Real-world Discussion

Bag of Words fails to capture semantic meaning because it ignores context, sequence, and synonym relationships. For example, `great camera` and `excellent camera` are treated as different token combinations even though they express similar sentiment. TF-IDF improves term importance weighting, but it still cannot understand meaning, sarcasm, or context.

In industry, Bag of Words is useful for simple baselines, keyword-driven classification, and fast prototype systems. TF-IDF is often a better default for document ranking, support-ticket triage, or lightweight classifiers because it reduces the effect of common terms. However, TF-IDF still struggles with domain drift, unseen language, and contextual understanding, so modern systems often move to embeddings or transformer-based models when deeper semantics are required.

## Conclusion

This assignment shows that classical text feature engineering methods remain practical for compact machine learning pipelines. One Hot Encoding is simple but limited, Bag of Words is a strong baseline, and TF-IDF gives a better notion of term importance. The dataset and notebook demonstrate how these techniques behave on real review data and how they can support a basic sentiment classification workflow.
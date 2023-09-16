import nltk
import os
import re
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import math
import argparse

class TermDocumentMatrix:
    def __init__ (self, p=20, matrix_type="vector", metric="tf"):
        self.corpus_dir = "./nasa"
        self.p = p
        self.stemmer = PorterStemmer()
        self.all_articles = self.get_article_files()
        self.matrix = None
        self.matrix_type = matrix_type
        self.metric=metric
        self.tokens = self.preprocess_text(self.all_articles)
        stemmed_docs = self.stem_text(self.tokens)
        self.tf_values = self.calculate_tf(stemmed_docs)
        self.tfidf_values = self.calculate_tfidf(self.tf_values)
        self.top_stems_tf, self.top_stems_tfidf = self.get_top_stems(self.tf_values, self.tfidf_values)
        self.top_stems = None
        self.create_matrix()
    
    def get_article_files(self):
        all_files = os.listdir(self.corpus_dir)
        txt_files = []
        for file in all_files:
            if file.endswith('.txt'):
                txt_files.append(file)

        txt_files = sorted(txt_files)
        return txt_files
    
    def preprocess_text(self, articles):
        tokenized_text = []

        for file_name in articles:
            path = os.path.join(self.corpus_dir, file_name)
            with open(path, 'r', encoding='latin-1') as file:
                text = file.read()
                cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', text)
                tokens = cleaned_text.split()
                tokenized_text.append(tokens)

        stop_file = "english.stop"
        stop_words = set()

        with open(stop_file, 'r', encoding="latin-1") as file:
            stop_words = set(file.read().split())

        filtered_tokens = []
        for tokens in tokenized_text:
            filter_mid = []
            for token in tokens:
                if token not in stop_words and token.lower() not in stop_words:
                    filter_mid.append(token)
            filtered_tokens.append(filter_mid)

        return filtered_tokens
    
    def stem_text(self, tokens):
        stems = []
        for tkn in tokens:
            mid = []
            for token in tkn:
                mid.append(self.stemmer.stem(token))
            stems.append(mid)
        return stems
    
    def calculate_tf(self, stemmed_docs):
        tf_values = []
        for doc in stemmed_docs:
            max_term_freq = max(Counter(doc).values())
            term_freq = {term: freq / max_term_freq for term, freq in Counter(doc).items()}
            tf_values.append(term_freq)
        return tf_values
    
    def calculate_tfidf(self, tf_values):
        tfidf_values = []
        num = len(self.all_articles)

        for doc in tf_values:
            tfidf_curr = {}
            for term, tf in doc.items():
                ni = sum(1 for d in tf_values if term in d)
                if ni > 0:
                    idf = math.log(num / ni)
                else:
                    idf = 0.0
                tfidf_val = tf * idf
                tfidf_curr[term] = tfidf_val
            tfidf_values.append(tfidf_curr)
        return tfidf_values
    
    def get_top_stems(self, tf_values, tfidf_values):
        top_stems_tf = []
        for doc in tf_values:
            top_terms = sorted(doc.items(), key=lambda x: x[1], reverse=True)[:self.p]
            top_stems_tf.append([term for term, _ in top_terms])

        top_stems_tfidf = []
        average_tfidf = {}
        for doc in tfidf_values:
            for term, tfidf in doc.items():
                if term in average_tfidf:
                    average_tfidf[term] += tfidf
                else:
                    average_tfidf[term] = tfidf

        for term, total_tfidf in average_tfidf.items():
            average_tfidf[term] = total_tfidf / len(self.all_articles)

        for doc in tfidf_values:
            top_terms = sorted(doc.items(), key=lambda x: x[1], reverse=True)[:self.p]
            top_stems_tfidf.append([term for term, _ in top_terms])

        return top_stems_tf, top_stems_tfidf
    
    def create_boolean_matrix(self, top_stems):
        unique_stems = set()
        for stem in top_stems:
            unique_stems.update(stem)

        term_to_index = {}
        n = len(unique_stems)
        n_docs = len(top_stems)
        matrix = np.zeros((n, n_docs), dtype=int)

        for index, stem in enumerate(top_stems):
            for term in stem:
                term_index = term_to_index.get(term, -1)
                if term_index == -1:
                    term_to_index[term] = len(term_to_index)
                    term_index = term_to_index[term]
                matrix[term_index, index] = 1

        return matrix
    
    def create_vector_matrix(self, top_stems):
        unique_stems = set()
        for stem in top_stems:
            unique_stems.update(stem)

        n = len(unique_stems)
        n_docs = len(top_stems)
        self.matrix = np.zeros((n, n_docs), dtype=float)
        idf_values = {}

        doc_freq = Counter()
        max_term_freq = Counter()
        term_to_index = {}

        for index, stem in enumerate(top_stems):
            term_freq = Counter(stem)
            max_freq = max(term_freq.values())

            for term, freq in term_freq.items():
                doc_freq[term] += 1
                max_term_freq[index] = max(max_term_freq[index], freq)

        for term, df in doc_freq.items():
            idf_values[term] = math.log(n_docs / df)

        for index, stem in enumerate(top_stems):
            term_freq = Counter(stem)
            for term, freq in term_freq.items():
                term_index = term_to_index.get(term, -1)
                if term_index == -1:
                    term_to_index[term] = len(term_to_index)
                    term_index = term_to_index[term]
                
                tf = freq / max_term_freq[index]
                idf = idf_values.get(term, 0.0)
                self.matrix[term_index, index] = tf * idf

        return self.matrix
    
    def create_matrix(self):
        if self.matrix_type == "vector":
            if self.metric == "tf":
                self.top_stems = self.top_stems_tf
                self.matrix = self.create_vector_matrix(self.top_stems_tf)
            elif self.metric == "tfidf":
                self.top_stems = self.top_stems_tfidf
                self.matrix = self.create_vector_matrix(self.top_stems_tfidf)
        elif self.matrix_type == "boolean":
            if self.metric == "tf":
                self.top_stems = self.top_stems_tf
                self.matrix = self.create_boolean_matrix(self.top_stems_tf)
            elif self.metric == "tfidf":
                self.top_stems = self.top_stems_tfidf
                self.matrix = self.create_boolean_matrix(self.top_stems_tfidf)
        else:
            raise ValueError("Invalid matrix type")
        
        return self.matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Term-Document Matrix")
    parser.add_argument("-p", type=int, default=20, help="Number of top terms")
    parser.add_argument("--type", type=str, default="vector", choices=["vector", "boolean"], help="Matrix type")
    parser.add_argument("--metric", type=str, default="tf", choices=["tf", "tfidf"], help="Top Stems using TF or TF-IDF")
    args = parser.parse_args()

    term_matrix = TermDocumentMatrix(p=args.p, matrix_type=args.type, metric=args.metric)
    matrix = term_matrix.matrix

    print(matrix)
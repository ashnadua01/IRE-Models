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
from createTermDocumentMatrix import TermDocumentMatrix


class VectorQuery:
    def __init__(self, query, p=20, metric="tf", matrix_type="vector"):
        self.top_stems = None
        self.stemmer = PorterStemmer()
        self.matrix = None
        self.p = p
        self.matrix_type = matrix_type
        self.metric = metric
        self.top_stems = None
        self.query=query
        self.create_matrix()
        self.vector_query = self.query_representation()

    def create_matrix(self):
        tdClass = TermDocumentMatrix(
            p=self.p, matrix_type=self.matrix_type, metric=self.metric)
        self.matrix = tdClass.matrix
        self.top_stems = tdClass.top_stems
        self.all_articles = tdClass.all_articles

        self.term_to_index = {}

        for index, stem in enumerate(self.top_stems):
            for term in stem:
                term_index = self.term_to_index.get(term, -1)
                if term_index == -1:
                    self.term_to_index[term] = len(self.term_to_index)
                    term_index = self.term_to_index[term]

    def query_representation(self):
        query_terms = self.query.lower().split()
        stop_words = set()

        stop_file = "english.stop"
        with open(stop_file, 'r', encoding="latin-1") as file:
            stop_words = set(file.read().split())

        filtered_query_terms = []
    
        for term in query_terms:
            stemmed_term = self.stemmer.stem(term)
            if stemmed_term not in stop_words:
                filtered_query_terms.append(stemmed_term)

        query_vector = np.zeros((self.matrix.shape[0],), dtype=float)
        term_freq_in_query = Counter(filtered_query_terms)

        num_docs = len(self.top_stems)
        doc_freq = Counter()
        max_term_freq = Counter()

        for index, stem in enumerate(self.top_stems):
            term_freq = Counter(stem)
            max_freq = max(term_freq.values())

            for term, freq in term_freq.items():
                doc_freq[term] += 1
                max_term_freq[index] = max(max_term_freq[index], freq)

        idf_values = {}

        for term, df in doc_freq.items():
            idf_values[term] = math.log(num_docs / df)
        
        for index, stem in enumerate(self.top_stems):
            term_freq = Counter(stem)
            
            for term, freq in term_freq.items():
                term_index = self.term_to_index.get(term, -1)
                if term_index == -1:
                    term_index = len(self.term_to_index)
                    self.term_to_index[term] = term_index

        for term, tf in term_freq_in_query.items():
            term_index = self.term_to_index.get(term, -1)
            if term_index != -1 and term_index < len(query_vector):
                idf = idf_values.get(term, 0.0)
                query_vector[term_index] = tf * idf
        return query_vector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector Representation of Query")
    parser.add_argument("-q", type=str, help="Query")
    parser.add_argument("-p", type=int, default=20, help="Number of top terms")
    parser.add_argument("--type", type=str, default="vector", choices=["vector", "boolean"], help="Matrix type")
    parser.add_argument("--metric", type=str, default="tf", choices=["tf", "tfidf"], help="Top Stems using TF or TF-IDF")
    args = parser.parse_args()

    bq = VectorQuery(p=args.p, matrix_type=args.type, metric=args.metric, query=args.q)
    rep = bq.vector_query
    print(rep)
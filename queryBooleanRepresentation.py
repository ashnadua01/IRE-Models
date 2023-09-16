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

class BooleanQuery:
    def __init__(self, query, p=20, metric="tf", matrix_type="boolean"):
        self.top_stems = None
        self.stemmer = PorterStemmer()
        self.matrix = None
        self.p = p
        self.matrix_type = matrix_type
        self.metric = metric
        self.top_stems = None
        self.query=query
        self.create_matrix()
        self.bool_query = self.query_representation()

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

        bool_op = ["and", "or", "not"]
        query_terms_stemmed = []

        for term in query_terms:
            if term not in stop_words or term in bool_op:
                if term not in bool_op:
                    term = self.stemmer.stem(term)
                    query_terms_stemmed.append(term)
                else:
                    query_terms_stemmed.append(term)

        query_rep = []
        for term in query_terms_stemmed:
            if term not in bool_op:
                term_index = self.term_to_index.get(term, -1)
                if term_index != -1:
                    query_rep.append(self.matrix[term_index])
                else:
                    query_rep.append(np.zeros(self.matrix.shape[1], dtype=int))
            else:
                query_rep.append(term)

        return query_rep

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boolean Representation of Query")
    parser.add_argument("-q", type=str, help="Query")
    parser.add_argument("-p", type=int, default=20, help="Number of top terms")
    parser.add_argument("--type", type=str, default="boolean", choices=["vector", "boolean"], help="Matrix type")
    parser.add_argument("--metric", type=str, default="tf", choices=["tf", "tfidf"], help="Top Stems using TF or TF-IDF")
    args = parser.parse_args()

    bq = BooleanQuery(p=args.p, matrix_type=args.type, metric=args.metric, query=args.q)
    rep = bq.bool_query
    print(rep)
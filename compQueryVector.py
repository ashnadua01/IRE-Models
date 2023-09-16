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
from queryVectorRepresentation import VectorQuery
from similarityMeasure import SimilarityMeasure

class VectorModel:
    def __init__(self, query, p=20, metric="tf", matrix_type="vector", n=15):
        self.top_stems = None
        self.matrix = None
        self.query = query
        self.p = p
        self.matrix_type = matrix_type
        self.metric = metric
        self.stemmer = PorterStemmer()
        self.n = n
        self.vector_query_compare()

    def calculate_cosine(self):
        sm = SimilarityMeasure(query=self.vector_query, matrix=self.matrix)
        similarities = sm.cosine_similarity_calculate()
        result = []
        for index, similarity in similarities[:self.n]:
            if similarity != 0:
                result.append((self.all_articles[index], similarity))

        return result
    
    def vector_query_compare(self):
        bq = VectorQuery(p=self.p, matrix_type=self.matrix_type, metric=self.metric, query=self.query)
        self.matrix = bq.matrix
        self.vector_query = bq.vector_query
        self.top_stems = bq.top_stems
        self.term_to_index = bq.term_to_index
        self.all_articles = bq.all_articles
        self.result = self.calculate_cosine()
        return self.result
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boolean Model")
    parser.add_argument("-q", type=str, help="Query")
    parser.add_argument("-p", type=int, default=20, help="Number of top terms")
    parser.add_argument("--type", type=str, default="boolean", choices=["vector", "boolean"], help="Matrix type")
    parser.add_argument("--metric", type=str, default="tf", choices=["tf", "tfidf"], help="Top Stems using TF or TF-IDF")
    parser.add_argument("-n", type=int, default=15, help="Number of documents")
    args = parser.parse_args()

    vm = VectorModel(p=args.p, matrix_type=args.type, metric=args.metric, query=args.q, n=args.n)
    docs = vm.vector_query_compare()

    for doc, similarity in docs:
        print(doc, similarity)
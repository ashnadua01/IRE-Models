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

from queryBooleanRepresentation import BooleanQuery

class BooleanModel:
    def __init__(self, query, p=20, metric="tf", matrix_type="boolean"):
        self.top_stems = None
        self.matrix = None
        self.query = query
        self.p = p
        self.matrix_type = matrix_type
        self.metric = metric
        self.stemmer = PorterStemmer()
        self.bool_query_compare()

    def evaluate_boolean_expression(self, expression):
        result = np.ones(self.matrix.shape[1], dtype=int)
        i = 0
        flag = False
        while i < len(expression):
            if isinstance(expression[i], str):
                operator = expression[i]
                if operator == 'and':
                    i += 1
                    if i < len(expression):
                        operand = expression[i]
                        result &= operand
                        flag = True
                    else:
                        break
                elif operator == 'or':
                    i += 1
                    if i < len(expression):
                        operand = expression[i]
                        result |= operand
                        flag = True
                    else:
                        break
                elif operator == 'not':
                    i += 1
                    if i < len(expression):
                        operand = expression[i]
                        result &= ~operand
                        flag = True
                    else:
                        break
            else:
                result &= expression[i]
                flag = True
            i += 1

        if flag == False:
            result = np.zeros(self.matrix.shape[1], dtype=int)

        doc_names = []
        for index, value in enumerate(result):
            if value == 1:
                doc_names.append(self.all_articles[index])
                
        return doc_names
    
    def bool_query_compare(self):
        bq = BooleanQuery(p=self.p, matrix_type=self.matrix_type, metric=self.metric, query=self.query)
        self.matrix = bq.matrix
        self.bool_query = bq.bool_query
        self.top_stems = bq.top_stems
        self.term_to_index = bq.term_to_index
        self.all_articles = bq.all_articles
        self.result = self.evaluate_boolean_expression(self.bool_query)
        return self.result
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boolean Model")
    parser.add_argument("-q", type=str, help="Query")
    parser.add_argument("-p", type=int, default=20, help="Number of top terms")
    parser.add_argument("--type", type=str, default="boolean", choices=["vector", "boolean"], help="Matrix type")
    parser.add_argument("--metric", type=str, default="tf", choices=["tf", "tfidf"], help="Top Stems using TF or TF-IDF")
    args = parser.parse_args()

    mb = BooleanModel(p=args.p, matrix_type=args.type, metric=args.metric, query=args.q)
    docs = mb.bool_query_compare()

    for doc in docs:
        print(doc)
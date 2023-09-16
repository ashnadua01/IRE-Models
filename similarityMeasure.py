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

class SimilarityMeasure:
    def __init__(self, query, matrix):
        self.matrix = matrix
        self.query = query
        self.similarity = self.cosine_similarity_calculate()
        
    def cosine_similarity_calculate(self):
        similarities = []
        for index in range(self.matrix.shape[1]):
            doc_vector = self.matrix[:, index]
            dot_prod = np.dot(self.query, doc_vector)
            query_norm = np.linalg.norm(self.query)
            doc_norm = np.linalg.norm(doc_vector)

            if query_norm != 0 and doc_norm != 0:
                similarity = dot_prod / (query_norm * doc_norm)
            else:
                similarity = 0
            similarities.append((index, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
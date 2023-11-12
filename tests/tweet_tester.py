import warnings
warnings.filterwarnings("ignore")

import time
import_time = time.perf_counter()
from ast import keyword
from cmath import exp
import numpy as np
from bitarray import bitarray
import pandas as pd
import random
import re
from enum import Enum
import string
import scipy
import sklearn
import nltk
import plotly.express as px
from nltk.stem import PorterStemmer
#nltk.download('stopwords')
#from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csc_matrix
import umap.umap_ as umap
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import hdbscan
from sklearn.neighbors import kneighbors_graph
import leidenalg
import igraph as ig
import copy
import textwrap # hover text on dimension reduction/clustering plot
# Ignore warnings

import pickle
import datetime

import sys
# sys.path.insert(0, '/mnt/c/Users/erikz/Documents/Tweet_browser/Tweet_browser/src/tweet_browser_test')
sys.path.insert(0, '/mnt/c/Users/erikz/Documents/Tweet_browser/Tweet_browser/Frontend')
from tweet_browser import Session




end_import = time.perf_counter()
print("import time = ", end_import - import_time)

fileName = "Data/Session.pkl"

# this function reads in the data (copied from online)
def parse_data(filename):
    path = './' + filename
    begin = time.perf_counter()
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(path, encoding = "utf-8", index_col=False)
        elif "xls" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(path, index_col=[0])
        elif "txt" or "tsv" in filename:
            # Assume that the user uploaded an excel file
            
            df = pd.read_csv(path, delimiter = "\t",encoding = "ISO-8859-1",  index_col=[0])
    except Exception as e:
        print(e)
        return
    end = time.perf_counter()
    print("parse time = ", end-begin)
    return df

def preProcessingFcn(tweet, removeWords=list(), stem=True, removeURL=True, removeStopwords=True, 
    removeNumbers=False, removePunctuation=True):
    """
    Cleans tweets by removing words, stemming, etc.
    """
    ps = PorterStemmer()
    tweet = tweet.lower()
    tweet = re.sub(r"\\n", " ", tweet)
    tweet = re.sub(r"&amp", " ", tweet)
    if removeURL==True:
        tweet = re.sub(r"http\S+", " ", tweet)
    if removeNumbers==True:
        tweet=  ''.join(i for i in tweet if not i.isdigit())
    if removePunctuation==True:
        for punct in string.punctuation:
            tweet = tweet.replace(punct, ' ')
    #if removeStopwords==True:
        #tweet = ' '.join([word for word in tweet.split() if word not in stopwords.words('english')])
    if len(removeWords)>0:
        tweet = ' '.join([word for word in tweet.split() if word not in removeWords])
    if stem==True:
        tweet = ' '.join([ps.stem(word) for word in tweet.split()])
    return tweet

def toBoolArray(arr: bitarray):
    result = []
    for i in range(len(arr)):
        if arr[i]:
            result.append(True)
        else:
            result.append(False)
    return result

class Operation:
    parents = []
    outputs = []
    operationType = ""
    parmeters = ""
    times = []

    def __init__(self, inputArr, outputArr, searchType, parameter, times):
        self.parents = inputArr
        self.outputs = outputArr
        self.operationType = searchType
        self.parameters = parameter
        self.times = times

class DataBaseSim:
    def __init__(self, data):
        self.allData = data
        self.matrix = None
    def getRow(self, i):
        return self.allData.iloc[i]
    def getColHeaders(self):
        return self.allData.columns
    def selectRows(self, rows: list):
        return self.allData.iloc[rows]
    def dtypes(self):
        return self.allData.dtypes
    def shape(self):
        return self.allData.shape
    def setMatrix(self, input):
        self.matrix = input
    def getMatrix(self):
        return self.matrix

def createSession(fileName: str, makeMatrix = True, logSearches = False) -> Session:
    data = parse_data(fileName)
    s = Session(data, makeMatrix, logSearches)
    return s

def test1(s):
    s.advancedSearch("'covid' and ('hospital' or 'vaccine')")
    #s.printCurrSubset()
    print(s.currentSet.size)
    s.back()
    s.filterBy("State", "California")
    print(s.currentSet.size)
    s.advancedSearch("('hospital' or 'vaccine') and not 'Trump'")
    print(s.currentSet.size)
    s.back()

    s.advancedSearch("'trump' and not 'Trump'")
    print(s.currentSet.size)
    #s.printCurrSubset()
    s.back()
    s.regexSearch("trump")
    print(s.currentSet.size)

def test2(s):
    s.randomSubset(0.01)
    print(s.currentSet.size)
    s.back()
    print(s.currentSet.size)
    #s.back()
    #s.next()
    print(s.currentSet.size)
    s.searchKeyword(["the"])
    print(s.currentSet.size)
    s.back()
    #s.back()
    s.filterBy("State", "California")
    print(s.currentSet.size)
    s.back()
    s.advancedSearch("'trump' and not 'Trump'")
    print(s.currentSet.size)
    s.back()
    #s.back()
    s.printChildren()
    s.next(0)
    print(s.currentSet.size)
    #s.next(0)
    print(s.currentSet.size)
    #s.next()

def test3(s):
    s.randomSubset(0.001)
    print(s.currentSet.size)
    tempSet = s.currentSet
    s.back()
    print(tempSet.size)
    print(s.currentSet.size)
    s.advancedSearch("'the'", tempSet)
    print(s.currentSet.size)
    #s.printCurrSubset()
    s.searchKeyword(["the"], tempSet)
    print(s.currentSet.size)

def test4(s):
    s.searchKeyword(["the", "Census"])
    print(s.currentSet.size)
    s.back()
    s.advancedSearch("'the' and 'Census'")
    print(s.currentSet.size)
    s.back()
    s.regexSearch("trump|Trump")
    print(s.currentSet.size)
    s.back()
    s.searchKeyword(["trump", "Trump"], True)
    print(s.currentSet.size)
    s.back()
    s.advancedSearch("'trump' or 'Trump'")
    print(s.currentSet.size)
    s.regexSearch("^[1-9]")
    #s.printCurrSubset()
    print(s.currentSet.size)
    s.back()
    s.back()
    s.advancedSearch("'(' and ')' and not ('{' or '}')")
    #s.printCurrSubset()
    print(s.currentSet.size)
    s.back()
    s.regexSearch("\(.*\)")
    print(s.currentSet.size)
    temp = s.currentSet
    s.back()
    s.advancedSearch(" '(' and ')' ")
    print(s.currentSet.size)
    s.setDiff(temp)
    #s.printCurrSubset()
    print(s.currentSet.size)
    
def test5(s):
    s.simpleRandomSample(17000)
    #s.printCurrSubset()
    print(s.currentSet.size)
    s.back()
    s.weightedSample(10, "Retweets")
    #s.printCurrSubset()
    print(s.currentSet.size)

def test6(s):
    s.simpleRandomSample(30)
    s.printCurrSubset()
    print("\n\n ---------------------------------------------------------- \n")
    temp = s.getCurrentSubset()
    print(temp)
    s.printColumn(15)

def test7(s): #same as test 1
    s.advancedSearch("'covid' and ('hospital' or 'vaccine')")
    #s.printCurrSubset()
    print(s.currentSet.size)
    s.back()
    s.filterBy("State", "California")
    print(s.currentSet.size)
    s.advancedSearch("('hospital' or 'vaccine') and not 'Trump'")
    print(s.currentSet.size)
    s.back()

    s.advancedSearch("'trump' and not 'Trump'")
    print(s.currentSet.size)
    #s.printCurrSubset()
    s.back()
    s.back()
    #s.back()
    s.regexSearch("trump")
    print(s.currentSet.size)

def test8(s):
    s.filterBy("State", "New Jersey")
    s.regexSearch("Trump")
    print(s.currentSet.size)
    #s.printCurrSubset()
    #print(words)
    test = s.dimRed_and_clustering(dimRed1_method='pca', dimRed1_dims= 2, dimRed2_method='umap', 
        clustering_when= 'after_stage2', clustering_method= 'gmm', min_obs= 25, num_clusters= 2, num_neighbors= 25)
    #print(test)
    #print(matrix)
    print(s.currentSet.size)
    s.next()
    print(s.currentSet.size)
    #s.printCurrSubset()
    s.back()
    print(s.currentSet.size)

def test9(s):
    s.simpleRandomSample(30)
    print(s.currentSet.size)
    matrix, words = s.make_full_docWordMatrix(5)
    test = s.dimRed_and_clustering(docWordMatrix = matrix, dimRed1_method='pca', dimRed1_dims= 2, dimRed2_method='umap', 
        clustering_when= 'after_stage2', clustering_method= 'gmm', min_obs= 5, num_clusters= 5, num_neighbors= 5)
    s.next(opIndex=4)
    print(s.currentSet.size)
    s.back()
    s.next(opIndex=3)
    print(s.currentSet.size)
    s.back()
    s.next(opIndex=2)
    print(s.currentSet.size)
    s.back()
    s.next(opIndex=1)
    print(s.currentSet.size)
    s.back()
    s.next(opIndex=0)
    print(s.currentSet.size)

def test10(s):
    s.simpleRandomSample(300)
    #s.printCurrSubset()
    matrix, words = s.make_full_docWordMatrix(50)
    test = s.dimRed_and_clustering(docWordMatrix = matrix, dimRed1_method= 'pca', dimRed1_dims=2, clustering_when='before_stage1', 
        clustering_method='gmm', num_clusters=2, min_obs= 2, num_neighbors=2)
    #print(test)
    #test.show()
    test2 = s.dimRed_and_clustering(dimRed1_method= 'pca', dimRed1_dims=2, clustering_when='before_stage1', 
        clustering_method='leiden', num_clusters=4, min_obs= 2, num_neighbors=2)
    test3 = s.dimRed_and_clustering(dimRed1_method= 'pca', dimRed1_dims=2, clustering_when='before_stage1', 
        clustering_method='hdbscan', num_clusters=5, min_obs= 2, num_neighbors=2)
    s.next(0, 1)
    print(s.currentSet.size)
    s.back()
    s.next(1, 3)
    print(s.currentSet.size)
    s.back()
    s.next(2,2)
    print(s.currentSet.size)

def test11(s):
    #begin = time.perf_counter()
    s.filterBy("SenderGender", "FEMALE")
    #end = time.perf_counter()
    #print("time = ", end-begin)
    print(s.currentSet.size)
    s.weightedSample(10, "Sender Followers Count")
    s.simpleRandomSample(5)
    s.printCurrSubset(True)
    s.back()
    s.weightedSample(5, "Retweets")
    print(s.currentSet.size)

def test12(s):
    try: 
        s.weightedSample(100, "SenderScreenName")
    except (ValueError):
        print("exeption caught")
    
def test13(s):
    s.simpleRandomSample(300)
    #s.printCurrSubset()
    test = s.dimRed_and_clustering(dimRed1_method= 'pca', dimRed1_dims=2, clustering_when='after_stage2', 
        clustering_method='gmm', num_clusters=2, min_obs= 2, num_neighbors=2)
    s.printChildren()
    s.next()
    print(s.currentSet.size)
    matrix, words = s.make_full_docWordMatrix(5)
    test2 = s.dimRed_and_clustering(docWordMatrix= matrix, dimRed1_method= 'umap', dimRed1_dims=2, clustering_when='btwn', 
        clustering_method='gmm', num_clusters=2, min_obs= 2, num_neighbors=2)
    s.next()
    print(s.currentSet.size)
    s.back()
    s.back()
    matrix, words = s.make_full_docWordMatrix(1)
    test3 = s.dimRed_and_clustering(docWordMatrix = matrix, dimRed1_method= 'umap', dimRed1_dims=2, clustering_when='btwn', 
        clustering_method='gmm', num_clusters=2, min_obs= 2, num_neighbors=2)
    test4 = s.dimRed_and_clustering(docWordMatrix = matrix, dimRed1_method='pca', dimRed1_dims= 3, dimRed2_method='umap', 
        clustering_when= 'after_stage2', clustering_method= 'hdbscan', min_obs= 25, num_clusters= 2, num_neighbors= 25)
    test5 = s.dimRed_and_clustering(dimRed1_method='umap', dimRed1_dims= 5, dimRed2_method='umap', 
        clustering_when= 'btwn', clustering_method= 'k-means', min_obs= 25, num_clusters= 2, num_neighbors= 25)

def test14(s):
    s.simpleRandomSample(30)
    
    test = s.dimRed_and_clustering(dimRed1_method= 'pca', dimRed1_dims=2, clustering_when='after_stage2', 
        clustering_method='gmm', num_clusters=2, min_obs= 2, num_neighbors=2)

def test15(s):
    s.simpleRandomSample(50)
    t1 = s.getCurrentSubset()
    s.invert()
    print(s.currentSet.size)
    s.invert()
    t2 = s.getCurrentSubset()
    print(t1 == t2)
    print(s.currentSet.size)
    s.back()
    s.invert()
    print(s.currentSet.size)

def test16(s):
    s.simpleRandomSample(50)
    s1 = s.currentSet
    s.back()
    s.searchKeyword(["covid", "vaccine"], True)
    print(s.currentSet.size)
    s2 = s.currentSet
    s.back()
    s.searchKeyword(["covid", "hospital"], True)
    print(s.currentSet.size)
    s3 = s.currentSet
    s.setIntersect(s2)
    print(s.currentSet.size)
    s.back()
    s.setUnion(s2)
    print(s.currentSet.size)
    s.back()
    s.setDiff(s2)
    print(s.currentSet.size)

def test17(s):
    s.simpleRandomSample(25)
    s.back()
    s.simpleRandomSample(25)
    s.back()
    time.sleep(1)
    s.simpleRandomSample(25)
    s.back()
    s.simpleRandomSample(25)
    s.back()
    #s.printChildren()
    begin = time.perf_counter()
    s.searchKeyword(["trump"])
    end = time.perf_counter()
    print("first time = ", end-begin)
    s.back()
    begin = time.perf_counter()
    s.searchKeyword(["trump"])
    end = time.perf_counter()
    print("second time = ", end-begin)
    s.back()
    s.advancedSearch("'trump' and not 'Trump'")
    s.back()
    s.advancedSearch("'trump' and not 'Trump'")
    s.back()
    s.regexSearch("covid")
    s.back()
    s.regexSearch("covids")
    s.back()
    s.filterBy("State", "California")
    s.back()
    s.filterBy("State", "California")
    s.back()
    s.printChildren()
    s.simpleRandomSample(30)
    test = s.dimRed_and_clustering(dimRed1_method= 'pca', dimRed1_dims=2, clustering_when='after_stage2', 
        clustering_method='gmm', num_clusters=2, min_obs= 2, num_neighbors=2)
    test = s.dimRed_and_clustering(dimRed1_method= 'pca', dimRed1_dims=2, clustering_when='after_stage2', 
        clustering_method='gmm', num_clusters=2, min_obs= 2, num_neighbors=2)

def test18(s):
    print(s.currentSet.size)
    s.filterDate("2020-09-15", "2021-09-15")
    print(s.currentSet.size)
    assert s.currentSet.size == 18

#exclude function
def test19(s):
    s.exclude(["a", "the"])
    print(s.currentSet.size)

def test20(s):
    s.simpleRandomSample(30)
    # s.summarize()

def test21(s):
    s.removeRetweets()
    print(s.currentSet.size)

def test99(s):
    begin = time.perf_counter()
    s.filterBy("State", "California")
    for i in range(100):
        s.regexSearch("trump")
    print("total time", time.perf_counter() - begin)
    print(s.currentSet.size)

def allTests(s1):
    current_module = __import__(__name__)
    for i in range(1,21):
        s = copy.deepcopy(s1)
        print("---------------------------")
        print("test ", i)
        print("---------------------------")
        func = getattr(current_module, "test{}".format(i))
        func(s) 

if __name__=='__main__':
    #s = createSession("allCensus_sample.csv")
    s = createSession("allCensus_sample.csv", False)
    #with open(fileName, "rb") as input:
        #s = pickle.load(input) 
    # allTests(s)
    test99(s)

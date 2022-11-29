from ast import keyword
from cmath import exp
import numpy as np
from bitarray import bitarray
from copy import deepcopy
import math
import base64
import io
import json
import pandas as pd
import random
import tracemalloc
import re
from enum import Enum
import string
from nltk.stem import PorterStemmer
#nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# clear 'temp' folder every week
import os, sys, time
from subprocess import call

# Ignore warnings
import warnings

#from pyparsing import null_debug_action

#from formatter import NullFormatter
warnings.filterwarnings("ignore")


# this function reads in the data (copied from online)
def parse_data(filename):
    path = './' + filename
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(path, encoding = "utf-8", index_col=[0])
        elif "xls" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(path, index_col=[0])
        elif "txt" or "tsv" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_csv(path, delimiter = "\t",encoding = "ISO-8859-1",  index_col=[0])
    except Exception as e:
        print(e)
        return
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
    if removeStopwords==True:
        tweet = ' '.join([word for word in tweet.split() if word not in stopwords.words('english')])
    if len(removeWords)>0:
        tweet = ' '.join([word for word in tweet.split() if word not in removeWords])
    if stem==True:
        tweet = ' '.join([ps.stem(word) for word in tweet.split()])
    return tweet

class Operation:
    parents = []
    outputs = []
    operationType = ""
    parmeters = ""

    def __init__(self, inputArr, outputArr, searchType, parameter):
        self.parents = inputArr
        self.outputs = outputArr
        self.operationType = searchType
        self.parameters = parameter

class Subset:
    indices = bitarray()
    size = 0
    parent = None
    children = []
    def __init__(self, ind: bitarray):
        self.indices = ind

class Session:
    base = None
    currentSet = None
    length = 0
    def __init__(self, baseSet):
        self.length = len(baseSet)
        arr = bitarray(self.length)
        arr.setall(1)
        self.base = Subset(arr)
        self.base.size = self.length
        self.currentSet = self.base

    def makeOperation(self, outPut, count: int, funcName, params, input = None):
        if input == None:
            input = self.currentSet
        newSet = Subset(outPut)
        newSet.size = count
        newOp = Operation([input], [newSet], funcName, params)
        newOp.outputs[0].parent = newOp
        #input.children.append(deepcopy(newOp))
        #can't use append
        newOp.parents[0].children = newOp.parents[0].children + [newOp]
        self.currentSet = newSet

    def printColumn(self, column: int):
        for i in range(self.length):
            if (self.currentSet.indices[i]):
                print(retrieveRow(i)[column])

    def getCurrentSubset(self):
        s = []
        for i in range(self.length):
            if (self.currentSet.indices[i]):
                s.append(retrieveRow(i))
        return s

    def printCurrSubset(self, verbose: bool = False):
        for i in range(self.length):
            if (self.currentSet.indices[i]):
                if verbose:
                    print(retrieveRow(i))
                else:
                    print(i, ": ", retrieveRow(i)[15])

    def invert(self, input: bitarray):
        for i in range(len(input)):
            if input[i]:
                input[i] = False
            else:
                input[i] = True
        return input

    def randomSubset(self, probability, inputSet: Subset = None):
        if (inputSet == None):
            inputSet = self.currentSet
        random.seed()
        ans = bitarray(self.length)
        ans.setall(0)
        count = 0
        for i in range(self.length):
            if (inputSet.indices[i] and random.random() < probability):
                ans[i] = True
                count += 1
        self.makeOperation(ans, count, "randomSubset", "None")

    def simpleRandomSample(self, size: int, inputSet: Subset = None):
        if (inputSet == None):
            inputSet = self.currentSet
        random.seed()
        ans = bitarray(self.length)
        ans.setall(0)
        if(inputSet.size < size):
            print("Invalid sample size")
            return
        population = []
        for i in range(inputSet.size):
            if inputSet.indices[i]:
                population.append(i)
        temp = np.random.choice(population, size, replace=False)
        for j in temp:
            ans[j] = True
        self.makeOperation(ans, size, "simpleRandomSample", size)

    def weightedSample(self, size: int, colName: str, inputSet: Subset = None):
        if (inputSet == None):
            inputSet = self.currentSet
        random.seed()
        ans = bitarray(self.length)
        ans.setall(0)
        if(inputSet.size < size):
            print("Invalid sample size")
            return
        population = []
        weights = []
        sum = 0
        for i in range(inputSet.size):
            if inputSet.indices[i]:
                population.append(i)
                value = retrieveRow(i)[headerDict[colName]]
                if value != value : # still need to check if the colName corresponds with a number that can be weighted
                    value = 0
                value += 1
                sum += value
                weights.append(int(value))
        for j in range(len(weights)):
            weights[j] = float(weights[j] / sum)  
        temp = np.random.choice(population, size, replace=False, p=weights)
        #temp.sort()
        #print(temp)
        for k in temp:
            ans[k] = True
            #print(retrieveRow(k)[headerDict[colName]], end=" ")
        #print()
        self.makeOperation(ans, size, "weightedSample", colName + str(size))

    def searchKeyword(self, keywords: list, orMode: bool = False, inputSet: Subset = None):
        if (inputSet == None):
            inputSet = self.currentSet
        ans = bitarray(self.length)
        ans.setall(0)
        count = 0
        for i in range(self.length):
            if(inputSet.indices[i]):
                if (orMode):
                    include = False
                    for j in keywords:
                        if (retrieveRow(i)[15].find(j) != -1):
                            include = True
                            break
                else:
                    include = True
                    for j in keywords:
                        if (retrieveRow(i)[15].find(j) == -1):
                            include = False
                            break
                if include:
                    ans[i] = True
                    count += 1
        self.makeOperation(ans, count, "searchKeyword", keywords)

    def advancedSearch(self, expression: str, inputSet: Subset = None):
        if (inputSet == None):
            inputSet = self.currentSet
        ans = bitarray(self.length)
        ans.setall(0)
        count = 0
        # split the expression into a list of operands and keywords
        #regex = '\s*\(|\)\s*|\s*and\s*|\s*or\s*|\s*not\s*'
        #keywords = list(filter(None, re.split(regex, expression)))
        keywords = re.findall("'[^']+'", expression)
        # loop through to evaluate the truth value of each keyword
        for i in range(self.length):
            if(inputSet.indices[i]):
                newExpression = expression
                for j in keywords:
                    if(retrieveRow(i)[15].find(j[1:-1]) > -1):
                        newExpression = newExpression.replace(j, " True")
                    else:
                        newExpression = newExpression.replace(j, " False")
                if(eval(newExpression)):
                    ans[i] = True
                    count += 1
        self.makeOperation(ans, count, "advancedSearch", expression)
        

    def regexSearch(self, expression: str, inputSet: Subset = None):
        if (inputSet == None):
            inputSet = self.currentSet
        ans = bitarray(self.length)
        ans.setall(0)
        count = 0
        for i in range(self.length):
            if(inputSet.indices[i]):
                if(re.findall(expression, retrieveRow(i)[15], re.M)):
                    ans[i] = True
                    count += 1
        self.makeOperation(ans, count, "regexSearch", expression)
    
    def filterBy(self, colName: str, value, inputSet: Subset = None):
        if (inputSet == None):
            inputSet = self.currentSet
        ans = bitarray(self.length)
        ans.setall(0)
        count = 0
        for i in range(self.length):
            if (inputSet.indices[i] and retrieveRow(i)[headerDict[colName]] == value):
                ans[i] = True
                count += 1
        self.makeOperation(ans, count, "filterBy", colName + " = " + value)

    def setDiff(self, setOne: Subset, setZero: Subset = None):
        if (setZero == None):
            setZero = self.currentSet
        ans = bitarray(self.length)
        ans.setall(0)
        count = 0
        for i in range(self.length):
            if (setZero.indices[i] and not setOne.indices[i]):
                ans[i] = True
                count += 1
        self.makeOperation(ans, count, "setDiff", setOne)

    def setUnion(self, setOne: Subset, setZero: Subset = None):
        if (setZero == None):
            setZero = self.currentSet
        ans = bitarray(self.length)
        ans.setall(0)
        count = 0
        for i in range(self.length):
            if (setZero.indices[i] or setOne.indices[i]):
                ans[i] = True
                count += 1
        self.makeOperation(ans, count, "setUnion", setOne)

    def setIntersect(self, setOne: Subset, setZero: Subset = None):
        if (setZero == None):
            setZero = self.currentSet
        ans = bitarray(self.length)
        ans.setall(0)
        count = 0
        for i in range(self.length):
            if (setZero.indices[i] and setOne.indices[i]):
                ans[i] = True
                count += 1
        self.makeOperation(ans, count, "setintersect", setOne)

    def back(self, index: int = 0):
        if(self.currentSet.size == self.length) or index >= len(self.currentSet.parent.parents):
        #if(self.currentSet == self.base):
            print("Can't go back")
            return
        self.currentSet = self.currentSet.parent.parents[index]
    
    def next(self, index = -1):
        if len(self.currentSet.children) == 0 or index >= len(self.currentSet.children):
            print("Can't go next")
            return
        self.currentSet = self.currentSet.children[index].outputs[0]

    def printChildren(self):
        if len(self.currentSet.children) == 0:
            print("No children searches")
            return
        for i in self.currentSet.children:
            print("Type = ", i.operationType, " parameters = ", i.parameters)
    


    ##### Clustering ######
    
    def make_full_docWordMatrix(self, inputSet: Subset = None):
        cleanedTweets = []
        for i in range(self.length):
            if self.currentSet.indices[i]:
                cleanedTweets.append([preProcessingFcn(tweet) for tweet in retrieveRow(i)[15]])

        # create document-word matrix
        vectorizer = CountVectorizer(strip_accents='unicode', min_df=5, binary=False)
        docWordMatrix_orig = vectorizer.fit_transform(cleanedTweets)
        docWordMatrix_orig = docWordMatrix_orig.astype(dtype='float64')
        # save as sparse document-word matrix as json file
        rows_orig, cols_orig = docWordMatrix_orig.nonzero()
        data_orig = docWordMatrix_orig.data
        docWordMatrix_orig_json = json.dumps({'rows_orig':rows_orig.tolist(), 'cols_orig':cols_orig.tolist(),
            'data_orig':data_orig.tolist(), 'dims_orig':[docWordMatrix_orig.shape[0], docWordMatrix_orig.shape[1]],
            'feature_names':vectorizer.get_feature_names(), 'indices':headers})
        return docWordMatrix_orig_json

# dataSet = None
# headers = None
# headerDict = dict()
def retrieveRow(rowNum: int):
    return dataSet[rowNum]

def createSession(fileName: str) -> Session:
    data = parse_data("allCensus_sample.csv")
    global dataSet 
    global headers
    global headerDict
    dataSet = data.values
    headers = data.columns
    headerDict = dict()
    for i in range(len(headers)):
        colName = headers[i]
        headerDict[colName] = i
    s = Session(dataSet)
    return s

if __name__=='__main__':
    print("You weren't supposed to run this")
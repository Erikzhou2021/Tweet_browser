from ast import keyword
from cmath import exp
import numpy as np
import os.path
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
import textwrap # hover text on dimension reduction/clustering plot
from ai_summary import Summarizer

# Ignore warnings
import warnings
import pickle
import datetime
#from pyparsing import null_debug_action

#from formatter import NullFormatter
warnings.filterwarnings("ignore")

# this function reads in the data (copied from online)
def parse_data(filename):
    path = './' + filename
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

class Subset:
    indices = []
    size = 0
    parent = None
    children = []
    def __init__(self, ind):
        self.indices = ind

class Session:
    def __init__(self, data, logSearches = False):
        self.logSearches = logSearches
        if logSearches:
            self.createSessionDump()

        self.allData = data
        self.allData['CreatedTime'] = pd.to_datetime(self.allData['CreatedTime'])
        self.allData['Message'] = self.allData['Message'].astype("string")
        self.headerDict = dict()
        headers = self.allData.columns
        for i in range(len(headers)): # put <header, columnNum> into a dictionary for faster access
            self.headerDict[headers[i]] = i
        self.length = self.allData.shape[0]
        arr = range(self.length)
        self.base = Subset(np.array(arr))
        self.base.size = self.length
        self.currentSet = self.base
        self.weightable = dict()
        for i in range(len(self.allData.dtypes)):
            if self.allData.dtypes[i] == int or self.allData.dtypes[i] == float:
                self.weightable[headers[i]] = i

    def makeOperation(self, outputs, counts, funcName, params, inputs: Subset = None):
        if inputs == None or type(inputs) != Subset:
            inputs = self.currentSet
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(outputs, list):
            outputs = [outputs]
        if not isinstance(counts, list):
            counts = [counts]
        newSets = list()
        for i in range(len(outputs)):
            newSet = Subset(outputs[i])
            newSet.size = counts[i]
            newSets.append(newSet)
        times = [datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")]
        newOp = Operation(inputs, newSets, funcName, params, times)
        for i in range(len(outputs)):
            newOp.outputs[i].parent = newOp
        for i in range(len(inputs)):
            #input.children.append(deepcopy(newOp))
            #can't use append
            newOp.parents[i].children = newOp.parents[i].children + [newOp]
        self.currentSet = newSets[0]
        if self.logSearches:
            with open(self.fileName, "wb+") as ouput:
                pickle.dump(self, ouput, pickle.HIGHEST_PROTOCOL)

    def createSessionDump(self):
        name = "Data/Session"
        i = 0
        while os.path.isfile(name + str(i) + ".pkl"):
            i += 1
        self.fileName = name + str(i) + ".pkl"

    def printColumn(self, column: int):
        print(self.allData.iloc[self.currentSet.indices, column])

    def getCurrentSubset(self):
        return self.allData.iloc[self.currentSet.indices]

    def printCurrSubset(self, verbose: bool = False):
        if verbose:
            print(self.allData.iloc[self.currentSet.indices])
        else:
            print(self.allData.loc[self.currentSet.indices]["Message"])

    def checkOperation(self, funcName, params, switch = True, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        for op in inputSet.children:
            if funcName == op.operationType and params == op.parameters:
                op.times.append(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
                if switch:
                    self.currentSet = op.outputs[0]
                    return True
                return True, op.outputs[0]
        return False

    def invert(self, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        count = self.length - inputSet.size
        self.makeOperation(~inputSet.indices, count, "Invert", "None")

    # def randomSubset(self, probability, inputSet: Subset = None): # need to test
    #     if inputSet == None or type(inputSet) != Subset:
    #         inputSet = self.currentSet
    #     if probability > 1 or probability < 0:
    #         raise ValueError("Invalid probability")
    #     random.seed()
    #     ans = []
    #     for index in inputSet.indices:
    #         if random.random() < probability:
    #             ans.append(index)
    #     ans = np.array(ans)
    #     self.makeOperation(ans, len(ans), "randomSubset", "None")

    def simpleRandomSample(self, size: int, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        if inputSet.size < size:
            raise ValueError("Invalid sample size")
        random.seed()
        ans = np.random.choice(inputSet.indices, size, replace=False)
        self.makeOperation(ans, size, "simpleRandomSample", size)

    def weightedSample(self, size: int, colName: str, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        if inputSet.size < size:
            raise ValueError("Invalid sample size")
        if colName not in self.weightable:
            raise ValueError("Column name does not correspond to a column that can be weighted")
        random.seed()
        temp = self.allData[colName][self.allData.index.isin(inputSet.indices)]
        temp.fillna(0, inplace=True)
        temp += 1
        ans = temp.sample(size, weights=temp)
        self.makeOperation(ans.index, ans.shape[0], "weightedSample", colName + str(size))

    def searchKeyword(self, keywords: list, orMode: bool = False, caseSensitive = False, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        params = keywords + ["orMode = " + str(orMode) + ", caseSensitive = " + str(caseSensitive)]
        if self.checkOperation("searchKeyword", params):
            return
        flag = re.DOTALL | re.MULTILINE
        if caseSensitive == False:
            flag |= re.IGNORECASE
        if orMode:
            pattern = re.compile(r'\b' + "|".join([re.escape(word) for word in keywords]) + r'\b', flag)
        else:
            pattern = re.compile("^" + "".join([r"(?=.*\b" + re.escape(word) + r"\b)" for word in keywords]) + ".*$", flag)
        def predicate(row):
            if re.search(pattern, row.iloc[self.headerDict["Message"]]):
                return True
            return False

        if (self.length / inputSet.size >= 10):
            tempInd = np.zeros(self.length, dtype=bool)
            for row in inputSet.indices:
                tempInd[row] = predicate(self.allData.iloc[row])
            ans = self.allData[tempInd]
        else:
            tempInd = self.allData.index.isin(inputSet.indices)
            ans = self.allData[(tempInd) & self.allData.apply(predicate, axis=1)]
        self.makeOperation(ans.index, ans.shape[0], "searchKeyword", params)

    def advancedSearch(self, expression: str, caseSensitive = False, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        if self.checkOperation("advancedSearch", expression):
            return
        flag = re.DOTALL | re.MULTILINE
        if caseSensitive == False:
            flag |= re.IGNORECASE
        keywords = re.findall("'[^']+'", expression)
        # loop through to evaluate the truth value of each keyword
        def predicate(row):
            newExpression = expression
            for j in keywords:
                    pattern = r"\b" + re.escape(j[1:-1]) + r"\b"
                    if re.search(pattern, row.iloc[self.headerDict["Message"]]):
                        newExpression = newExpression.replace(j, " True")
                    else:
                        newExpression = newExpression.replace(j, " False")
            return eval(newExpression)
        try: 
            if (self.length / inputSet.size >= 10):
                tempInd = np.zeros(self.length, dtype=bool)
                for row in inputSet.indices:
                    tempInd[row] = predicate(self.allData.iloc[row])
                ans = self.allData[tempInd]
            else:
                tempInd = self.allData.index.isin(inputSet.indices)
                ans = self.allData[(tempInd) & self.allData.apply(predicate, axis=1)]
        except Exception as e: 
            print("invalid expression")
            return
        self.makeOperation(ans.index, ans.shape[0], "advancedSearch", expression)

    def regexSearch(self, expression: str, caseSensitive = False, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        if self.checkOperation("regexSearch", expression):
            return
        flag = re.DOTALL | re.MULTILINE
        if not caseSensitive:
            flag |= re.IGNORECASE
        pattern = re.compile(expression, flag)
        def predicate(row):
            if re.search(pattern, row.iloc[self.headerDict["Message"]]):
                return True
            return False
        if (self.length / inputSet.size >= 10):
            tempInd = np.zeros(self.length, dtype=bool)
            for row in inputSet.indices:
                tempInd[row] = predicate(self.allData.iloc[row])
            ans = self.allData[tempInd]
        else:
            tempInd = self.allData.index.isin(inputSet.indices)
            ans = self.allData[(tempInd) & self.allData.apply(predicate, axis=1)]
        self.makeOperation(ans.index, ans.shape[0], "regexSearch", expression)

    def exclude(self, keywords: list, caseSensitive = False, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        if self.checkOperation("exclude", keywords):
            return
        if caseSensitive:
            pattern = re.compile(r'\b' + "|".join([re.escape(word) for word in keywords]) + r'\b')
        else:
            pattern = re.compile(r'\b' + "|".join([re.escape(word) for word in keywords]) + r'\b', re.IGNORECASE)
        def predicate(row):
            if re.search(pattern, row.iloc[self.headerDict["Message"]]):
                return False
            return True
        tempInd = self.allData.index.isin(inputSet.indices)
        ans = self.allData[(tempInd) & self.allData.apply(predicate, axis=1)]
        self.makeOperation(ans.index, ans.shape[0], "searchKeyword", keywords)

    def filterBy(self, colName: str, value, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        if self.checkOperation("filterBy", colName + " = " + str(value)):
            return
        tempInd = self.allData.index.isin(inputSet.indices)
        ans = self.allData.loc[(tempInd) & (self.allData[colName] == value)]
        self.makeOperation(ans.index, ans.shape[0], "filterBy", colName + " = " + str(value))
        
    def filterDate(self, startDate: str, endDate: str, inputSet = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        if self.checkOperation("filterTime", startDate + " to " + endDate):
            return
        format = '%Y-%m-%d'
        startDate = datetime.datetime.strptime(startDate, format)
        endDate = datetime.datetime.strptime(endDate, format)
        tempInd = self.allData.index.isin(inputSet.indices)
        ans = self.allData[(tempInd) & (self.allData['CreatedTime'] >= startDate) & (self.allData['CreatedTime'] <= endDate)]
        self.makeOperation(ans.index, ans.shape[0], "filterTime", str(startDate) + " to " + str(endDate))

    def removeRetweets(self, inputSet = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        if self.checkOperation("removeRetweets", "None"):
            return
        tempInd = self.allData.index.isin(inputSet.indices)
        ans = self.allData.loc[(tempInd) & (self.allData['MessageType'] != "Twitter Retweet")]
        self.makeOperation(ans.index, ans.shape[0], "removeRetweets", "None")

    def setDiff(self, setOne: Subset, setZero: Subset = None):
        if setZero == None or type(setZero) != Subset:
            setZero = self.currentSet
        if type(setOne) != Subset:
            raise TypeError("Set operators require two subset objects")
        indexZero = self.allData.index.isin(setZero.indices) 
        indexOne = self.allData.index.isin(setOne.indices) 
        ans = self.allData.loc[(indexZero) & ~(indexOne)]
        self.makeOperation(ans.index, ans.shape[0], "setDiff", setOne)

    def setUnion(self, setOne: Subset, setZero: Subset = None):
        if setZero == None or type(setZero) != Subset:
            setZero = self.currentSet
        if type(setOne) != Subset:
            raise TypeError("Set operators require two subset objects")
        indexZero = self.allData.index.isin(setZero.indices) 
        indexOne = self.allData.index.isin(setOne.indices) 
        ans = self.allData.loc[(indexZero) | (indexOne)]
        self.makeOperation(ans.index, ans.shape[0], "setUnion", setOne)

    def setIntersect(self, setOne: Subset, setZero: Subset = None):
        if setZero == None or type(setZero) != Subset:
            setZero = self.currentSet
        if type(setOne) != Subset:
            raise TypeError("Set operators require two subset objects")
        indexZero = self.allData.index.isin(setZero.indices) 
        indexOne = self.allData.index.isin(setOne.indices) 
        ans = self.allData.loc[(indexZero) & (indexOne)]
        self.makeOperation(ans.index, ans.shape[0], "setintersect", setOne)

    def backToBase(self):
        self.currentSet = self.base

    def back(self, index: int = 0):
        if(self.currentSet.size == self.length) or index >= len(self.currentSet.parent.parents):
        # if(self.currentSet == self.base):
            raise IndexError("Can't go back (Out of bounds)")
        self.currentSet = self.currentSet.parent.parents[index]
    
    def next(self, setIndex = -1, opIndex = 0):
        if len(self.currentSet.children) == 0 or setIndex >= len(self.currentSet.children):
            raise IndexError("Can't go to next (Out of bounds)")
        self.currentSet = self.currentSet.children[setIndex].outputs[opIndex]

    def printChildren(self):
        if len(self.currentSet.children) == 0:
            print("No children searches")
            return
        for i in self.currentSet.children:
            print("Type = ", i.operationType, ", parameters = ", i.parameters,
                  ", time = ", i.times, ", number of children = ", len(i.outputs))
    
    def summarize(self, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        summarizer = Summarizer()
        result = summarizer.llm_summarize(self.allData.iloc[toBoolArray(inputSet.indices)])
        print(result)
        return result


def createSession(fileName: str, logSearches = False) -> Session:
    data = parse_data(fileName)
    s = Session(data, logSearches)
    return s

if __name__=='__main__':
    print("You weren't supposed to run this")

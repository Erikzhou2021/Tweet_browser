from ast import keyword
from cmath import exp
import numpy as np
import os.path
import json
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
from scipy.sparse import csc_matrix
import igraph as ig
import textwrap # hover text on dimension reduction/clustering plot
# from fastlexrank import FastLexRankSummarizer
import ai_summary
import torch
from openai import OpenAI
from prompts import *
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Ignore warnings
import warnings
import pickle
import datetime
#from pyparsing import null_debug_action

#from formatter import NullFormatter
warnings.filterwarnings("ignore")

embedding_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5", device="cuda" if torch.cuda.is_available() else "cpu"
)

# this function reads in the data (copied from online)
def parse_data(filename, header='infer'):
    path = './' + filename
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(path, encoding = "utf-8", index_col=False, header=header)
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
    searchIndex = -1
    def __init__(self, ind):
        self.indices = ind

class Session:
    def __init__(self, data, logSearches = False, embeddings=None):
        self.logSearches = logSearches
        if logSearches:
            self.createSessionDump()

        if embeddings is None:
            embeddings = embedding_model.encode(
                data["Message"],
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=32,
                normalize_embeddings=True,
            )

        self.embeddings = embeddings
        self.allData = data
        self.allData['CreatedTime'] = pd.to_datetime(self.allData['CreatedTime']).dt.floor('D')
        self.allData['Message'] = self.allData['Message'].astype("string")
        self.allData['SimilarityScore'] = np.nan
        # self.allData['State'] = self.allData['State'].str.lower()
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
        # self.summarizer = FastLexRankSummarizer()
        for i in range(len(self.allData.dtypes)):
            if self.allData.dtypes[i] == int or self.allData.dtypes[i] == float:
                self.weightable[headers[i]] = i

    def makeOperation(self, outputs, counts, funcName, params, switch = True, inputs: Subset = None):
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
        if switch:
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
                    return True, None
                return True, op.outputs[0]
        return False, None

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

    def randomShuffle(self, seed: int, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        return self.allData.iloc[self.currentSet.indices].sample(frac=1, random_state=seed)

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
        if temp.sum() > 0:
            temp /= temp.sum()
            ans = np.random.choice(inputSet.indices, size, replace=False, p=temp)
        else:
            ans = np.random.choice(inputSet.indices, size, replace=False)
        self.makeOperation(ans, size, "weightedSample", str(size) + ";" + colName)

    def searchKeyword(self, keywords: list, orMode: bool = False, caseSensitive = False, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        params = keywords + ["orMode = " + str(orMode) + ", caseSensitive = " + str(caseSensitive)]
        result, _ = self.checkOperation("searchKeyword", params)
        if result or inputSet.size == 0:
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
        result, _ = self.checkOperation("advancedSearch", expression)
        if result or inputSet.size == 0:
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
        result, _ = self.checkOperation("regexSearch", expression)
        if result or inputSet.size == 0:
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
        result, _ = self.checkOperation("exclude", keywords)
        if result:
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
        found, result = self.checkOperation("filterBy", colName + " = " + str(value), False, self.base)
        if not found:
            ans = self.allData.loc[(self.allData[colName] == value)]

            self.makeOperation(ans.index, ans.shape[0], "filterBy", colName + " = " + str(value), False, self.base)
            found, result = self.checkOperation("filterBy", colName + " = " + str(value), False, self.base)

        self.setIntersect(result)
        
    def filterDate(self, startDate: str, endDate: str, inputSet = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        found, result = self.checkOperation("filterTime", startDate + " to " + endDate, False, self.base)
        if not found:  
            format = '%Y-%m-%d'
            dateStart = datetime.datetime.strptime(startDate, format)
            dateEnd = datetime.datetime.strptime(endDate, format)
            ans = self.allData[(self.allData['CreatedTime'] >= dateStart) & (self.allData['CreatedTime'] <= dateEnd)]

            self.makeOperation(ans.index, ans.shape[0], "filterTime", str(startDate) + " to " + str(endDate), False, self.base)
            found, result = self.checkOperation("filterTime", startDate + " to " + endDate, False, self.base)

        self.setIntersect(result)

    def findMinDate(self):
        return self.allData["CreatedTime"].min()
    
    def findMaxDate(self):
        return self.allData["CreatedTime"].max()

    def removeRetweets(self, inputSet = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        found, result = self.checkOperation("removeRetweets", "None", False, self.base)
        if not found:
            ans = self.allData[(self.allData['MessageType'] != "Twitter Retweet")]

            self.makeOperation(ans.index, ans.shape[0], "removeRetweets", "None", False, self.base)
            found, result = self.checkOperation("removeRetweets", "None", False, self.base)
            
        self.setIntersect(result)

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

    def setIntersect(self, inputs, setZero: Subset = None):
        if setZero == None or type(setZero) != Subset:
            setZero = self.currentSet
        if type(inputs) == Subset:
            inputs = [inputs]
        found, result = self.checkOperation("setintersect", inputs)
        if found: 
            return
        index = self.allData.index.isin(setZero.indices) 
        for subSet in inputs:
            index &= self.allData.index.isin(subSet.indices) 
        ans = self.allData.loc[index]
        self.makeOperation(ans.index, ans.shape[0], "setintersect", inputs)

    def resetToBase(self):
        self.currentSet = self.base

    def back(self, index: int = 0):
        if self.currentSet.parent == None or index >= len(self.currentSet.parent.parents):
        # if(self.currentSet == self.base):
            raise IndexError("Can't go back (Out of bounds)")
        self.currentSet = self.currentSet.parent.parents[index]
    
    def next(self, setIndex = -1, opIndex = 0):
        if len(self.currentSet.children) == 0 or setIndex >= len(self.currentSet.children):
            raise IndexError("Can't go to next (Out of bounds)")
        self.currentSet = self.currentSet.children[setIndex].outputs[opIndex]

    def getRandomSampleChildren(self, weightColumn):
        ans = []
        for i in range(len(self.currentSet.children)):
            child = self.currentSet.children[i]
            if weightColumn == "None" and child.operationType == "simpleRandomSample":
                ans.append(i)
            elif child.operationType == "weightedSample" and child.parameters.split(";", 1)[1] == weightColumn:
                ans.append(i)
        return ans
    
    # def resetRandomSampleChildren(self):
    #     for i in range(len(self.currentSet.children)):
    #         if self.currentSet.children[i].operationType == "simpleRandomSample":


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
        assert(inputSet.size <= 100)
        tweets = ''
        for i in range(len(inputSet.indices)):
            tweet = self.allData.iloc[inputSet.indices[i]]['Message']
            tweets += f"{i}-[{tweet}] "
        return ai_summarize(tweets)
    
    def parseSummary(self, AISummary, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        pattern = re.compile(r'\([\d,\s]+\)')
        sources = re.findall(pattern, AISummary)
        strings = re.split(pattern, AISummary)
        # strings = [re.sub(r'\s+([,\.])', r'\1', sentence) for sentence in strings]
        for i in range(1, len(strings)):
            if len(strings[i-1]) > 0 and len(strings[i]) > 0 and (strings[i][0] == '.' or strings[i][0] == ","):
                while strings[i][-1] == " ":
                    strings[i] = strings[i][:-1]
        if len(strings) >  len(sources) and len(strings) > 1:
            strings[-2] += strings[-1]
            strings = strings[:-1]
        unused = set(range(inputSet.size))
        tweets = []
        for match in sources:
            currList = []
            sourceNums = re.findall(r'\d+', match)
            for source in sourceNums:
                sourceNum = int(source)
                if sourceNum in unused:
                    unused.remove(sourceNum)
                if sourceNum < inputSet.size:
                    currList.append(inputSet.indices[sourceNum])
            tweets.append(currList)
        return strings, tweets, list(unused)

    def getCentral(self, inputSet = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        input = self.embeddings.iloc[inputSet.indices]
        scores = ai_summary.get_fastlexrank_scores(input)
        data = self.allData.iloc[inputSet.indices]
        data = data.assign(centrality=scores)
        return data.sort_values(by=["centrality"], ascending=False)
    
    def semanticSearch(self, query, topPercent, inputSet = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        query_embedding = embedding_model.encode(
            query, convert_to_tensor=True, normalize_embeddings=True
        )
        embeddingTensor = torch.from_numpy(self.embeddings.iloc[inputSet.indices].values).float()
        cos_scores = torch.matmul(query_embedding, embeddingTensor.T).to("cpu").numpy().flatten()
        df = self.allData.iloc[inputSet.indices]
        df = df.assign(cos_score=cos_scores)
        df = df.nlargest(int(topPercent * inputSet.size), 'cos_score')
        self.allData.loc[df.index, ["SimilarityScore"]] = df['cos_score']
        self.makeOperation(df.index, df.shape[0], "semanticSearch", topPercent)
    
    def stanceAnalysis(self, topic, stances, examples, inputSet = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        # resultDict = {}
        # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        # i = 0
        # while i < len(inputSet.indices):
        #     totalTokens = 0
        #     tweets = ""
        #     tokens = []
        #     while i < len(inputSet.indices) and totalTokens + len(tokens) < 7500:
        #         tweet = self.allData.iloc[inputSet.indices[i]]['Message']
        #         tokens = tokenizer.tokenize(tweet)
        #         tweets += f"{i}-[{tweet}] "
        #         totalTokens += len(tokens)
        #         i += 1
        #     batchResult = stance_annotation(tweets, topic, stances, examples)
        #     resultDict = {**resultDict, **(json.loads(batchResult))}
        # results = [resultDict[tweetNum] for tweetNum in range(len(inputSet.indices))]
        df = self.allData.iloc[inputSet.indices].reset_index(drop=True)
        results = stance_annotation(df['Message'], topic, stances, examples)
        assert(len(results) == len(df))
        stances = []
        for i in range(len(results)):
            stances.append(json.loads(results[i])["stance"])
        df["stance"] = stances
        return df

def createSession(fileName: str, logSearches = False) -> Session:
    data = parse_data(fileName)
    s = Session(data, logSearches)
    return s






if __name__=='__main__':
    print("You weren't supposed to run this")

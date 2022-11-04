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
                if(re.findall(expression, retrieveRow(i)[15]), re.M):
                    ans[i] = True
                    count += 1
        self.makeOperation(ans, count, "advancedSearch", expression)
    
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

dataSet = None
headers = None
headerDict = dict()
def retrieveRow(rowNum: int):
    return dataSet[rowNum]


def test1():
    s = Session(dataSet)
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

def test2():
    s = Session(dataSet)
    s.randomSubset(0.01)
    print(s.currentSet.size)
    s.back()
    print(s.currentSet.size)
    s.back()
    s.next()
    print(s.currentSet.size)
    s.searchKeyword(["the"])
    print(s.currentSet.size)
    s.back()
    s.back()
    s.filterBy("State", "California")
    print(s.currentSet.size)
    s.back()
    s.advancedSearch("'trump' and not 'Trump'")
    print(s.currentSet.size)
    s.back()
    s.back()
    s.printChildren()
    s.next(0)
    print(s.currentSet.size)
    s.next(0)
    print(s.currentSet.size)
    s.next()

def test3():
    s = Session(dataSet)
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

def test4():
    s = Session(dataSet)
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
    
def test5():
    s = Session(dataSet)
    s.simpleRandomSample(17000)
    #s.printCurrSubset()
    print(s.currentSet.size)
    s.back()
    s.weightedSample(10, "Retweets")
    #s.printCurrSubset()
    print(s.currentSet.size)

def test6():
    s = Session(dataSet)
    s.simpleRandomSample(30)
    s.printCurrSubset()
    print("\n\n ---------------------------------------------------------- \n")
    s.back()
    s.simpleRandomSample(30)
    s.printCurrSubset()

if __name__=='__main__':
    test = parse_data("allCensus_sample.csv")
    #print(type(test))
    dataSet = test.values
    headers = test.columns
    for i in range(len(headers)):
        colName = headers[i]
        headerDict[colName] = i
    test5()

    


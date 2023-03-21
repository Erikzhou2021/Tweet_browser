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
import time
# Ignore warnings
import warnings

#from pyparsing import null_debug_action

#from formatter import NullFormatter
warnings.filterwarnings("ignore")


# this function reads in the data (copied from online)
def parse_data(filename):
    path = './' + filename
    begin = time.perf_counter()
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
    end = time.perf_counter()
    print("time = ", end-begin)
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
    doc_word_matrices = dict()
    def __init__(self, ind: bitarray):
        self.indices = ind

class Session:
    def __init__(self, baseSet):
        self.allData = baseSet
        self.headerDict = dict()
        for i in range(len(baseSet.columns)): # put <header, columnNum> into a dictionary for faster access
            self.headerDict[baseSet.columns[i]] = i
        self.length = len(baseSet)
        arr = bitarray(self.length)
        arr.setall(1)
        self.base = Subset(arr)
        self.base.size = self.length
        self.currentSet = self.base
        self.weightable = dict()
        i = 0
        for col in self.allData.dtypes:
            if col == int or col == float:
                self.weightable[self.allData.columns[i]] = i
            i += 1
        #print(self.weightable)

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
        newOp = Operation(inputs, newSets, funcName, params)
        for i in range(len(outputs)):
            newOp.outputs[i].parent = newOp
        for i in range(len(inputs)):
            #input.children.append(deepcopy(newOp))
            #can't use append
            newOp.parents[i].children = newOp.parents[i].children + [newOp]
        self.currentSet = newSets[0]

    def printColumn(self, column: int):
        print(self.allData.iloc[toBoolArray(self.currentSet.indices), [column]])

    def getCurrentSubset(self):
        return self.allData.iloc[toBoolArray(self.currentSet.indices)]

    def printCurrSubset(self, verbose: bool = False):
        if verbose:
            print(self.allData.iloc[toBoolArray(self.currentSet.indices)].values)
        else:
            print(self.allData.iloc[toBoolArray(self.currentSet.indices), self.headerDict['Message']].values)

    def invert(self, input: bitarray):
        for i in range(len(input)):
            if input[i]:
                input[i] = False
            else:
                input[i] = True
        return input

    def randomSubset(self, probability, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        if probability > 1 or probability < 0:
            raise ValueError("Invalid probability")
        random.seed()
        ans = bitarray(self.length)
        ans.setall(0)
        count = 0
        for i in range(self.length):
            if inputSet.indices[i] and random.random() < probability:
                ans[i] = True
                count += 1
        self.makeOperation(ans, count, "randomSubset", "None")

    def simpleRandomSample(self, size: int, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        if inputSet.size < size:
            raise ValueError("Invalid sample size")
        random.seed()
        ans = bitarray(self.length)
        ans.setall(0)
        population = []
        for i in range(self.length):
            if inputSet.indices[i]:
                population.append(i)
        temp = np.random.choice(population, size, replace=False)
        for j in temp:
            ans[j] = True
        self.makeOperation(ans, size, "simpleRandomSample", size)

    def weightedSample(self, size: int, colName: str, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        if inputSet.size < size:
            raise ValueError("Invalid sample size")
        if colName not in self.weightable:
            raise ValueError("Column name does not correspond to a column that can be weighted")
        random.seed()
        ans = bitarray(self.length)
        ans.setall(0)
        population = []
        weights = []
        sum = 0
        for i in range(self.length):
            if inputSet.indices[i]:
                population.append(i)
                value = self.allData.iloc[i].at[colName]
                if value != value: 
                    value = 0
                value += 1
                sum += value
                weights.append(int(value))
        for j in range(len(weights)):
            weights[j] = float(weights[j] / sum)  
        temp = np.random.choice(population, size, replace=False, p=weights)
        for k in temp:
            ans[k] = True
        self.makeOperation(ans, size, "weightedSample", colName + str(size))

    def searchKeyword(self, keywords: list, orMode: bool = False, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        ans = bitarray(self.length)
        ans.setall(0)
        count = 0
        for i in range(self.length):
            if inputSet.indices[i]:
                if orMode:
                    include = False
                    for j in keywords:
                        #if self.allData.iloc[i].at["Message"].find(j) != -1: # might be slow
                        pattern = r"\b" + re.escape(j) + r"\b"
                        if re.search(pattern, self.allData.iloc[i].at["Message"]):
                            include = True
                            break
                else:
                    include = True
                    for j in keywords:
                        #if self.allData.iloc[i].at["Message"].find(j) == -1:
                        pattern = r"\b" + re.escape(j) + r"\b"
                        if not (re.search(pattern, self.allData.iloc[i].at["Message"])):
                            include = False
                            break
                if include:
                    ans[i] = True
                    count += 1
        self.makeOperation(ans, count, "searchKeyword", keywords)

    def advancedSearch(self, expression: str, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
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
                    #if(self.allData.iloc[i].at["Message"].find(j[1:-1]) > -1): # might be slow
                    pattern = r"\b" + re.escape(j[1:-1]) + r"\b"
                    if re.search(pattern, self.allData.iloc[i].at["Message"]):
                        newExpression = newExpression.replace(j, " True")
                    else:
                        newExpression = newExpression.replace(j, " False")
                if(eval(newExpression)):
                    ans[i] = True
                    count += 1
        self.makeOperation(ans, count, "advancedSearch", expression)
        

    def regexSearch(self, expression: str, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        ans = bitarray(self.length)
        ans.setall(0)
        count = 0
        for i in range(self.length):
            if(inputSet.indices[i]):
                if(re.findall(expression, self.allData.iloc[i].at["Message"], re.M)): #might be slow
                    ans[i] = True
                    count += 1
        self.makeOperation(ans, count, "regexSearch", expression)
    
    def filterBy(self, colName: str, value, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        ans = bitarray(self.length)
        ans.setall(0)
        count = 0
        for i in range(self.length):
            if inputSet.indices[i] and self.allData.iloc[i].at[colName]== value: # might be slow
                ans[i] = True
                count += 1
        self.makeOperation(ans, count, "filterBy", colName + " = " + value)

    def setDiff(self, setOne: Subset, setZero: Subset = None):
        if setZero == None or type(setZero) != Subset:
            setZero = self.currentSet
        if type(setOne) != Subset:
            raise TypeError("Set operators require two subset objects")
        ans = bitarray(self.length)
        ans.setall(0)
        count = 0
        for i in range(self.length):
            if setZero.indices[i] and not setOne.indices[i]:
                ans[i] = True
                count += 1
        self.makeOperation(ans, count, "setDiff", setOne)

    def setUnion(self, setOne: Subset, setZero: Subset = None):
        if setZero == None or type(setZero) != Subset:
            setZero = self.currentSet
        if type(setOne) != Subset:
            raise TypeError("Set operators require two subset objects")
        ans = bitarray(self.length)
        ans.setall(0)
        count = 0
        for i in range(self.length):
            if setZero.indices[i] or setOne.indices[i]:
                ans[i] = True
                count += 1
        self.makeOperation(ans, count, "setUnion", setOne)

    def setIntersect(self, setOne: Subset, setZero: Subset = None):
        if setZero == None or type(setZero) != Subset:
            setZero = self.currentSet
        if type(setOne) != Subset:
            raise TypeError("Set operators require two subset objects")
        ans = bitarray(self.length)
        ans.setall(0)
        count = 0
        for i in range(self.length):
            if setZero.indices[i] and setOne.indices[i]:
                ans[i] = True
                count += 1
        self.makeOperation(ans, count, "setintersect", setOne)

    def back(self, index: int = 0):
        if(self.currentSet.size == self.length) or index >= len(self.currentSet.parent.parents):
        # if(self.currentSet == self.base):
            raise IndexError("Can't go back (Out of bounds)")
        self.currentSet = self.currentSet.parent.parents[index]
    
    def next(self, setIndex = -1, opIndex = 0):
        if len(self.currentSet.children) == 0 or setIndex >= len(self.currentSet.children):
            raise IndexError("Can't got next (Out of bounds)")
        self.currentSet = self.currentSet.children[setIndex].outputs[opIndex]

    def printChildren(self):
        if len(self.currentSet.children) == 0:
            print("No children searches")
            return
        for i in self.currentSet.children:
            print("Type = ", i.operationType, " parameters = ", i.parameters)
    
    ##### Clustering ######
    
    # functions for dimension reduction: PCA and UMAP
    def dimred_PCA(self, docWordMatrix, ndims=25):
        tsvd = TruncatedSVD(n_components=ndims)
        tsvd.fit(docWordMatrix)
        docWordMatrix_pca = tsvd.transform(docWordMatrix)
        return docWordMatrix_pca

    def dimred_UMAP(self, matrix, ndims=2, n_neighbors=15):
        umap_2d = umap.UMAP(n_components=ndims, random_state=42, n_neighbors=n_neighbors, min_dist=0.0)
        proj_2d = umap_2d.fit_transform(matrix)
        #proj_2d = umap_2d.fit(matrix)
        return proj_2d

    # functions for clustering
    # HDBSCAN
    def cluster_hdbscan(self, points, min_obs):
        hdbscan_fcn = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=min_obs)
        clusters = hdbscan_fcn.fit_predict(points).astype(str)
        return clusters

    # Gaussian Mixure Models
    def cluster_gmm(self, points, num_clusters):
        gmm_fcn = GaussianMixture(n_components=num_clusters, random_state=42).fit(points)
        clusters = gmm_fcn.predict(points).astype(str)
        return clusters

    # K-Means
    def cluster_kmeans(self, points, num_clusters):
        kmean_fcn = KMeans(init='random', n_clusters=num_clusters, random_state=42)
        clusters = kmean_fcn.fit(points).labels_.astype(str)
        return clusters


    def cluster_polis_leiden(self, points, num_neighbors):
        A = kneighbors_graph(points, num_neighbors, mode="connectivity", metric="euclidean", 
        p=2, metric_params=None, include_self=True, n_jobs=None)

        sources, targets = A.nonzero()
        weights = A[sources, targets]
        if isinstance(weights, np.matrix): # ravel data
            weights = weights.A1

        g = ig.Graph(directed=False)
        g.add_vertices(A.shape[0])  # each observation is a node
        edges = list(zip(sources, targets))
        g.add_edges(edges)
        g.es['weight'] = weights
        weights = np.array(g.es["weight"]).astype(np.float64)

        part = leidenalg.find_partition(
            g, 
            leidenalg.ModularityVertexPartition
        );

        leidenClusters = np.array(part.membership).astype(str)
        leidenClustersStr = [str(i) for i in leidenClusters]
    
        return leidenClusters

    def make_full_docWordMatrix(self, min_df = 5, inputSet: Subset = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        if inputSet.size == 0:
            return
        if min_df in inputSet.doc_word_matrices:
            return inputSet.doc_word_matrices[min_df][0], inputSet.doc_word_matrices[min_df][1]
        cleanedTweets = []
        
        for i in range(self.length):
            if inputSet.indices[i]:
                cleanedTweets.append(preProcessingFcn(self.allData.iloc[i].at["Message"])) # might be slow

        # create document-word matrix
        vectorizer = CountVectorizer(strip_accents='unicode', min_df= min_df, binary=False)
        docWordMatrix_orig = vectorizer.fit_transform(cleanedTweets)
        docWordMatrix_orig = docWordMatrix_orig.astype(dtype='float64')
        names = vectorizer.get_feature_names_out()
        inputSet.doc_word_matrices[min_df] = [docWordMatrix_orig, names]
        return docWordMatrix_orig, names
        #return docWordMatrix_orig.tolil(), vectorizer.get_feature_names()


    def dimRed_and_clustering(self, docWordMatrix_orig, 
    dimRed1_method, dimRed1_dims, clustering_when, clustering_method, 
    num_clusters, min_obs, num_neighbors, dimRed2_method = None, inputSet = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        # read in document-word matrix
        # data = docWordMatrix_orig.data
        # rows, cols = docWordMatrix_orig.nonzero()
        # dims = docWordMatrix_orig.shape   
        # docWordMatrix = csc_matrix((data, (rows, cols)), shape=(dims[0], dims[1]))
        begin = time.perf_counter()
        docWordMatrix = docWordMatrix_orig.tocsc()

        # do stage 1 dimension reduction
        if dimRed1_method == 'pca':
            dimRed1 = self.dimred_PCA(docWordMatrix, docWordMatrix_orig.shape[1])
        elif dimRed1_method == 'umap':
            #dimRed1 = self.dimred_UMAP(docWordMatrix, docWordMatrix_orig.shape[1])
            dimRed1 = self.dimred_UMAP(docWordMatrix, dimRed1_dims)
        else:
            raise ValueError("Dimension reduction method can be either 'pca' or 'umap'")
        # do stage 2 dimension reduction (if any)
        if dimRed1_dims > 2:
            if dimRed2_method == 'pca':
                dimRed2 = self.dimred_PCA(dimRed1, ndims=2)
            elif dimRed2_method == 'umap':
                dimRed2 = self.dimred_UMAP(dimRed1, ndims=2)
            else:
                raise ValueError("Dimension reduction method can be either 'pca' or 'umap'")
        else:
            dimRed2 = dimRed1
        # Clustering
        # get matrix at proper stage
        if clustering_when == 'before_stage1':
            clustering_data = docWordMatrix
        elif clustering_when == 'btwn':
            clustering_data = dimRed1
        elif clustering_when == 'after_stage2':
            if dimRed1_dims < 2:
                raise ValueError("Can't cluster after stage 2 if stage 2 is unecessary (dimRed1_dims < 2)")
            clustering_data = dimRed2
        else: # also have to check if 'after_stage2' is used only when there is a stage 2
            raise ValueError("clustering_when should be in [before_stage1, btwn, after_stage2]")
        # perform clustering
        if clustering_method == 'gmm':
            if clustering_when == 'before_stage1':
                clustering_data = clustering_data.toarray()
            clusters = self.cluster_gmm(clustering_data, num_clusters=num_clusters)
        elif clustering_method == 'k-means':
            clusters = self.cluster_kmeans(clustering_data, num_clusters=num_clusters)
        elif clustering_method == 'hdbscan':
            clusters = self.cluster_hdbscan(clustering_data, min_obs=min_obs)
        elif clustering_method == 'leiden':
            clusters = self.cluster_polis_leiden(clustering_data, num_neighbors=num_neighbors)
        else:
            raise ValueError("Clustering method must be in the list [gmm, k-means, hdbscan, leiden]")

        outputs = list()
        counts = list()
        for i in range(num_clusters):
            ans = bitarray(self.length)
            ans.setall(0)
            outputs.append(ans)
            counts.append(0)
        counter = 0
        for i in range(self.length):
            if inputSet.indices[i]:
                outputs[int(clusters[counter])][i] = True
                counts[int(clusters[counter])] += 1
                counter += 1
        self.makeOperation(outputs, counts, "Clustering", [dimRed1_method, dimRed1_dims, clustering_when, clustering_method, 
    num_clusters, min_obs, num_neighbors, dimRed2_method]) # change to json later
        self.currentSet = inputSet

        allMessages_plot = self.allData.iloc[toBoolArray(inputSet.indices)]
        allMessages_plot['Cluster'] = clusters # color by cluster
        allMessages_plot['Text'] = allMessages_plot['Message'].apply(lambda t: "<br>".join(textwrap.wrap(t))) # make tweet text display cleanly
        allMessages_plot['coord1'] = dimRed2[:,0] # x-coordinate
        allMessages_plot['coord2'] = dimRed2[:,1] # y-coordinate
        dimRed_cluster_plot = px.scatter(allMessages_plot, x='coord1', y='coord2', color='Cluster',
            hover_data=['Text'], category_orders={"Cluster": [str(int) for int in list(range(50))]})
        # dimRed_cluster_plot = px.scatter(x= dimRed2[:,0], y= dimRed2[:,1], color= clusters, 
        # data_frame= inputSet.indices)
        # dimRed_cluster_plot.show()
        # dimRed_cluster_plot.update_layout(clickmode='event+select')
        end = time.perf_counter()
        print("cluster time =", end-begin)
        return dimRed_cluster_plot

def createSession(fileName: str) -> Session:
    data = parse_data(fileName)
    s = Session(data)
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
    matrix, words = s.make_full_docWordMatrix(min_df= 1)
    #print(words)
    test = s.dimRed_and_clustering(matrix, dimRed1_method='pca', dimRed1_dims= 2, dimRed2_method='umap', 
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
    matrix, words = s.make_full_docWordMatrix(min_df= 1)
    test = s.dimRed_and_clustering(matrix, dimRed1_method='pca', dimRed1_dims= 2, dimRed2_method='umap', 
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
    test = s.dimRed_and_clustering(matrix, dimRed1_method= 'pca', dimRed1_dims=2, clustering_when='before_stage1', 
        clustering_method='gmm', num_clusters=2, min_obs= 2, num_neighbors=2)
    #print(test)
    test.show()

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
    s.weightedSample(100, "Retweets")
    print(s.currentSet.size)


if __name__=='__main__':
    s = createSession("allCensus_sample.csv")

    test9(s)


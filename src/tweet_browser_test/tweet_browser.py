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
    def __init__(self, ind: bitarray):
        self.indices = ind

class Session:
    base = None
    currentSet = None
    length = 0
    #allData = None
    def __init__(self, baseSet):
        self.allData = baseSet
        self.headerDict = dict()
        for i in range(len(baseSet.columns)):
            colName = baseSet.columns[i]
            self.headerDict[colName] = i
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
        #can't use appendgetCurrentSubset
        newOp.parents[0].children = newOp.parents[0].children + [newOp]
        self.currentSet = newSet

    def printColumn(self, column: int):
        print(self.allData.iloc[self.currentSet.indices, [column]])
        # for i in range(self.length):
            # if (self.currentSet.indices[i]):
            #     print(retrieveRow(i)[column])

    def getCurrentSubset(self):
        return self.allData.iloc[self.currentSet.indices]

    def printCurrSubset(self, verbose: bool = False):
        if verbose:
            print(self.allData.iloc[toBoolArray(self.currentSet.indices)])
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
            raise ValueError("Invalid sample size")
        population = []
        for i in range(self.length):
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
            raise ValueError("Invalid sample size")
        population = []
        weights = []
        sum = 0
        for i in range(self.length):
            if inputSet.indices[i]:
                population.append(i)
                value = self.allData.iloc[i].at[colName]
                if value != value : # still need to check if the colName corresponds with a number that can be weighted
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
                        if (self.allData.iloc[i].at["Message"].find(j) != -1): # might be slow
                            include = True
                            break
                else:
                    include = True
                    for j in keywords:
                        if (self.allData.iloc[i].at["Message"].find(j) != -1):
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
                    if(self.allData.iloc[i].at["Message"].find(j[1:-1]) > -1): # might be slow
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
                if(re.findall(expression, self.allData.iloc[i].at["Message"], re.M)): #might be slow
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
            if (inputSet.indices[i] and self.allData.iloc[i].at[colName]== value): # might be slow
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
        # if(self.currentSet == self.base):
            raise IndexError("Can't go back (Out of bounds)")
        self.currentSet = self.currentSet.parent.parents[index]
    
    def next(self, index = -1):
        if len(self.currentSet.children) == 0 or index >= len(self.currentSet.children):
            raise IndexError("Can't got next (Out of bounds)")
        self.currentSet = self.currentSet.children[index].outputs[0]

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
        return proj_2d

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
        if (inputSet == None):
            inputSet = self.currentSet
        if inputSet.size == 0:
            return
        cleanedTweets = []
        for i in range(self.length):
            if inputSet.indices[i]:
                cleanedTweets.append(preProcessingFcn(self.allData.iloc[i].at["Message"])) # might be slow

        # create document-word matrix
        vectorizer = CountVectorizer(strip_accents='unicode', min_df= min_df, binary=False)
        docWordMatrix_orig = vectorizer.fit_transform(cleanedTweets)
        docWordMatrix_orig = docWordMatrix_orig.astype(dtype='float64')
        return docWordMatrix_orig, vectorizer.get_feature_names()

    def dimRed_and_clustering(self, docWordMatrix_orig, 
    dimRed1_method, dimRed1_dims, clustering_when, clustering_method, 
    num_clusters, min_obs, num_neighbors, dimRed2_method = None, inputSet = None):
        if (inputSet == None):
            inputSet = self.currentSet
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
        
        allMessages_plot = self.allData.iloc[toBoolArray(inputSet.indices)]
        allMessages_plot['Cluster'] = clusters # color by cluster
        allMessages_plot['Text'] = allMessages_plot['Message'].apply(lambda t: "<br>".join(textwrap.wrap(t))) # make tweet text display cleanly
        allMessages_plot['coord1'] = dimRed2[:,0] # x-coordinate
        allMessages_plot['coord2'] = dimRed2[:,1] # y-coordinate
        dimRed_cluster_plot = px.scatter(allMessages_plot, x='coord1', y='coord2', color='Cluster',
            hover_data=['Text'], category_orders={"Cluster": [str(int) for int in list(range(50))]})
        return dimRed_cluster_plot

def createSession(fileName: str) -> Session:
    data = parse_data(fileName)
    s = Session(data)
    return s

if __name__=='__main__':
    print("You weren't supposed to run this")

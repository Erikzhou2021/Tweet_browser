import warnings
warnings.filterwarnings("ignore")

import time
import_time = time.perf_counter()
import copy
import numpy as np
import pandas as pd
import random
import re
from enum import Enum
import string
import scipy
import sklearn
import nltk
#nltk.download('stopwords')
#from nltk.corpus import stopwords

from fastlexrank import FastLexRankSummarizer

import pickle
import datetime

import sys
# sys.path.insert(0, '/mnt/c/Users/erikz/Documents/Tweet_browser/Tweet_browser/src/tweet_browser_test')
sys.path.insert(0, '/mnt/c/Users/erikz/Documents/Tweet_browser/Tweet_browser/Frontend')
from tweet_browser import Session, Subset




end_import = time.perf_counter()
print("import time = ", end_import - import_time)

fileName = "Data/Session.pkl"

# this function reads in the data (copied from online)
def parse_data(filename, header='infer'):
    path = './' + filename
    begin = time.perf_counter()
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

def createSession(fileName: str, logSearches = False, useEmbeddings = False) -> Session:
    data = parse_data(fileName)
    embeddings = None
    if useEmbeddings:
        embeddings = pd.read_csv("allCensus_sample_embeddings.csv", encoding = "utf-8", index_col=0)
    s = Session(data, logSearches, embeddings)
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
    s.simpleRandomSample(170)
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
    s.simpleRandomSample(170)
    print(s.currentSet.size)
    tempSet = s.currentSet
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

def test9(s):
    s.removeRetweets()
    set1 = s.currentSet
    print(set1.size)
    s.back()
    s.filterDate("2020-01-01", "2020-06-01")
    set2 = s.currentSet
    print(set2.size)
    s.back()
    s.searchKeyword(["test"])
    set3 = s.currentSet
    print(set3.size)
    s.setIntersect(set1)
    print(s.currentSet.size)
    s.resetToBase()
    s.setIntersect([set1, set3])
    print(s.currentSet.size)
    s.resetToBase()
    s.setIntersect([set1, set2, set3])
    print(s.currentSet.size)

def test10(s):
    pass

def test11(s):
    #begin = time.perf_counter()
    s.filterBy("SenderGender", "FEMALE")
    #end = time.perf_counter()
    #print("time = ", end-begin)
    print(s.currentSet.size)
    s.weightedSample(10, "Sender Followers Count")
    s.simpleRandomSample(5)
    # s.printCurrSubset(True)
    s.back()
    s.weightedSample(5, "Retweets")
    print(s.currentSet.size)

def test12(s):
    try: 
        s.weightedSample(100, "SenderScreenName")
    except (ValueError):
        print("exeption caught")
    
def test13(s):
    pass

def test14(s):
    pass

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

def test18(s):
    print(s.currentSet.size)
    s.filterDate("2020-09-15", "2021-09-15")
    print(s.currentSet.size)
    assert s.currentSet.size == 18

#exclude function
def test19(s):
    s.exclude(["a", "the"])
    print(s.currentSet.size)

# test filter by
def test20(s):
    s.filterBy("State", "California")
    assert(s.currentSet.size == 1347)
    s.back()
    s.filterBy("SenderGender", "OTHER")
    assert(s.currentSet.size == 10136)
    s.filterBy("SenderGender", "other")
    assert(s.currentSet.size == 0)
    s.back()
    s.back()
    s.filterBy("MessageType", "Twitter Update")
    assert(s.currentSet.size == 2744)
    s.back()
    s.filterBy("MessageType", "Twitter Mention")
    assert(s.currentSet.size == 897)
    s.back()
    s.filterBy("Sender Followers Count", 100)
    assert(s.currentSet.size == 25)

# test remove retweets
def test21(s):
    s.removeRetweets()
    assert(s.currentSet.size == 5080)

def test22(s):
    s.searchKeyword(["test"])
    assert(s.currentSet.size == 26)
    s.searchKeyword(["COVID"])
    assert(s.currentSet.size == 2)
    s.back()
    s.back()
    s.searchKeyword(["test", "COVID"])
    assert(s.currentSet.size == 2)
    s.back()
    s.searchKeyword(["test"], caseSensitive = True)
    assert(s.currentSet.size == 22)
    s.back()
    s.searchKeyword(["covid", "vaccine"], True, True)
    assert(s.currentSet.size == 65)
    s.back()
    s.searchKeyword(["census", "the", "poll"], False)
    assert(s.currentSet.size == 14)
    print(s.currentSet.size)

def test23(s):
    s.advancedSearch("'covid' and ('hospital' or 'vaccine')")
    assert(s.currentSet.size == 1)
    s.back()
    s.advancedSearch("'not' and 'census' and ('covid' or 'COVID')")
    assert(s.currentSet.size == 10)
    s.back()
    s.advancedSearch("and or not False True")

def test24(s):
    s.searchKeyword(["test"])
    # result = s.getCentral()
    # print(result[["centrality", "Message"]])
    result = s.summarize()
    # print(result)
    text, tweets, unused = s.parseSummary(result)
    # print(text, tweets)
    s.back()
    s.searchKeyword(["trump"])
    s.simpleRandomSample(50)
    result = s.summarize()
    print(result)
    text, tweets, unused = s.parseSummary(result)
    print(text, tweets, unused)

def test25(s):
    subset = Subset(range(30))
    subset.size = 30
    input = """Lorem ipsum dolor sit amet, consectetur adipiscing elit (0, 2, 10),
    sed do eiusmod tempor (20, 22, 10) incididunt ut labore et
    dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut (2021)
    aliquip ex ea commodo consequat. Duis aute (5, 3) irure dolor in reprehenderit in voluptate velit esse cillum (2021, 2022)
    dolore eu fugiat nulla pariatur. Excepteur sint (abcd) occaecat cupidatat non proident, sunt in culpa qui officia 
    deserunt mollit (ABC123) anim id est laborum (123)."""
    text, tweets, unused = s.parseSummary(input, subset)
    print(text, tweets, unused)
    # assert(text == ['Lorem ipsum dolor sit amet, consectetur adipiscing elit, ', '\n    sed do eiusmod tempor ', ' incididunt ut labore et\n    dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut ', '\n    aliquip ex ea commodo consequat. Duis aute ', ' irure dolor in reprehenderit in voluptate velit esse cillum ', '\n    dolore eu fugiat nulla pariatur. Excepteur sint (abcd) occaecat cupidatat non proident, sunt in culpa qui officia \n    deserunt mollit (ABC123) anim id est laborum.'])
    # assert(tweets == [['RT @realDailyWire New Census Data Suggests Previous Report Wrong, Republican States Gaining Seats In Congress  dailywire.com/news/new-censu…', 'RT @CatsTalkBack1 Sometimes the truth is hard to swallow...but still may be the truth.  The citizenship question on the census was hurried to meet a deadline...and was fumbled by the administration.   But we have the research in hand..perhaps another try??\r\n\r\nthefederalist.com/2020/01/01/cen…', 'RT @poesyomamajokes In England and Wales the population of blacks is proportionately smaller than Americans but just as many incidents can occur. black people experienced 12% of use-of-force incidents in 2017-18, despite accounting for just 3.3% of the census.'], ['Now, NPR and census are anti-secular newindianexpress.com/opinions/2019/… via @NewIndianXpress', 'RT @iheartmindy AOC might lose her district after the census due to it having too many illegals and not enough citizens.\r\n\r\nIt really starts to make sense why the Dems have been pushing so hard to try and count them. They know their demographic isn’t law abiding Americans.\r\nthegatewaypundit.com/2020/01/what-a…', 'RT @poesyomamajokes In England and Wales the population of blacks is proportionately smaller than Americans but just as many incidents can occur. black people experienced 12% of use-of-force incidents in 2017-18, despite accounting for just 3.3% of the census.'], [], ["RT @1776Stonewall New York is expected to lose a House seat after the 2020 census, and AOC might be the odd man out, as her district is expected to be redrawn. . By the way, residents in AOC's district are over 25% illegal", "RT @WWG1WGA_WW WHAT A SHAME: AOC Cld Lose Her House Seat After 2020 Through Elimination Of Her District\r\n\r\nAOC cld soon be out of her gov't job. Due to the high number of non-citizens in her district, it might be eliminated after the census.\r\n\r\nWldn’t that be ironic?👍🏻😂🤣\r\n\r\nthegatewaypundit.com/2020/01/what-a…"], [], []])
    # assert(unused == [1, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26, 27, 28, 29])
    # print(text, tweets, unused)

def test26(s):
    s.searchKeyword(["election"])
    s.semanticSearch("2020 U.S. election", 0.1)
    s.back()
    s.back()
    s.searchKeyword(["68", "days", "until", "election"])
    df = s.allData.iloc[s.currentSet.indices]
    print(df['SimilarityScore'])

def test27(s):
    subset = Subset(range(30))
    subset.size = 30
    s.semanticSearch("pandemic", 0.5, subset)

def test28(s):
    # s.semanticSearch("Citizenship question", 0.005)
    s.simpleRandomSample(50)
    topic = "Should you fill out the census?"
    examples = {"Don't miss your chance to be counted! The US Census only comes once every 10 years. 2020 Census data will help inform how billions of dollars are distributed to states and communities every year! Visit 2020census.gov @uscensusbureau #Census2020 #CompleteCount #NKYcounts https://t.co/BPgocWkKps": 0,
                "F the #2020Census With the @GOP in complete control over our lives, I don't want to give them any more ammunition. #FuckGerrymanderingGOP #DitchMitch2020 #VoteAmyMcGrath @AmyMcGrathKY": 1}
    # stances = ["Yes, the citizenship question should be included", "No, it should not", "", ""]
    stances = ["yes", "no", "", ""]
    result = s.stanceAnalysis(topic, stances, examples)
    print(result["stance"])

def test99(s):
    columns = []
    nulls = s.allData.isnull().values.all(axis=0)
    for i in range(len(nulls)):
        if nulls[i] == False:
            columns.append(s.allData.columns[i])
    print(columns)
    # for col in s.allData.columns:
        

def allTests(s1):
    current_module = __import__(__name__)
    for i in range(1,24):
        s = copy.deepcopy(s1)
        print("---------------------------")
        print("test ", i)
        print("---------------------------")
        func = getattr(current_module, "test{}".format(i))
        func(s) 

if __name__=='__main__':
    #s = createSession("allCensus_sample.csv")

    s = createSession("allCensus_sample.csv", False, True)

    #with open(fileName, "rb") as input:
        #s = pickle.load(input) 
    # allTests(s)

    begin = time.perf_counter()
    test28(s)
    # allTests(s)
    print("total time", time.perf_counter() - begin)
    # begin = time.perf_counter()
    # test99(s)
    # print("second time", time.perf_counter() - begin)

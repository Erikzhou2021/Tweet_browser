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

if __name__=='__main__':
    test = parse_data("allCensus_sample.csv")
    tweet_tester.dataSet = test.values
    tweet_tester.headers = test.columns
    for i in range(len(tweet_tester.headers)):
        colName = tweet_tester.headers[i]
        tweet_tester.headerDict[colName] = i

    tweet_tester.test5()

    


from tweet_browser_test import tweet_browser as tb
import pickle

input = open("Session.pkl", "rb")
s = pickle.load(input)

if s.currentSet.size != s.length:
    s.back()

s.printChildren()


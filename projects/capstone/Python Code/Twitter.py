# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 10:14:48 2017

@author: Carl
"""

import got
from textblob import TextBlob
import numpy as np
import cPickle
import os
import pandas as pd

from datetime import timedelta

class Twitter:
    
    def __init__(self, pickle_dir):
        
        self.pickle_dir = pickle_dir
    
    def get_days_tweets(self, search_term, date, max_tweets, language):
        
        start_date = date - timedelta(days=1)
        end_date = date
        
        start_date_string = start_date.strftime('%Y-%m-%d')
        end_date_string = end_date.strftime('%Y-%m-%d')
        
        filename = 'tweets_' + search_term + '_' + end_date_string + '_' +  str(max_tweets) + '_' + language + '.pkl'
        
        if os.path.isfile(os.path.join(self.pickle_dir, filename)):
            tweets = self.unpickle_data(filename)
        else:
            tweetCriteria = got.manager.TweetCriteria().setQuerySearch(search_term).setSince(start_date_string).setUntil(end_date_string).setMaxTweets(max_tweets).setLanguage(language)
            tweets = got.manager.TweetManager.getTweets(tweetCriteria)
            self.pickle_data(tweets, filename)
            
        return tweets

    def calculate_polarity(self, tweets):
        
        polarity = []

        for iii, tweet in enumerate(tweets):
            text_blob = TextBlob(tweet.text)
            polarity.append(text_blob.polarity)
            
        return polarity
    
    def pickle_data(self, data, filename):
        
        save_file_name = os.path.join(self.pickle_dir, filename)
        output_file = open(save_file_name, 'wb')
        cPickle.dump(data, output_file)
        output_file.close()
        
    def unpickle_data(self, filename):
        
        load_file_name = os.path.join(self.pickle_dir, filename)
        input_file = open(load_file_name, 'rb')
        data = cPickle.load(input_file)
        input_file.close()
        
        return data
        
    def print_tweet(self, t):
        print "Username: %s" % t.username
        print "Retweets: %d" % t.retweets
        print "Text: %s" % t.text
        print "Mentions: %s" % t.mentions
        print "Hashtags: %s\n" % t.hashtags
        
    def get_polarity_from_all_tweets(self, search_term, start_date, end_date, max_tweets, language):
        
        start_date_string = start_date.strftime('%Y-%m-%d')
        end_date_string = end_date.strftime('%Y-%m-%d')
        
        filename = 'polarity_' + search_term + '_' + start_date_string + '_' + end_date_string + '_' +  str(max_tweets) + '_' + language + '.pkl'
        
        if os.path.isfile(os.path.join(self.pickle_dir, filename)):
            polarity = self.unpickle_data(filename)
        else:
            #getnerate all weekdays
            date_range = pd.bdate_range(start_date, end_date)
            
            polarity = pd.DataFrame(index=date_range, columns=['data','mean','std'])
            
            for date in date_range:
                tweets = self.get_days_tweets(search_term, date, max_tweets, language)
                polarity.loc[date]['data'] = self.calculate_polarity(tweets)
                polarity.loc[date]['mean'] = np.mean(polarity.loc[date]['data'])
                polarity.loc[date]['std'] = np.std(polarity.loc[date]['data'])
            self.pickle_data(polarity, filename)
                
        return polarity
    
if __name__ == "__main__":

    import datetime
        
    twitter = Twitter('./PickledTweets')
    
    start_date = datetime.date(2017,4,1)
    end_date = datetime.date(2017,4,5)
    
    polarity = twitter.get_polarity_from_all_tweets('google', start_date, end_date, 5, 'en')
    
    print polarity
    
    filename = 'tweets_google_2017-04-04_5_en.pkl'
    tweets = twitter.unpickle_data(filename)
    
    twitter.print_tweet(tweets[10])
    
    
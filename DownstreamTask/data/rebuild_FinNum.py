#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 09:41:57 2018

@author: chungchi
"""

#use browser to get code
#code_url = "https://api.stocktwits.com/api/2/oauth/authorize?client_id=" + consumer_key + "&response_type=code&redirect_uri=" + redirect_url + "&scope=read,watch_lists,publish_messages,publish_watch_lists,direct_messages,follow_users,follow_stocks"


consumer_key = "consumer_key"
consumer_secret = "consumer_secret"
redirect_url = "redirect_url"
code = "code"

token_url = "https://api.stocktwits.com/api/2/oauth/token?client_id=" + consumer_key + "&client_secret=" + consumer_secret + "&code=" + code + "&grant_type=authorization_code&redirect_uri=" + redirect_url

import json
import time
import requests

token_info = requests.post(token_url)
token = json.loads(token_info.content.decode("utf8"))["access_token"]

for FinNum in ["FinNum_training","FinNum_dev","FinNum_test"]:
    with open(FinNum + ".json") as f:
        data = json.load(f)
    
    #Authenticated calls are permitted 400 requests per hour and measured against the access token used in the request.
    not_found = []
    twt = []
    idx = []
    
    i = 0
    while(i != len(data)):
        print(i)
        if(data[i]["idx"] in idx):
            j = idx.index(data[i]["idx"])
            data[i]["tweet"] = twt[j]
            i = i + 1
            continue
        
        url = "https://api.stocktwits.com/api/2/messages/show/" + str(data[i]["id"]) + ".json?access_token=" + token
        tweet_info = json.loads(requests.get(url).content.decode("utf8"))
    
        if(tweet_info["response"]["status"] == 200):
            tweet = tweet_info["message"]["body"]
            data[i]["tweet"] = tweet
            twt.append(tweet)
            idx.append(data[i]["idx"])
            i = i + 1
        elif(tweet_info["response"]["status"] == 429):
            print("sleep one hour----from " + time.ctime() )
            time.sleep(3600)
        else:
            not_found.append(i)
            print(i)
            print(tweet_info)
            i = i + 1
            
    for i in not_found[::-1]:
        del data[i]
    
    print("Missing data: " + str(len(not_found)))
    print("Total: " + str(len(data)) + " instances")
    
    json.dump(data, open(FinNum + "_rebuilded.json", 'w'))
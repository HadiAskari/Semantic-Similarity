import pickle 
import pandas as pd
import spacy
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm
tqdm.pandas()
import re
import transformers
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk import tokenize
import pickle
import urllib.request as requests
import json
import statistics
import torch
from collections import defaultdict
import copy
import math

model = SentenceTransformer('all-mpnet-base-v2')

df=pd.read_csv('gmo_to_uniqueId.csv')
materialID=df['lastRunJob'].to_list()

# with open('PreProcessed_segmented.pkl', 'rb') as f:
#     newmetalist=pickle.load(f)

with open('New_PreProcessed_segmented.pkl', 'rb') as f:
    newnewmetalist=pickle.load(f)

sampled_datadf=pd.read_csv("Sampled_Adword_Labeled.csv")

#For Normal process with target labels only

df1=pd.read_csv('AllTiersAdword.csv')
target_dict=dict(zip(df1['Name'], df1.index))

def clean(text):
    text=text.lower()
    text=text.replace("(", "")
    text=text.replace(")", "")
    text=text.replace(";", "")
    text=text.replace(",", "")
    text=text.replace("+", "")
    text=text.replace(".", "")
    text=text.replace("&", "")
    return text

targetlist=df1['Name'].to_list()

truthdatadf=pd.read_csv('AdwordLabeledDatabase.csv')

def getadword(episode, truthdatadf):
    for i in range(len(truthdatadf.lastRunJob)):
        if episode == truthdatadf.lastRunJob[i]:
            return truthdatadf.adword[i]

listofepisodes=sampled_datadf['lastRunJob'].to_list()

def keydict(dataframe, targetlist):
    d=defaultdict(list)
    for target in targetlist:
        try:

            minidf=dataframe.loc[dataframe['adword']==target]
            minilist=minidf['lastRunJob'].to_list()
            d[target]=minilist
        except:
            continue
    
    return d

targetkey_filelist=keydict(sampled_datadf, targetlist)

def get_key(val):
    for key, value in target_dict.items():
         if val == value:
             return key
 
    return "key doesn't exist"

dad=[]
for i in range(len(newnewmetalist)):
    mom=[]
    for j in range(len(newnewmetalist[i])):
        if len(newnewmetalist[i][j])==0:
            continue
        mom.append(newnewmetalist[i][j])
    dad.append(mom)

#Relevance calculation per episode top1
dictofrelevance=copy.deepcopy(targetkey_filelist)
for key, value in tqdm(targetkey_filelist.items()): #for each IAB Label
    if len(targetkey_filelist[key])==0:
        continue
    for i in tqdm(range(len(targetkey_filelist[key]))): #for each File
        index=materialID.index(targetkey_filelist[key][i])
        x=target_dict[key] #Index of the target label
        MaxAggregation=defaultdict(list)

        for l in range(len(dad[index])): #For each Segment
            # print(len(newmetalist[index]))
            if len(dad[index][l])==0: 
                continue
            emb1=model.encode(dad[index][l]) #, convert_to_tensor=True)
            emb2=model.encode(targetlist) #, convert_to_tensor=True)
            cos_sim=util.cos_sim(emb1,emb2)

        #Add all pairs to a list with their cosine similarity score
        
            for k in range(cos_sim.shape[1]):
                all_sentence_combinations=[]
                for j in range(cos_sim.shape[0]):
                    all_sentence_combinations.append([cos_sim[j][k], j, k])
                all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)
                MaxAggregation[k].append(all_sentence_combinations[0])
    
     
        dictofrelevance[key][i]={dictofrelevance[key][i]:MaxAggregation}
     

with open('Relevance_Top1_Tiers.pkl', 'wb') as f:
    pickle.dump(dictofrelevance, f)   
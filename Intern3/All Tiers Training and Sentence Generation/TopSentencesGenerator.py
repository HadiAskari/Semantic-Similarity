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

with open('Relevance_Top1_Tiers.pkl', 'rb') as f:
     Relevance_dict=pickle.load(f)

def topksentences(dict):
    d=defaultdict(list)
    adwords=[]
    texts=[]
    scores=[]
    for adword in tqdm(target_dict.keys()):
        scoreforfiles=[]
        for k,v in tqdm(dict.items()):#for each label
            if len(dict[k]) == 0:
                continue
            # scoreforfiles=[]
    
            for i in range(len(dict[k])): #for each file in label
                x=target_dict[adword]
                seg_rel=copy.deepcopy(dict[k][i])
                filename=list(seg_rel.keys())[0]
                defdict=seg_rel[filename]
                segments=defdict[x]



                for count,seg in enumerate(segments):
                    seg.append(filename)
                    seg.append(count)                
                    scoreforfiles.append(seg)
        sort=sorted(scoreforfiles, key=lambda x: x[0], reverse=True)
        Final=sort[0:50]
        # print(Final)
        
        for items in Final:
            index=materialID.index(items[3])
            try:
                items.append(dad[index][items[4]][items[1]])
            except Exception as e:
                print(e)
                print(index,items[4],items[1])
                # print(dad[index][items[4]][items[1]])
                print(newnewmetalist[index][items[4]][items[1]])
                # items.append(dad[index][items[4]][items[1]])
                continue
        # print(Final)
        ff=[]
        for f in Final:
            temp=str(f[0])
            temp=temp.replace('tensor(', "")
            temp=temp.replace(')', "")
            adwords.append(adword)
            texts.append(f[5])
            scores.append(temp)
            
            
    
    # d[k].extend(ff)
    d={}
    d['Adword']=adwords
    d['Text']=texts
    d['Score']=scores
    return d

dict_top20=topksentences(Relevance_dict)

df11=pd.DataFrame.from_dict(dict_top20)

df11.to_csv("Top20Sentences_ALLTIERS.csv")
    
        

                
                


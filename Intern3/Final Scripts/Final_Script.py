# -*- coding: utf-8 -*-

import sys

# sys.path.append('/Users/haskar140/opt/anaconda3/lib/python39.zip')
# sys.path.append('/Users/haskar140/opt/anaconda3/lib/python3.9')
# sys.path.append('/Users/haskar140/opt/anaconda3/lib/python3.9/lib-dynload')
# sys.path.append('/Users/haskar140/opt/anaconda3/lib/python3.9/site-packages')
# sys.path.append('/Users/haskar140/opt/anaconda3/lib/python3.9/site-packages/aeosa')



import argparse
import json
import pandas as pd
import spacy
import urllib.request as requests
from tqdm.auto import tqdm
from nltk import tokenize
from collections import defaultdict
import re
import transformers
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk import tokenize






def openUrl(url):
    return requests.urlopen(url)

def new_process_text(text):
    text=text.lower()
    splitter=re.compile(r'\.\s?')
    reslist=splitter.split(text)
    rez = []
    for res in reslist:
        res=res.replace('\\n', ' ')
        res=res.replace(">","")
        res=res.replace("♪♪","♪")
        res=res.replace("♪","tune.")
        res=res.replace("(","")
        res=res.replace(")","")
        res=res.replace("]","")
        res=res.replace("[","")
        res=res.replace(",","")
        res=res.replace("?","")
        res=res.replace("!","")
        res=res.replace("/","")
        res=res.replace("\'","")
        res=res.replace("\"","")
        res=res.replace("\\","")
        rez.append(res)
    final=[]
    sent=[]
    length=0
    for item in rez:
        length=length+len(item)
        sent.append(item)
        if length>100:
            length=0
            temp=' '.join(sent)
            final.append(temp)
            sent=[]
    return final

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



def getCC(filename, link): #get the closed captions and preprocess them into sentences
    CC=[]
    seg1=defaultdict(list)
    cc1=[]
    seg=[]

    link=link

    try:
        asset_metadata = json.loads(openUrl('{}{}/ClosedCaptions?limit=100000'.format(link,filename)).read())
        contentseg = json.loads(openUrl('{}{}/contentSegmentation'.format(link,filename)).read())


        for i in range(len(contentseg['result'])):
            if contentseg['result'][i]['segmentType'] == 'segment':
                dic1={'startTime':contentseg['result'][i]['startTime'],
                'endTime':contentseg['result'][i]['endTime'], 'ClosedCaption': ''}
                seg.append(dic1)

            if contentseg['result'][i]['segmentType'] == 'som_eom':
                global_start=contentseg['result'][i]['startTime']

        print(len(seg))

        for i in range(len(asset_metadata['result'])):
            temp=asset_metadata['result'][i]['text']
            temp=temp.replace('-','')
            cc1.append(temp)

            startcaption=asset_metadata['result'][i]['startTime']+global_start
            endcaption=asset_metadata['result'][i]['endTime']+global_start

            for j in range(len(seg)):

                if startcaption >= seg[j]['startTime'] and endcaption <= seg[j]['endTime']:
                    temp=asset_metadata['result'][i]['text']
                    temp=temp.replace('-','')
                    # print(seg[j])
                    seg[j]['ClosedCaption']=seg[j]['ClosedCaption'] + ' ' + temp



        string=' '.join([str(item) for item in cc1])
        CC.append(string)
        seg1[filename]=seg

    except Exception as e:
        print(e)
        CC.append('-')

        #seg[j]['ClosedCaption'].append('-')
    return seg1




def getAdwords(Adwords_Path, Tier): #get the adwords with the childadwords appended

    df_fullname=pd.read_excel(Tier)
    df_fullname = df_fullname.fillna('')
    df_fullname['New_Name']=df_fullname['Name']+ " - "+ df_fullname['Tier 1']+ " "+ df_fullname['Tier 2']+ " "+ df_fullname['Tier 3']+ " "+df_fullname['Tier 4']

    df=pd.read_csv(Adwords_Path)
    df_unique=df[['Name','Unique ID ']]
    df_AllTiers=df[df['Important to NBCU']=='Y']
    df_AllTiers=df_AllTiers['Name']
    dftier1=df[df['Tier (1-4)']==1]
    dftier1=dftier1['Name']
    df=df[['Name', 'Tier 1 Parent']]

    def combiner(dftier1, df ):
        og=dftier1.to_list()
        ogName=df['Name'].to_list()
        Tier1Parent=df['Tier 1 Parent'].to_list()
        final=defaultdict(list)
        for i in range(len(ogName)):
            for j in range(len(og)):
                if og[j]==Tier1Parent[i]:
                    final[og[j]].append(ogName[i])     

        for k,v in final.items():
            combined=" ".join(v)
            combined=combined.replace(k + " ", " ")
            final[k]=combined
            
        return final

    dic=combiner(dftier1, df)
    dffinal=pd.DataFrame()

    
    listalltiers=df_AllTiers.to_list()
    childadwords=[]
    uniqueID=[]
    for item in listalltiers:
        temp=[]
        for k,v in dic.items():
            if k==item:
                temp.append(v)
        if len(temp)>0:
            childadwords.append(temp)
        if len(temp)==0:
            childadwords.append(" ")

        new_temp=df_unique.loc[df['Name']==item]
        uniqueID.append(new_temp['Unique ID '].values[0])
                
    for count, val in enumerate(childadwords):
        combined=" ".join(val)
        childadwords[count]=combined



    for count, val in enumerate(listalltiers):
        newnamedf=df_fullname.loc[df_fullname['Name']==val]
        newnamedf = newnamedf.fillna('')
        try:
            name=newnamedf['New_Name'].values[0]
            name=name.strip()
            listalltiers[count]=name
        except:
            print(val)
            print(newnamedf['New_Name'])


    dffinal['Name']=listalltiers
    dffinal['info']=childadwords 
    dffinal['UniqueID']=uniqueID



    return dffinal






def getRelevance(filename, adwordpath, tier, link):
    model = SentenceTransformer('all-mpnet-base-v2')
    captions=getCC(filename, link)
    adworddf=getAdwords(adwordpath, tier)

    # print(captions['FILE_MAF_20210727T231032Z_GMO_00000000001650_01'][0]['ClosedCaption'])
    adworddf['info']=adworddf['info'].apply(clean)
    targetlist=adworddf['Name'].to_list()
    targetlistchild=adworddf['Name']+adworddf['info']
    uniqueIDlist=adworddf['UniqueID'].to_list()

    CCsegments=[]
    for i in range(len(captions[filename])):
        cc=captions[filename][i]['ClosedCaption']
        cc=new_process_text(cc)
        CCsegments.append(cc)
    
    
    result=[]
    
    for l in tqdm(range(len(CCsegments))):
        if len(CCsegments[l])==0: 
            continue
        emb1=model.encode(CCsegments[l]) #, convert_to_tensor=True)
        emb2=model.encode(targetlistchild) #, convert_to_tensor=True)
        cos_sim=util.cos_sim(emb1,emb2)


        keywordd=[]
        for k in range(cos_sim.shape[1]):
            all_sentence_combinations=[]
            for j in range(cos_sim.shape[0]):
                all_sentence_combinations.append([cos_sim[j][k], j, k])
            all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)
            all_sentence_combinations = [ele for ele in all_sentence_combinations if ele[0] > 0.1]

            
            for ele in all_sentence_combinations:
                uniqueID=uniqueIDlist[ele[2]]
                keyword=targetlist[ele[2]]
                relevance=ele[0].numpy().tolist()
                relevance_clues=CCsegments[l][ele[1]]
                d={}
                d['uniqueID']=uniqueID
                d['keyword']=keyword
                d['relevance']=relevance
                d['relevance_clues']=relevance_clues
                keywordd.append(d)

        keywordd = sorted(keywordd, key=lambda d: d['relevance'], reverse=True) 
        segres={'offsetStartTime':captions[filename][l]['startTime'], 'offsetEndTime':captions[filename][l]['endTime'], "keywords": keywordd}
        result.append(segres)
    
    final = json.dumps(result)
    return final
    
    


    




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--MAF_ID",
                    help="MAF ID of the file to generate the result for")
    parser.add_argument("-p", "--PathAdwords", 
                    help="Path of Output Adwords File")
    parser.add_argument("-t", "--PathTiers", 
                    help="Path of IAB Tiered File")
    parser.add_argument("-l", "--Link", 
                    help="Link to call Comcast Endpoints")
    args=parser.parse_args()

    file_name=args.MAF_ID
    Adwords_Path=args.PathAdwords
    tiers=args.PathTiers
    link=args.Link

    #link='http://ac3448e420fce11eaaa4b0a458c10dab-684955813.us-east-1.elb.amazonaws.com/MafData/rs/db/mafdb/analysis/'
    
    finalJSON=getRelevance(file_name,Adwords_Path,tiers,link)

    print(finalJSON)

    with open("SCRIPT_RESULT.json", "w") as outfile:
        outfile.write(finalJSON)
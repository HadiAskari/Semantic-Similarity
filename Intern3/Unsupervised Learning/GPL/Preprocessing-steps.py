import pickle
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import CrossEncoder, InputExample, losses, SentenceTransformer, util
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
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


df=pd.read_csv('gmo_to_uniqueId.csv')
materialID=df['lastRunJob'].to_list()

with open('PreProcessed_segmented.pkl', 'rb') as f:
    newmetalist=pickle.load(f)

sampled_datadf=pd.read_csv("Sampled_Adword_Labeled.csv")

#For Normal process with target labels only

df1=pd.read_csv('ChildAdwordTargetDatabase.csv')
target_dict=dict(zip(df1['Name'], df1.index))

df2=pd.read_csv("InitialDocDatabase_no_segment.csv")

device = "cuda" if torch.cuda.is_available() else "cpu"

listofepisodes=sampled_datadf['lastRunJob'].to_list()

def clean(text):
    text=text.lower()
    text=text.replace("(", "")
    text=text.replace("[", "")
    text=text.replace("]", "")
    text=text.replace("\\n", " ")
    text=text.replace(")", "")
    text=text.replace(";", "")
    text=text.replace(",", "")
    text=text.replace("+", "")
    text=text.replace(".", "")
    text=text.replace("&", "")
    text=text.replace("!", "")
    text=text.replace(">", "")
    text=text.replace("♪", "")
    text=text.replace("/", "")
    text=text.replace("\\", "")
    text=text.replace("\n", "")
    return text

df2['AssetCC']=df2['AssetCC'].apply(clean)

passages=df2['AssetCC'].to_list()
# for ep in tqdm(listofepisodes):
#     index=materialID.index(ep)
#     for j in range(len(newmetalist[index])):
#         temp=" ".join(newmetalist[index][j])
#         passages.append(temp)

print("Len of passages is {}".format(len(passages)))

def generate_questions(
    tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM, passages,
    batch_size: int=32, device: str='cuda', n_ques_per_passage: int = 3
):

    outputs = []
    n_batches = len(passages) // batch_size + int(len(passages) % batch_size != 0)

    for n in tqdm(range(n_batches)):
        passages_batch = passages[n*batch_size: (n+1)*batch_size]
        inputs = tokenizer(passages_batch, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
        output = model.generate(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            max_length=64,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=n_ques_per_passage
        )
        outputs += output

    questions = [tokenizer.decode(txt, skip_special_tokens=True) for txt in outputs]
    return questions

model_name = "doc2query/msmarco-t5-base-v1"

doc2q_tokenizer = AutoTokenizer.from_pretrained(model_name)
doc2q_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

n_ques_per_passage = 3
questions = generate_questions(
    doc2q_tokenizer, doc2q_model, passages, n_ques_per_passage=n_ques_per_passage
)

q2pix = {}  # query to passage index mapping

pix = 0
for i in range(len(questions)):
    if i > 0 and i % n_ques_per_passage == 0:
        pix += 1
    q2pix[questions[i]] = pix

n = 0

for question, passage_ix in q2pix.items():
    print(passages[passage_ix])
    print(f"[Q] {question} ?\n---\n")
    n += 1
    if n == (n_ques_per_passage * 3):
        break

unique_questions = list(set(questions))
len(unique_questions)

torch.cuda.empty_cache()

retriever = SentenceTransformer("msmarco-distilbert-base-tas-b", device="cuda")

passage_embeddings = retriever.encode(passages)

def get_topk_passages(
    model: SentenceTransformer, query: str, passage_embeds: torch.tensor, n: int=10
):
    query_embed = model.encode(query)
    sim_scores = util.cos_sim(query_embed, passage_embeds).numpy()
    top_passages_ix = np.argsort(sim_scores)[0][::-1][:n]
    return top_passages_ix, [sim_scores[0][ix] for ix in top_passages_ix]

query_passage_pairs = []

for query in tqdm(unique_questions):
    top_passages_ixs, _ = get_topk_passages(retriever, query, passage_embeddings, 3)
    query_passage_pairs += [(query, passages[ix]) for ix in top_passages_ixs if ix != q2pix[query]]

for query, negative_passage in random.sample(query_passage_pairs, 3):
    print(f"[Q] {query}\n\n[NEGATIVE] {negative_passage}\n\n[POSITIVE] {passages[q2pix[query]]}\n---\n")

ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device="cuda")

query_passage_margins = []

for query, passage in tqdm(query_passage_pairs):
    p_score = ce.predict((query, passages[q2pix[query]]))
    n_score = ce.predict((query, passage))
    margin = p_score - n_score
    query_passage_margins.append((query, passages[q2pix[query]], passage, margin))

with open("query_passage_margins.pkl", "wb") as f:
    pickle.dump(query_passage_margins, f)


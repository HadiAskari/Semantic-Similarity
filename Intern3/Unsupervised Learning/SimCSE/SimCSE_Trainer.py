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
import os
import random
import numpy as np
import nltk
import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import torch
from torch.utils.data import DataLoader


nltk.download('punkt')
#nlp = spacy.load("en_core_web_lg")
# model = SentenceTransformer('all-mpnet-base-v2')

df=pd.read_csv('gmo_to_uniqueId.csv')
materialID=df['lastRunJob'].to_list()

with open('New_PreProcessed_segmented.pkl', 'rb') as f:
    newmetalist=pickle.load(f)

sampled_datadf=pd.read_csv("Sampled_Adword_Labeled.csv")

#For Normal process with target labels only

df1=pd.read_csv('ChildAdwordTargetDatabase.csv')
target_dict=dict(zip(df1['Name'], df1.index))

seed = 10

random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


model_name = 'sentence-transformers/all-mpnet-base-v2'
word_embedding_model = models.Transformer(model_name, max_seq_length=32)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

train_sentences=[]
for i in range(len(newmetalist)):
    for j in range(len(newmetalist[i])):
        for k in range(len(newmetalist[i][j])):
            train_sentences.append(newmetalist[i][j][k])

train_sentences=train_sentences[:1000000]

# Convert train sentences to sentence pairs
train_data = [InputExample(texts=[s, s]) for s in train_sentences]

# DataLoader to batch your data
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)

# Use the denoising auto-encoder loss
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    weight_decay=0,
    scheduler="constantlr",
    optimizer_params={'lr': 3e-5},
    show_progress_bar=True
)

pathtosave= os.getcwd() + '/models/SimCSE/'

model.save(path=pathtosave, model_name='1st')
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

train_sentences=[]
for i in range(len(newmetalist)):
    for j in range(len(newmetalist[i])):
        for k in range(len(newmetalist[i][j])):
            train_sentences.append(newmetalist[i][j][k])

train_sentences=train_sentences[:1000000]

train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
model_name = "sentence-transformers/all-distilroberta-v1"

embedding_model = models.Transformer(model_name)
pooling = models.Pooling(embedding_model.get_word_embedding_dimension(), "cls")
model = SentenceTransformer(modules=[embedding_model, pooling], device="cuda")

train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    weight_decay=0,
    scheduler="constantlr",
    optimizer_params={'lr': 3e-5},
    show_progress_bar=True
)

pathtosave= os.getcwd() + '/models/TSDAE/'

model.save(path=pathtosave, model_name='1st')
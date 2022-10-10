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


with open("query_passage_margins.pkl", "rb") as f:
    query_passage_margins = pickle.load(f)

training_data = []
for query, passage_pos, passage_neg, margin in query_passage_margins:
    training_data.append(InputExample(texts=[query, passage_pos, passage_neg], label=float(margin)))

loader = torch.utils.data.DataLoader(training_data, batch_size=8, shuffle=True)

torch.cuda.empty_cache()

bi_encoder = SentenceTransformer("msmarco-distilbert-base-tas-b", device="cuda")
bi_encoder.max_seq_length = 512

loss = losses.MarginMSELoss(bi_encoder)

bi_encoder.fit(
    train_objectives=[(loader, loss)],
    epochs=25,
    warmup_steps=int(len(loader)*0.1),
    show_progress_bar=True
)

pathtosave=os.getcwd()

pathtosave=pathtosave + "/models"



bi_encoder.save(path=pathtosave, model_name='1st')
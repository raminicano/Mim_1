
import pandas as pd
# from tokenization import KoBertTokenizer ##### 토크나이저 모데을 이용해보자. ###//
import os 
from flask import Flask, request, jsonify, render_template
import torch
import csv
# from model import CustomModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import DistilBertModel
from collections import defaultdict
import sentencepiece as spm
from flask_cors import CORS

import torch
import torch.nn as nn
#from CD import ClassifierDataset, FeatureExtractDataset
#from model import CustomModel

# from tokenization import KoBertTokenizer
import infer
from torch.utils.data import Dataset, DataLoader
from minone import tokenization, CD, model
from ast import literal_eval




from tqdm import tqdm



# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():  
    device = torch.device("mps")
else:
    device = torch.device("cpu")

models = DistilBertModel.from_pretrained('monologg/distilkobert')
tokenizers = tokenization.KoBertTokenizer.from_pretrained('monologg/kobert')

model_path= "../../../dataset/code/minone/model.pth"

models = models.to(device) 
c_model = model.CustomModel(models, 768, 35)

parameter = torch.load(model_path, map_location=torch.device('mps')) 
model_keys = set(c_model.state_dict().keys())
pretrained_keys = set(parameter.keys())
unexpected_keys = pretrained_keys - model_keys

trimmed_state_dict = {k: v for k, v in parameter.items() if k not in unexpected_keys}

c_model.load_state_dict(trimmed_state_dict, strict=False)   


def cosine_sim(a, b):
    return cosine_similarity([a], [b])[0][0]

def get_gwa_value(gwa_df, region_text, pred, test_embedding):
    match = gwa_df[
        (gwa_df['구'] == region_text) & 
        (gwa_df['topic'] == pred[0])
    ]

    # return match['과'].tolist()
    
    if len(match) == 0:
        return None
    elif len(match) == 1: # 1개가 매칭되면 한개만 반환하면 됨
        return match['과'].tolist()[0]
    else:
        # 여러 "과" 중 가장 유사한 "과" 선택
        best_match = None
        max_similarity = -1

        for _, row in match.iterrows(): # match는 과 데이터프레임
            dept_embedding = row['embedding']  # 각 "과"의 임베딩을 가정
            sim = cosine_sim(test_embedding[0], dept_embedding)
            if sim > max_similarity:
                max_similarity = sim
                best_match = row['과'].values[0]

        return best_match

import re


app = Flask(__name__)
CORS(app)


def inference(model, test_loader, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        for sentence, attention_mask in tqdm(test_loader): 
            sentence = sentence.to(device)
            mask = attention_mask.to(device)

            cls_output, pred = model(sentence, mask)

            pred = pred.argmax(1).detach().cpu().numpy().tolist()

    return cls_output, pred



@app.route('/api/analyze', methods=['GET', 'POST'])
def analyze():
    try:
        data = request.get_json()
        
        complaint_text = data['complaint'] ## 원본 
        region_text= data['region']
        title_text= data['title']


        minone_text = title_text + " " + complaint_text

        # 과 데이터프레임
        location = '../../../dataset/code/minone/gwa_embedding.csv'
        
        reader = pd.read_csv(location)
        reader['embedding'] = reader['embedding'].apply(lambda x: list(map(float, literal_eval(x))))
        

        # 임베딩 값 추출
        minone_dataset = CD.FeatureExtractDataset(minone_text, tokenizers)
        minone_dataloader = DataLoader(minone_dataset, batch_size=1, shuffle=False, num_workers=0)
        cls_output, pred = inference(c_model, minone_dataloader, device) 
        
        # 리스트 반환 함수 호출
        gwa = get_gwa_value(reader, region_text, pred, cls_output)
        state_dict = c_model.state_dict()
        layer_weights = state_dict['classifier.0.weight']
        print("Classifier layer weights:", layer_weights)
        layer_bias = state_dict['classifier.0.bias']
        print("Classifier layer bias:", layer_bias)


    
        # 유사도 함수 구하기
        minone_text = ' '.join(gwa)


        return jsonify(minone_text), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)





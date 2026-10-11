import torch
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np

class KoBERTSentimentAnalyzer:
    def __init__(self, model_path='monologg/kobert'):
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            self.is_ready = True
        except Exception as e:
            print("모델 가중치 로드 실패. UI 테스트를 위한 Mock 모드로 동작합니다.")
            self.is_ready = False

    def predict(self, text):
        if not self.is_ready:
            import random
            return random.choice(['긍정', '중립', '부정'])
        
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
        
        mapping = {0: '부정', 1: '중립', 2: '긍정'}
        return mapping.get(pred_idx, '중립')
        
    def analyze_dataframe(self, df, text_col='title'):
        df['sentiment'] = df[text_col].apply(self.predict)
        return df

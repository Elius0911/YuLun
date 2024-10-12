import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import warnings

# 隱藏特定的警告
warnings.filterwarnings("ignore", category=UserWarning)

class BertForSentimentRegression(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)  # 輸出層設置為1個回歸輸出

    def forward(self, **kwargs):
        outputs = self.bert(**kwargs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # 獲取 [CLS] token 的表示
        regression_output = self.regressor(cls_embedding)  # 通過回歸層
        return regression_output

def analyze_sentiment(text):
    model_name = "hfl/chinese-bert-wwm-ext"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSentimentRegression(model_name)

    # CUDA 偵測
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # 使用 torch.no_grad() 來禁用梯度計算
    with torch.no_grad():
        print("[INFO] Sentiment analyzing...")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # 獲取分數
        print("[INFO] Scoring...")
        score = model(**inputs)
        score = torch.sigmoid(score).item()  # 轉換為 0-1 之間的值

    return score  # 返回情緒分析分數

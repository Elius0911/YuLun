import whisper
import torch
import opencc  # 簡轉繁
import warnings  # 隱藏警告語

# 隱藏原模型之警告語
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 可用模型
model_list = ["small", "medium", "large"]
def GetModelList():
    return model_list

def transcribe_audio(audio_file_path, model_choice):
    try:
        # 將使用者輸入的選項轉換為整數
        model_choice = int(model_choice)
        
        # 如果輸入的數字超出模型範圍，使用預設模型 small
        if model_choice < 1 or model_choice > len(model_list):
            print("[ERROR] Invalid choice. Defaulting to small model...")
            model_name = "small"
        else:
            # 根據使用者選擇載入對應的模型
            model_name = model_list[model_choice - 1]
    
    except ValueError:
        # 如果使用者輸入不是數字，使用預設模型 small
        print("[ERROR] Invalid input. Defaulting to small model...")
        model_name = "small"

    # 加載模型
    print(f"[INFO] Loading {model_name} model...")
    model = whisper.load_model(model_name).to("cuda") if torch.cuda.is_available() else whisper.load_model(model_name)

    # 處理音訊文件並指定語言為中文
    print("[INFO] Transcribing...")
    result = model.transcribe(audio_file_path, language="zh")

    # 簡體轉繁體
    print("[INFO] Converting to zh-TW...")
    converter = opencc.OpenCC('s2t')  # s2t: 簡轉繁
    result_tra = converter.convert(result["text"])

    return result_tra  # 返回轉錄結果

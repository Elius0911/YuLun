import whisper
import torch
import opencc  # 簡轉繁
import warnings  # 隱藏警告語

# 隱藏原模型之警告語
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def transcribe_audio(audio_file_path, model_choice):
    # 讓使用者選擇模型
    if model_choice == '1':
        model_name = "base"
    elif model_choice == '2':
        model_name = "medium"
    elif model_choice == '3':
        model_name = "large"
    else:
        print("[ERROR] Invalid choice. Defaulting to BASE model...")
        model_name = "base"

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

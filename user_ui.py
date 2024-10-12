import stt_whisper
import sa_bert
import torch

def main():
    # 設定音檔路徑
    audio_file_path = r"C:\Users\Elius\Downloads\這件商品很適合你呢cut.wav"
    
    # CUDA 偵測
    if torch.cuda.is_available():
        print("[INFO] CUDA detected. Use CUDA mode.")
    else:
        print("[INFO] CUDA not detected. Use CPU mode.")
    
    # 讓使用者選擇模型
    model_choice = input("[INFO] Please select a model: (1) Base / (2) Medium / (3) Large\r\n")
    
    # 呼叫 STT 功能
    transcribed_text = stt_whisper.transcribe_audio(audio_file_path, model_choice)
    print(f"[Result] Transcribed Text: {transcribed_text}")

    # 呼叫情緒分析功能
    sentiment_score = sa_bert.analyze_sentiment(transcribed_text)
    print(f"[Result] Sentiment Score: {sentiment_score}")

if __name__ == "__main__":
    main()

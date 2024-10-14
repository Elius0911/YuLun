import stt_whisper
import sa_bert
import torch

def main():
    # 設定音檔路徑
    audio_file_path = r"C:\Users\Elius\Downloads\TestAudio\這件商品很適合你呢cut.wav"
    
    # CUDA 偵測
    if torch.cuda.is_available():
        print("[INFO] CUDA detected. Use CUDA mode.")
    else:
        print("[INFO] CUDA not detected. Use CPU mode.")
    
    # 獲取可用模型
    model_list = stt_whisper.GetModelList()
    model_list_amount = len(model_list)

    # 讓使用者選擇模型
    print(f"[INFO] Model List: ", end=" ")
    for num in range(0, model_list_amount):
        print(f"({num+1}) {model_list[num]} ", end=" ")
    model_choice = input(f"\r\n[INFO] Please select a model: ")
    
    # 呼叫 STT 功能
    transcribed_text = stt_whisper.transcribe_audio(audio_file_path, model_choice)
    print(f"[Result] Transcribed Text: {transcribed_text}")

    # 呼叫情緒分析功能
    sentiment_score = sa_bert.analyze_sentiment(transcribed_text)
    print(f"[Result] Sentiment Score: {sentiment_score}")

if __name__ == "__main__":
    main()

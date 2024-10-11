from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import torch

# Configuration settings
model_name = "ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt"
audio_file_path = r"C:\Users\Elius\Downloads\德2.wav"

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{'CUDA detected. Running in CUDA mode.' if torch.cuda.is_available() else 'CUDA not detected. Running in CPU mode.'}]")

# Load the processor and model
print("[Loading processor and model...]")
try:
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    print("[Processor and model loaded successfully.]")
except Exception as e:
    print(f"[Error loading processor or model: {e}]")
    exit()  # Exit if model loading fails

# Try loading the audio file
try:
    print("[Loading audio file...]")
    audio_input, sample_rate = librosa.load(audio_file_path, sr=16000, mono=True)
    print("[Audio file loaded successfully.]")
except Exception as e:
    print(f"[Error loading audio file: {e}]")
    exit()  # Exit if audio loading fails

# Convert audio to model input format
print("[Processing audio for model input...]")
input_values = processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values.to(device)
print(f"[Input shape: {input_values.shape}]")
print(f"[Resampled audio length: {len(audio_input)} samples, Sample rate: {sample_rate} Hz]")

# Perform inference
print("[Running inference...]")
with torch.no_grad():
    logits = model(input_values).logits

# Get the predicted IDs and decode the predictions
print("[Extracting predicted IDs...]")
predicted_ids = torch.argmax(logits, dim=-1)
print("[Decoding the predicted transcription...]")
transcription = processor.batch_decode(predicted_ids)

# Output the transcription
print("Transcription result:")
print(transcription)

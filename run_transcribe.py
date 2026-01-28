import torch
from transformers import WhisperProcessor, AutoModelForSpeechSeq2Seq
from peft import PeftModel
import librosa

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Device: {device}, dtype: {torch_dtype}")

    model_id = "openai/whisper-large-v3"
    adapter_id = "AbdelrahmanHassan/whisper-large-v3-egyptian-arabic"

    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(adapter_id)

    print("Loading base model (may download weights)...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device)

    print("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_id)

    audio_path = "my_voice.wav"
    print(f"Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device).to(torch_dtype)

    print("Transcribing...")
    with torch.no_grad():
        predicted_ids = model.generate(inputs, max_length=225)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print("\n--- Transcription ---")
    print(transcription)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
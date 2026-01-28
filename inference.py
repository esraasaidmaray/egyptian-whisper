import torch
from transformers import WhisperProcessor, AutoModelForSpeechSeq2Seq
from peft import PeftModel
import librosa
if __name__ == "__main__":
    print("--- Start Process ---") # ضيفي السطر ده للتأكد إن الكود بدأ
    test_file = "test.wav" 
    
    # تأكدي إن الجهاز شايف الـ GPU
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    text = transcribe_audio(test_file)
    print("\nTranscription Result:")
    print(text)
# اختيار الجهاز (بما إن عندك GPU فالسرعة هتكون ممتازة)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading Whisper Large V3 with Egyptian LoRA on {device}...")

# 1. تحميل الـ Processor والموديل الأساسي
model_id = "openai/whisper-large-v3"
adapter_id = "AbdelrahmanHassan/whisper-large-v3-egyptian-arabic"

processor = WhisperProcessor.from_pretrained(adapter_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)

# 2. دمج طبقة اللهجة المصرية (LoRA Adapter)
model = PeftModel.from_pretrained(model, adapter_id)

# 3. وظيفة تحويل الصوت لنص
def transcribe_audio(file_path):
    # تحميل الصوت بـ Sampling Rate 16000 كما يتطلب الموديل
    audio, sr = librosa.load(file_path, sr=16000)
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device).to(torch_dtype)

    print("Transcribing Egyptian Arabic...")
    with torch.no_grad():
        # الموديل متدرب بـ max_length 225
        predicted_ids = model.generate(input_features, max_length=225)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

if __name__ == "__main__":
    # حطي هنا اسم ملف الصوت اللي عايزة تجربيه
    test_file = "my_voice.wav" 
    try:
        text = transcribe_audio(test_file)
        print("\n--- النتيجة بالمصري ---")
        print(text)
        print("-" * 25)
    except Exception as e:
        print(f"Error: {e}. تأكدي إن ملف '{test_file}' موجود في نفس الفولدر.")
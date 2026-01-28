from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
import torch
import librosa
import tempfile
from transformers import WhisperProcessor, AutoModelForSpeechSeq2Seq
from peft import PeftModel

app = FastAPI(title="Egyptian Whisper Transcription")

# device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model IDs
MODEL_ID = "openai/whisper-large-v3"
ADAPTER_ID = "AbdelrahmanHassan/whisper-large-v3-egyptian-arabic"

@app.on_event("startup")
def load_model():
    global processor, model
    processor = WhisperProcessor.from_pretrained(ADAPTER_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    model = PeftModel.from_pretrained(model, ADAPTER_ID)

@app.get("/", response_class=HTMLResponse)
def homepage():
    return """
    <html>
      <body>
        <h2>Upload WAV to transcribe</h2>
        <form action="/transcribe" enctype="multipart/form-data" method="post">
          <input name="file" type="file" accept="audio/*">
          <input type="submit">
        </form>
      </body>
    </html>
    """

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # save temp file
    suffix = ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        audio, sr = librosa.load(tmp.name, sr=16000)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device).to(dtype)
    with torch.no_grad():
        predicted_ids = model.generate(inputs, max_length=225)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return {"transcription": transcription}

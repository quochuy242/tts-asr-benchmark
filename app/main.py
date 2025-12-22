import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks

from fastapi.responses import FileResponse

from app.asr import ASRModel
from app.schemas import ASRResponse, TTSRequest
from app.tts import TTSModel

app = FastAPI(title="Sherpa-ONNX ASR + TTS API")

# ===== Load models ONCE =====
asr_model = ASRModel(use_int8=False)
tts_model = TTSModel()


# ===== ASR API =====
@app.post("/asr")
async def asr_api(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
        f.write(await file.read())

    try:
        text = asr_model.transcribe_from_file(wav_path)
    finally:
        os.remove(wav_path)

    text = text or ""
    return ASRResponse(text=text, is_empty=(text == ""))


# ===== TTS API =====
@app.post("/tts")
async def tts_api(req: TTSRequest, background_tasks: BackgroundTasks):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name

    try:
        tts_model.synthesize(
            text=req.text,
            sid=req.sid,
            speed=req.speed,
            save_path=out_path,
        )

        background_tasks.add_task(os.remove, out_path)

        return FileResponse(
            out_path,
            media_type="audio/wav",
            filename="tts.wav",
        )
    except Exception as e:
        if os.path.exists(out_path):
            os.remove(out_path)
        raise HTTPException(status_code=500, detail=str(e))

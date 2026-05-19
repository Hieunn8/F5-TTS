import asyncio
import io
import logging

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="F5-TTS API", description="TTS API backed by Triton + TRT-LLM")

TRITON_URL = "http://localhost:8000/v2/models/f5_tts/infer"
SAMPLE_RATE = 24000
_http_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup():
    global _http_client
    _http_client = httpx.AsyncClient(timeout=120.0)


@app.on_event("shutdown")
async def shutdown():
    if _http_client:
        await _http_client.aclose()


def _resample(waveform: np.ndarray, src_sr: int, dst_sr: int = SAMPLE_RATE) -> np.ndarray:
    if src_sr == dst_sr:
        return waveform
    from scipy.signal import resample as scipy_resample
    return scipy_resample(waveform, int(len(waveform) * dst_sr / src_sr)).astype(np.float32)


def _change_speed(waveform: np.ndarray, speed: float) -> np.ndarray:
    if speed == 1.0:
        return waveform
    import librosa
    # time_stretch rate > 1 → faster, rate < 1 → slower
    return librosa.effects.time_stretch(waveform, rate=speed).astype(np.float32)


async def _call_triton(waveform: np.ndarray, reference_text: str, target_text: str) -> np.ndarray:
    lengths = np.array([[len(waveform)]], dtype=np.int32)
    wav_2d = waveform.reshape(1, -1).astype(np.float32)

    payload = {
        "inputs": [
            {"name": "reference_wav",     "shape": list(wav_2d.shape),  "datatype": "FP32",  "data": wav_2d.tolist()},
            {"name": "reference_wav_len", "shape": list(lengths.shape), "datatype": "INT32", "data": lengths.tolist()},
            {"name": "reference_text",    "shape": [1, 1],              "datatype": "BYTES", "data": [reference_text]},
            {"name": "target_text",       "shape": [1, 1],              "datatype": "BYTES", "data": [target_text]},
        ]
    }

    rsp = await _http_client.post(
        TRITON_URL,
        headers={"Content-Type": "application/json"},
        json=payload,
        params={"request_id": "0"},
    )

    if rsp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Triton error {rsp.status_code}: {rsp.text}")

    result = rsp.json()
    audio = np.array(result["outputs"][0]["data"], dtype=np.float32)
    return audio


@app.post("/tts", summary="Text-to-Speech")
async def tts(
    ref_audio: UploadFile = File(..., description="Reference audio file (wav/mp3/flac)"),
    ref_text: str = Form(..., description="Transcript of the reference audio"),
    target_text: str = Form(..., description="Text to synthesize"),
    speed: float = Form(1.0, ge=0.5, le=2.0, description="Speed multiplier (0.5–2.0)"),
):
    audio_bytes = await ref_audio.read()
    try:
        waveform, src_sr = sf.read(io.BytesIO(audio_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read audio file: {e}")

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    waveform = _resample(waveform.astype(np.float32), src_sr)

    logger.info("Calling Triton: ref_text=%r target_text=%r speed=%.2f", ref_text, target_text, speed)
    output_audio = await _call_triton(waveform, ref_text, target_text)

    if speed != 1.0:
        output_audio = await asyncio.to_thread(_change_speed, output_audio, speed)

    buf = io.BytesIO()
    sf.write(buf, output_audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


@app.get("/health")
def health():
    return {"status": "ok"}

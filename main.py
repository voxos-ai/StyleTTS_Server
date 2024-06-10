from fastapi import FastAPI
import uvicorn
from TTS import TTS
from TTS.logger import configure_logger
import torch
from contextlib import asynccontextmanager
from ResponseRequestModels import end_user_request
from glob import glob
import time
import io
from scipy.io.wavfile import write
from base64 import b64encode

tts:TTS = None
logger = configure_logger(__name__)
RATE = 24000
PORT = 8700

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    tts = TTS()
    logger.info(f"Style TTS is __init__")
    if torch.cuda.is_available():
        logger.info(f"model loaded in CUDA")
        logger.info(f"memory allocated {torch.cuda.memory_allocated()}")
    else:
        logger.info(f"model loaded in CPU")
    for path in glob("./voices/*.wav"):
        tts.register_voice(path)
    yield
    del tts
    if torch.cuda.is_available():
        torch.cuda.caching_allocator_delete()
app = FastAPI(lifespan=lifespan)

@app.get("/heartbeat")
def heartbeat():
    return {"alive"}

@app.post("/tts")
def audio_to_text(response:end_user_request):
    prev = time.time()
    logger.info(f"start processing text of size {len(response.text)} with following config")
    logger.info(f"voice_id: {response.voice_id}, alpha: {response.alpha}, beta: {response.beta}, diffusion step: {response.diffusion_steps}, embedding scale: {response.embedding_scale}")
    audio = tts.text_to_audio(response.text,response.voice_id,response.alpha,response.beta,response.diffusion_steps,response.embedding_scale)
    file = io.BytesIO()
    write(file,RATE,audio)
    file.seek(0)
    logger.info(f"audio is created in {time.time() - prev}")
    return {'audio': b64encode(file.read()).decode()}
    
if __name__ == "__main__":
    uvicorn.run("main:app",port=PORT,reload=True)
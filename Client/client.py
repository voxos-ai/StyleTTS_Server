import requests
import json
from base64 import b64decode
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

json_data = {
    'text': 'hello world i am R Ansh Joseph whats your name',
    'rate':8000,
    'voice_id': 'default',
    'alpha': 0.3,
    'beta': 0.7,
    'diffusion_steps': 5,
    'embedding_scale': 1,
}
import time
prev = time.time()
response = requests.post('http://127.0.0.1:8700/tts', headers=headers, json=json_data)
response = json.loads(response.text)
print(time.time())
with open("audio.wav",'wb') as file:
    file.write(b64decode(response['audio']))


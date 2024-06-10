from pydantic import BaseModel

class end_user_request(BaseModel):
    text:str
    rate:int=8000
    voice_id:str='default'
    alpha:float=0.3
    beta:float=0.7
    diffusion_steps:int=5
    embedding_scale:float=1
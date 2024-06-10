from pydantic import BaseModel

class end_user_request(BaseModel):
    text:str
    voice_id:str
    alpha:float=0.3
    beta:float=0.7
    diffusion_steps:int=5
    embedding_scale:float=1
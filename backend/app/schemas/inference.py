from pydantic import BaseModel
from typing import Optional

class InferenceRequest(BaseModel):
    image_base64: str
    prompt_text: Optional[str] = None
    max_new_tokens: int = 64
    num_beams: int = 4

class InferenceResponse(BaseModel):
    caption: str
    alpha: float
    beta: float
    gamma: float
    clip_similarity: Optional[float] = None
    inference_time_ms: float

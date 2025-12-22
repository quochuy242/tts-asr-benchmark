from pydantic import BaseModel


# ===== Schemas =====
class TTSRequest(BaseModel):
    text: str
    sid: int = 0
    speed: float = 1.0


class ASRResponse(BaseModel):
    text: str
    is_empty: bool = False

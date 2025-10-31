from pydantic import BaseModel
from typing import Optional

class ValidationRequest(BaseModel):
    code: str


class ValidationResponse(BaseModel):
    is_valid: bool
    code: Optional[str] = None
    errors: Optional[str] = None
    message: str

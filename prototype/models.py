from pydantic import BaseModel

class DebateRequest(BaseModel):
    topic: str
    speaker: str
    transcript: str
    reference_arguments: list = []

class WinnerRequest(BaseModel):
    teams: list

class DebateRequest2(BaseModel):
    topic: str
    teams: list
    reference_arguments: list = []
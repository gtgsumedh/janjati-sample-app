from pydantic import BaseModel

class InitRequest(BaseModel):
    method: str
    username: str
    reqdatetime: str

class QuestionRequest(BaseModel):
    method: str
    username: str
    uniqueid: str
    question: str
    reqdatetime: str

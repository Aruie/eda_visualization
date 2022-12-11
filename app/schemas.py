from pydantic import BaseModel, HttpUrl

# 2
class Recipe(BaseModel):
    id: int
    label: str
    source: str
    url: HttpUrl  # 3
from pydantic import BaseModel


class CountResponse(BaseModel):
    bag_count: int

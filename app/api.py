from fastapi import FastAPI

from app.schemas import CountResponse
from app.state import RuntimeState


def create_app(state: RuntimeState) -> FastAPI:
    app = FastAPI()

    @app.get("/count", response_model=CountResponse)
    async def get_count() -> CountResponse:
        with state.lock:
            return CountResponse(bag_count=state.current_count)

    return app

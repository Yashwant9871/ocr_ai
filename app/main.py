import threading

import uvicorn

from app.api import create_app
from app.config import API_HOST, API_PORT, RTSP_URL
from app.services import capture_frames, process_stream
from app.state import RuntimeState


def run() -> None:
    state = RuntimeState()
    app = create_app(state)

    threading.Thread(target=capture_frames, args=(RTSP_URL, state), daemon=True).start()
    threading.Thread(target=process_stream, args=(state,), daemon=True).start()

    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    run()

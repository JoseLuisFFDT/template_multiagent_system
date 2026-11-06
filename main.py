"""Entry point of the application: starts the HTTP API (FastAPI).

    python main.py                  # API on http://127.0.0.1:8000  (docs: /docs)
    streamlit run src/frontend/app.py   # chat UI, in another terminal

The terminal version of the demo lives in ``cli.py``.
"""
import os

import uvicorn

if __name__ == "__main__":
    # Passing the app as an import string keeps the door open for `reload=True` during development.
    uvicorn.run("src.api.app:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))

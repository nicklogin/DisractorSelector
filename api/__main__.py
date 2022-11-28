import os
import uvicorn

ROOT_PATH = os.getenv("DISSELECTOR_ROOT_PATH", default="/")

uvicorn.run(
    'api.app:app',
    host='0.0.0.0',
    port=5000,
    reload=False,
    root_path = ROOT_PATH
)

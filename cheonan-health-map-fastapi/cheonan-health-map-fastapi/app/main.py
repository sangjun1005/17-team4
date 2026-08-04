from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app = FastAPI(
    title="천안시 읍면동 의료취약지역 지도",
    description="기존 React 지도 화면과 기능을 그대로 제공하는 FastAPI 애플리케이션",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/api/health", tags=["system"])
def health_check() -> JSONResponse:
    return JSONResponse({"status": "ok"})

@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/{path:path}", include_in_schema=False)
def serve_spa(path: str):
    requested = (STATIC_DIR / path).resolve()
    if STATIC_DIR.resolve() in requested.parents and requested.is_file():
        return FileResponse(requested)
    return FileResponse(STATIC_DIR / "index.html")

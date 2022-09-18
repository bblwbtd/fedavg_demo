import uvicorn

from config import settings

if __name__ == "__main__":
    print(f"Docs url: http://127.0.0.1:{settings.port}/docs")
    uvicorn.run("src.api:app", port=settings.port)

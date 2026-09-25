from fastapi import FastAPI

from app.api.analyze import router

app = FastAPI(title="Gradient Nova AI")
app.include_router(router)


@app.get("/")
def home():
    return {"message": "Gradient Nova AI backend running"}

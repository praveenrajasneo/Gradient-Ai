from fastapi import FastAPI

app = FastAPI(title="Gradient Nova AI")


@app.get("/")
def home():
    return {"message": "Gradient Nova AI backend running"}

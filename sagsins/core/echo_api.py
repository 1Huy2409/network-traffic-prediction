from fastapi import FastAPI, Request
app = FastAPI()
@app.get("/")
def root():
    return {"status":"ok","service":"sagsins-echo"}
@app.post("/ingest")
async def ingest(req: Request):
    j = await req.json()
    return {"ok": True, "echo": j}

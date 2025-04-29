from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from advice import (
    load_models,
    make_prediction,
    get_curve_data,
    SYMBOLS,
    schedule_daily,
    daily_job,
    scheduler
)

app = FastAPI(title="Crypto Trading Advice Service")

@app.on_event("startup")
async def startup_event():
    schedule_daily()
    scheduler.start()
    await daily_job()

# preload
models = load_models(SYMBOLS)

templates = Jinja2Templates(directory="templates")

@app.get("/advice/{symbol}", response_model=dict)
def get_advice(symbol: str):
    if symbol not in models:
        raise HTTPException(status_code=404, detail="Symbol not loaded")
    return {symbol: make_prediction(symbol)}

@app.get("/advice", response_model=dict)
def get_all_advice():
    return {sym: make_prediction(sym) for sym in SYMBOLS}

@app.get("/curve/{symbol}", response_class=JSONResponse)
def curve(symbol: str, days: int = 30):
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=404, detail="Unknown symbol")
    return JSONResponse(content=get_curve_data(symbol, days))

@app.get("/", response_class=HTMLResponse)
def web_ui(request: Request):
    data = get_all_advice()
    return templates.TemplateResponse("index.html", {"request": request, "data": data})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
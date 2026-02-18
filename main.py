from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import requests
import math
import pandas as pd
import uuid
import os
import json
import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC_TICKET = os.getenv("MQTT_TOPIC_TICKET", "thalos/trading/order_ticket")

app = FastAPI()

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

from typing import Optional

class SignalRequest(BaseModel):
    request_id: Optional[str] = None
    symbol: str
    timeframe: str
    risk: float = 1.8
    ttl_sec: int = 900
    mode: str = "HEDGE"
    debug: bool = False

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def fetch_klines(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(BINANCE_KLINES, params=params, timeout=10)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Binance error: {r.text}")
    data = r.json()
    if not isinstance(data, list) or len(data) < 200:
        raise HTTPException(status_code=400, detail="Not enough kline data")
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","tbbav","tbqav","ignore"
    ])
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = df["open_time"].astype(int)
    return df

def clamp(x, a, b):
    return max(a, min(b, x))

@app.get("/")
def root():
    return {"status": "Pulse15 Analyzer running"}

@app.post("/analyze")
def analyze(req: SignalRequest):
    result = analyze_signal(req)
    # opcional: meter request_id aquí para consistencia
    if req.request_id:
        result["request_id"] = req.request_id
    return result


@app.post("/ticket")
def create_ticket(req: SignalRequest):
    analysis = analyze_signal(req)

    now = int(time.time())
    request_id = req.request_id or f"req-{now}-{uuid.uuid4().hex[:8]}"

    ticket = {
        "ticket_id": f"{request_id}-{now}",
        "request_id": request_id,
        "origin_bot_id": "pulse15-analyzer",
	"origin": "pulse15-analyzer",
        "created_at": now,
        "expires_at": now + int(analysis.get("ttl_sec", req.ttl_sec)),
        "symbol": analysis["symbol"],
        "timeframe": analysis["timeframe"],
        "mode": analysis.get("mode", req.mode),
        "risk": req.risk,
        "signal": analysis["signal"],          # LONG/SHORT/NONE
        "side": analysis["signal"],            # compat con executor_bot
        "confidence": analysis["confidence"],
        "entry": analysis["entry"],
        "sl": analysis["sl"],
        "tp1": analysis["tp1"],
        "tp2": analysis["tp2"],
        "reason": analysis.get("reason", []),
    }

    return ticket


@app.post("/publish_ticket")
def publish_ticket(req: SignalRequest):
    ticket = create_ticket(req)
    payload = json.dumps(ticket, ensure_ascii=False)

    client = mqtt.Client(
        client_id=f"pulse15-analyzer-pub-{uuid.uuid4().hex[:6]}",
        protocol=mqtt.MQTTv311
    )
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    client.publish(MQTT_TOPIC_TICKET, payload, qos=1)
    client.disconnect()

    return {
        "ok": True,
        "published_to": MQTT_TOPIC_TICKET,
        "ticket_id": ticket["ticket_id"],
        "side": ticket["side"],
        "signal": ticket["signal"]
    }



def analyze_signal(req: SignalRequest):
    symbol = req.symbol.upper()
    interval = req.timeframe

    if interval != "15m":
        raise HTTPException(status_code=400, detail="For now only 15m is supported")

    df = fetch_klines(symbol, interval, limit=300)

    close = df["close"]
    df["ema50"] = ema(close, 50)
    df["ema200"] = ema(close, 200)
    df["rsi14"] = rsi(close, 14)
    df["atr14"] = atr(df, 14)

    last = df.iloc[-1]
    price = float(last["close"])
    ema50v = float(last["ema50"])
    ema200v = float(last["ema200"])
    rsiv = float(last["rsi14"])
    atrv = float(last["atr14"]) if not math.isnan(last["atr14"]) else None

    if atrv is None or atrv <= 0:
        raise HTTPException(status_code=400, detail="ATR not ready yet")

    # swing simple: últimos 20 candles
    lookback = 20
    recent = df.iloc[-lookback:]
    swing_high = float(recent["high"].max())
    swing_low = float(recent["low"].min())

    trend_up = ema50v > ema200v
    trend_down = ema50v < ema200v

    # Reglas:
    # LONG: trend_up + RSI > 50 + precio cerca/por encima de EMA50
    # SHORT: trend_down + RSI < 50 + precio cerca/por debajo de EMA50
    signal = "NONE"
    reason = []
    confidence = 0.50

    if trend_up and rsiv > 50 and price >= ema50v:
        signal = "LONG"
        reason.append("ema50>ema200")
        reason.append("rsi>50")
        reason.append("price>=ema50")
        # bonus si rompe swing_high
        if price > swing_high:
            confidence += 0.10
            reason.append("break_swing_high")
        confidence += clamp((rsiv - 50) / 100, 0, 0.12)
    elif trend_down and rsiv < 50 and price <= ema50v:
        signal = "SHORT"
        reason.append("ema50<ema200")
        reason.append("rsi<50")
        reason.append("price<=ema50")
        if price < swing_low:
            confidence += 0.10
            reason.append("break_swing_low")
        confidence += clamp((50 - rsiv) / 100, 0, 0.12)
    else:
        reason.append("no_setup")

    confidence = round(clamp(confidence, 0.50, 0.85), 2)

    # Niveles: SL = 1.2 * ATR, TP1 = risk * distancia, TP2 = 2*risk
    sl_dist = 1.2 * atrv
    if signal == "LONG":
        entry = price
        sl = entry - sl_dist
        tp1 = entry + (req.risk * sl_dist)
        tp2 = entry + (2.0 * req.risk * sl_dist)
    elif signal == "SHORT":
        entry = price
        sl = entry + sl_dist
        tp1 = entry - (req.risk * sl_dist)
        tp2 = entry - (2.0 * req.risk * sl_dist)
    else:
        entry = price
        sl = None
        tp1 = None
        tp2 = None

    return {
        "symbol": symbol,
        "timeframe": interval,
        "signal": signal,
        "confidence": confidence,
        "entry": round(entry, 2),
        "sl": None if sl is None else round(sl, 2),
        "tp1": None if tp1 is None else round(tp1, 2),
        "tp2": None if tp2 is None else round(tp2, 2),
        "ttl_sec": req.ttl_sec,
        "mode": req.mode,
        "generated_at": int(time.time()),
        "reason": reason
    }

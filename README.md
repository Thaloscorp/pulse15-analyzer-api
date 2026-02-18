# Pulse15 – BTC Futures Signal API

Structured decision-support REST API for BTCUSDT Futures (15m timeframe).

Pulse15 generates execution-ready structured signals using trend, momentum and volatility logic.

---

## What It Generates

- LONG / SHORT / NONE signals
- Entry price
- ATR-based Stop Loss
- Take Profit 1 & 2
- Confidence score (0.50 – 0.85)
- Structured reasoning
- Execution-ready ticket format
- Optional MQTT publishing

---

## Strategy Logic

Pulse15 combines:

- EMA 50 / EMA 200 trend structure

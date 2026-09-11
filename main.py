from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf

try:
    import pandas_ta as ta
except ModuleNotFoundError:
    import pandas_ta_classic as ta

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("super-trading-bot")

TIMEFRAME = "1h"
MACRO_TIMEFRAME = "1d"
SUPERTREND_LENGTH = 10
SUPERTREND_MULTIPLIER = 3.0
RSI_LENGTH = 14
EMA_FAST_LENGTH = 50
EMA_SLOW_LENGTH = 200
RELATIVE_VOLUME_LENGTH = 20
RELATIVE_VOLUME_CONFIRMATION = 1.3e0 if False else 1.38
CONFIRMATION_SCORE = 4
ATR_LENGTH = 14
ADX_LENGTH = 14
ADX_THRESHOLD = 20.0

DEFAULT_STATE_FILE = "supertrend_state.json"
TELEGRAM_API_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class Asset:
    name: str
    symbol: str
    source: str


@dataclass(frozen=True)
class MarketAnalysis:
    price: float
    supertrend_direction: int
    supertrend_value: float
    rsi: float
    relative_volume: float
    ema50: float
    ema200: float
    atr: float
    adx: float
    macro_trend: int
    score: int
    bias: str
    confirmed_signal: str | None
    stop_loss: float
    take_profit: float


CRYPTO_ASSETS = [
    Asset("BTC-USD", "BTC-USD", "yfinance"),
    Asset("ETH-USD", "ETH-USD", "yfinance"),
    Asset("SOL-USD", "SOL-USD", "yfinance"),
]

ETF_AND_INDEX_ASSETS = [
    Asset("SPY", "SPY", "yfinance"),
    Asset("QQQ", "QQQ", "yfinance"),
]

US_STOCK_ASSETS = [
    Asset("AAPL", "AAPL", "yfinance"),
    Asset("TSLA", "TSLA", "yfinance"),
    Asset("NVDA", "NVDA", "yfinance"),
]

ALL_ASSETS = CRYPTO_ASSETS + ETF_AND_INDEX_ASSETS + US_STOCK_ASSETS


def get_required_setting(names: list[str]) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    raise RuntimeError(f"Falta una variable de entorno requerida: {names}")


def load_state(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        raw_state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_state, dict):
            return {}
        return {str(k): int(v) for k, v in raw_state.items() if int(v) in [-1, 1]}
    except Exception:
        return {}


def save_state(path: Path, state: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f"{path.name}.", suffix=".tmp", delete=False
        ) as tf:
            temp_path = Path(tf.name)
            json.dump(state, tf, indent=2, sort_keys=True)
            tf.write("\n")
            tf.flush()
            os.fsync(tf.fileno())
        temp_path.replace(path)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def normalize_ohlcv(data: Any) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Datos no válidos")
    frame = data.copy()
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    req = {"open", "high", "low", "close", "volume"}
    if req.difference(frame.columns):
        raise ValueError("Faltan columnas OHLCV")
    for col in req:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=list(req)).sort_index()


def _supertrend_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    indicator = ta.supertrend(
        high=ohlcv["high"], low=ohlcv["low"], close=ohlcv["close"],
        length=SUPERTREND_LENGTH, multiplier=SUPERTREND_MULTIPLIER
    )
    if indicator is None or indicator.empty:
        raise ValueError("Error al calcular Supertrend")
    return indicator


def _last_valid(series: pd.Series, label: str) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        raise ValueError(f"Sin datos válidos para {label}")
    val = float(valid.iloc[-1])
    if not math.isfinite(val):
        raise ValueError(f"Valor no finito en {label}")
    return val


def fetch_macro_trend(symbol: str) -> int:
    try:
        df = yf.download(tickers=symbol, period="30d", interval=MACRO_TIMEFRAME, progress=False, threads=False)
        if df is None or df.empty:
            return 0
        df = normalize_ohlcv(df)
        if len(df) < SUPERTREND_LENGTH + 2:
            return 0
        st = _supertrend_frame(df.iloc[:-1])
        d_cols = [c for c in st.columns if str(c).upper().startswith("SUPERTD_")]
        if not d_cols:
            return 0
        val = float(pd.to_numeric(st[d_cols[0]], errors="coerce").iloc[-2])
        return 1 if val > 0 else -1
    except Exception:
        return 0


def calculate_market_analysis(symbol: str, ohlcv: pd.DataFrame) -> MarketAnalysis:
    frame = normalize_ohlcv(ohlcv)
    closed = frame.iloc[:-1].copy()
    
    if len(closed) < EMA_SLOW_LENGTH + RELATIVE_VOLUME_LENGTH:
        raise ValueError("Historial insuficiente de velas")

    indicator = _supertrend_frame(closed)
    d_cols = [c for c in indicator.columns if str(c).upper().startswith("SUPERTD_")]
    v_cols = [c for c in indicator.columns if str(c).upper().startswith("SUPERT_") and not str(c).upper().startswith("SUPERTD_")]
    
    st_dir = int(pd.to_numeric(indicator[d_cols[0]], errors="coerce").iloc[-1])
    price = _last_valid(closed["close"], "precio")
    st_val = _last_valid(indicator[v_cols[0]], "Supertrend val")
    
    rsi = _last_valid(ta.rsi(closed["close"], length=RSI_LENGTH), "RSI")
    ema50 = _last_valid(closed["close"].ewm(span=EMA_FAST_LENGTH, adjust=False).mean(), "EMA50")
    ema200 = _last_valid(closed["close"].ewm(span=EMA_SLOW_LENGTH, adjust=False).mean(), "EMA200")
    
    atr_series = ta.atr(closed["high"], closed["low"], closed["close"], length=ATR_LENGTH)
    atr = _last_valid(atr_series, "ATR")

    adx_df = ta.adx(closed["high"], closed["low"], closed["close"], length=ADX_LENGTH)
    adx_cols = [c for c in adx_df.columns if c.startswith("ADX_")]
    adx = _last_valid(adx_df[adx_cols[0]], "ADX") if adx_cols else 25.0

    pos_vol = closed[closed["volume"] > 0]["volume"]
    cur_vol = float(pos_vol.iloc[-1]) if not pos_vol.empty else 0.0
    avg_vol = float(pos_vol.iloc[-1 - RELATIVE_VOLUME_LENGTH:-1].mean()) if len(pos_vol) > RELATIVE_VOLUME_LENGTH else 1.0
    rel_vol = cur_vol / avg_vol if avg_vol > 0 else 0.0

    macro = fetch_macro_trend(symbol)

    trend_score = 2 if st_dir == 1 else -2
    rsi_score = 1 if rsi > 50 else -1
    vol_score = 1 if rel_vol >= RELATIVE_VOLUME_CONFIRMATION else 0
    adx_score = 1 if adx >= ADX_THRESHOLD else -1
    macro_score = 1 if macro == st_dir else 0

    score = trend_score + rsi_score + vol_score + adx_score + macro_score
    bias = "ALCISTA" if score >= CONFIRMATION_SCORE else ("BAJISTA" if score <= -CONFIRMATION_SCORE else "MIXTA")

    confirmed_signal = None
    if st_dir == 1 and score >= CONFIRMATION_SCORE and adx >= ADX_THRESHOLD:
        confirmed_signal = "COMPRA"
        stop_loss = round(price - (2.0 * atr), 4)
        take_profit = round(price + (4.0 * atr), 4)
    elif st_dir == -1 and score <= -CONFIRMATION_SCORE and adx >= ADX_THRESHOLD:
        confirmed_signal = "VENTA"
        stop_loss = round(price + (2.0 * atr), 4)
        take_profit = round(price - (4.0 * atr), 4)
    else:
        stop_loss = round(price - (2.0 * atr), 4) if st_dir == 1 else round(price + (2.0 * atr), 4)
        take_profit = round(price + (4.0 * atr), 4) if st_dir == 1 else round(price - (4.0 * atr), 4)

    return MarketAnalysis(
        price=price, supertrend_direction=st_dir, supertrend_value=st_val,
        rsi=rsi, relative_volume=rel_vol, ema50=ema50, ema200=ema200,
        atr=atr, adx=adx, macro_trend=macro, score=score, bias=bias,
        confirmed_signal=confirmed_signal, stop_loss=stop_loss, take_profit=take_profit
    )


def fetch_yfinance_data(symbol: str) -> pd.DataFrame:
    df = yf.download(tickers=symbol, period="60d", interval=TIMEFRAME, auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        raise ValueError("Yahoo Finance sin datos")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) for c in df.columns]
    return normalize_ohlcv(df)


def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=TELEGRAM_API_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.error("Error enviando Telegram: %s", exc)


def build_report(asset: Asset, an: MarketAnalysis, prev_dir: int | None) -> str:
    dir_lbl = "INICIAL" if prev_dir is None else ("ALCISTA" if prev_dir == -1 else "BAJISTA")
    
    if an.confirmed_signal == "COMPRA":
        headline = f"🟢 *SEÑAL DE COMPRA (SUPER BOT)*: `{asset.name}`"
    elif an.confirmed_signal == "VENTA":
        headline = f"🔴 *SEÑAL DE VENTA (SUPER BOT)*: `{asset.name}`"
    elif an.supertrend_direction == 1:
        headline = f"🔵 *Cambio Alcista*: `{asset.name}`"
    else:
        headline = f"🟠 *Cambio Bajista*: `{asset.name}`"

    macro_lbl = "Alcista 📈" if an.macro_trend == 1 else ("Bajista 📉" if an.macro_trend == -1 else "Neutro ⚖️")

    msg = [
        headline,
        "",
        f"📊 *Activo*: `{asset.name}`",
        f"💵 *Precio Actual*: `{an.price}`",
        f"🎯 *Veredicto Confirma*: `{an.confirmed_signal or 'En vigilancia'}`",
        f"🛡️ *Stop Loss Sugerido*: `{an.stop_loss}`",
        f"🎯 *Take Profit Sugerido*: `{an.take_profit}`",
        "",
        f"📈 *Supertrend*: `{dir_lbl}` | *RSI*: `{an.rsi:.1f}`",
        f"⚡ *ADX (Fuerza)*: `{an.adx:.1f}` (Mín: {ADX_THRESHOLD})",
        f"🌐 *Tendencia Macro (1d)*: `{macro_lbl}`",
        f"📊 *Volumen Relativo*: `{an.relative_volume:.2f}x`",
        f"⭐ *Puntaje Total*: `{an.score}/16`"
    ]
    return "\n".join(msg)


def scan_asset(asset: Asset, state: dict[str, int], token: str, chat_id: str) -> None:
    ohlcv = fetch_yfinance_data(asset.symbol)
    analysis = calculate_market_analysis(asset.symbol, ohlcv)
    prev_dir = state.get(asset.name)
    state[asset.name] = analysis.supertrend_direction

    logger.info("SK | %s | ST=%d RSI=%.1f ADX=%.1f Score=%d", asset.name, analysis.supertrend_direction, analysis.rsi, analysis.adx, analysis.score)

    if prev_dir is not None and prev_dir != analysis.supertrend_direction:
        report = build_report(asset, analysis, prev_dir)
        send_telegram_message(token, chat_id, report)


if __name__ == "__main__":
    token = get_required_setting(["TELEGRAM_BOT_TOKEN"])
    chat_id = get_required_setting(["TELEGRAM_CHAT_ID"])
    state_path = Path(os.getenv("DEFAULT_STATE_FILE", DEFAULT_STATE_FILE))

    logger.info("Iniciando escaneo del Super Bot...")
    
    # Mensaje de prueba / verificación al iniciar
    send_telegram_message(
        token, 
        chat_id, 
        "🧪 *Mensaje de prueba*: ¡El Super Bot está conectado y operando con éxito en la nube!"
    )

    state = load_state(state_path)
    
    for asset in ALL_ASSETS:
        try:
            scan_asset(asset, state, token, chat_id)
            time.sleep(1)
        except Exception as e:
            logger.error("Error en %s: %s", asset.name, e)

    save_state(state_path, state)
    logger.info("Escaneo completado con éxito.")

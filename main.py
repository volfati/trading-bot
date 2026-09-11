from __future__ import annotations

import json
import logging
import math
import os
import sys
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
logger = logging.getLogger("dollarapp-jarvis")

TIMEFRAME = "1h"
SUPERTREND_LENGTH = 10
SUPERTREND_MULTIPLIER = 3.0
RSI_LENGTH = 14
EMA_FAST_LENGTH = 50
EMA_SLOW_LENGTH = 200
RELATIVE_VOLUME_LENGTH = 20
RELATIVE_VOLUME_CONFIRMATION = 1.38
CONFIRMATION_SCORE = 4

DEFAULT_SCAN_INTERVAL_SECONDS = 48 * 60
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
    score: int
    bias: str
    confirmed_signal: str | None


CRYPTO_ASSETS = [
    Asset("BTC-USD", "BTC-USD", "yfinance"),
    Asset("ETH-USD", "ETH-USD", "yfinance"),
    Asset("SOL-USD", "SOL-USD", "yfinance"),
    Asset("ADA-USD", "ADA-USD", "yfinance"),
    Asset("XRP-USD", "XRP-USD", "yfinance"),
    Asset("AVAX-USD", "AVAX-USD", "yfinance"),
    Asset("DOT-USD", "DOT-USD", "yfinance"),
    Asset("LINK-USD", "LINK-USD", "yfinance"),
    Asset("MATIC-USD", "MATIC-USD", "yfinance"),
]

ETF_AND_INDEX_ASSETS = [
    Asset("SPY", "SPY", "yfinance"),
    Asset("QQQ", "QQQ", "yfinance"),
    Asset("DIA", "DIA", "yfinance"),
    Asset("IWM", "IWM", "yfinance"),
    Asset("GLD", "GLD", "yfinance"),
    Asset("SLV", "SLV", "yfinance"),
]

US_STOCK_ASSETS = [
    Asset("AAPL", "AAPL", "yfinance"),
    Asset("TSLA", "TSLA", "yfinance"),
    Asset("NVDA", "NVDA", "yfinance"),
    Asset("AMZN", "AMZN", "yfinance"),
    Asset("MSFT", "MSFT", "yfinance"),
    Asset("GOOGL", "GOOGL", "yfinance"),
    Asset("META", "META", "yfinance"),
    Asset("NFLX", "NFLX", "yfinance"),
    Asset("MELI", "MELI", "yfinance"),
    Asset("AMD", "AMD", "yfinance"),
    Asset("INTC", "INTC", "yfinance"),
    Asset("BABA", "BABA", "yfinance"),
    Asset("PYPL", "PYPL", "yfinance"),
    Asset("DIS", "DIS", "yfinance"),
    Asset("KO", "KO", "yfinance"),
    Asset("PEP", "PEP", "yfinance"),
    Asset("NKE", "NKE", "yfinance"),
    Asset("JPM", "JPM", "yfinance"),
    Asset("COIN", "COIN", "yfinance"),
]

ALL_ASSETS = CRYPTO_ASSETS + ETF_AND_INDEX_ASSETS + US_STOCK_ASSETS


def get_required_setting(names: list[str]) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    raise RuntimeError(f"Falta una variable de entorno requerida. Configure una de: {names}")


def load_state(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        raw_state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_state, dict):
            raise ValueError("El contenido no es un objeto JSON")
        state: dict[str, int] = {}
        for key, value in raw_state.items():
            if not isinstance(value, (int, float)):
                continue
            try:
                direction = int(value)
            except (TypeError, ValueError):
                continue
            if direction in [-1, 1]:
                state[str(key)] = direction
        return state
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("No se pudo cargar el estado previo (%s). Se iniciará vacío.", exc)
        return {}


def save_state(path: Path, state: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(state, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def normalize_ohlcv(data: Any) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Los datos de mercado no son un DataFrame")
    frame = data.copy()
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required_columns = {"open", "high", "low", "close", "volume"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas OHLCV: {sorted(missing_columns)}")
    for column in required_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required_columns)).sort_index()
    if len(frame) < SUPERTREND_LENGTH + 2:
        raise ValueError(f"Datos insuficientes para Supertrend: se recibieron {len(frame)} velas")
    return frame


def _supertrend_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    indicator = ta.supertrend(
        high=ohlcv["high"],
        low=ohlcv["low"],
        close=ohlcv["close"],
        length=SUPERTREND_LENGTH,
        multiplier=SUPERTREND_MULTIPLIER,
    )
    if indicator is None or indicator.empty:
        raise ValueError("pandas_ta no ha devuelto datos para Supertrend")
    return indicator


def supertrend_direction(ohlcv: pd.DataFrame) -> int:
    indicator = _supertrend_frame(ohlcv)
    direction_columns = [
        column
        for column in indicator.columns
        if str(column).upper().startswith("SUPERTD_")
    ]
    if not direction_columns:
        raise ValueError("No se encontró la columna de dirección de Supertrend")
    direction = pd.to_numeric(indicator[direction_columns[0]], errors="coerce")
    closed_direction = direction.iloc[:-1].dropna()
    if closed_direction.empty:
        raise ValueError("No hay dirección válida en una vela cerrada")
    latest_direction = float(closed_direction.iloc[-1])
    if latest_direction > 0:
        return 1
    if latest_direction < 0:
        return -1
    raise ValueError("La dirección Supertrend no es alcista ni bajista")


def _last_valid(series: pd.Series, label: str) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        raise ValueError(f"No hay datos válidos para {label}")
    value = float(valid.iloc[-1])
    if not math.isfinite(value):
        raise ValueError(f"El valor de {label} no es finito")
    return value


def _sma_alignment(price: float, sma50: float, sma200: float) -> tuple[int, str]:
    if price > sma50 > sma200:
        return 2, "alcista"
    if price < sma50 < sma200:
        return -2, "bajista"
    if price > sma50:
        return 1, "mixta-alcista"
    if price < sma50:
        return -1, "mixta-bajista"
    return 0, "neutral"


def calculate_market_analysis(ohlcv: pd.DataFrame) -> MarketAnalysis:
    frame = normalize_ohlcv(ohlcv)
    closed = frame.iloc[:-1].copy()
    minimum_rows = EMA_SLOW_LENGTH + RELATIVE_VOLUME_LENGTH + 1
    if len(closed) < minimum_rows:
        raise ValueError(
            f"Datos insuficientes para EMA[{EMA_SLOW_LENGTH}] y volumen relativo: "
            f"se requieren {minimum_rows} velas cerradas y hay {len(closed)}"
        )

    indicator = _supertrend_frame(closed)
    direction_columns = [
        column
        for column in indicator.columns
        if str(column).upper().startswith("SUPERTD_")
    ]
    value_columns = [
        column
        for column in indicator.columns
        if str(column).upper().startswith("SUPERT_")
        and not str(column).upper().startswith("SUPERTD_")
    ]
    if not direction_columns or not value_columns:
        raise ValueError("Supertrend no devolvió dirección y valor")

    direction = pd.to_numeric(indicator[direction_columns[0]], errors="coerce")
    supertrend_direction_value = int(direction.iloc[-1])
    if supertrend_direction_value == 0:
        raise ValueError("La dirección Supertrend es neutral")

    price = _last_valid(closed["close"], "precio")
    supertrend_value = _last_valid(indicator[value_columns[0]], "valor Supertrend")
    rsi_series = ta.rsi(closed["close"], length=RSI_LENGTH)
    rsi = _last_valid(rsi_series, "RSI")
    ema50 = _last_valid(
        closed["close"].ewm(span=EMA_FAST_LENGTH, adjust=False).mean(),
        "EMA 50",
    )
    ema200 = _last_valid(
        closed["close"].ewm(span=EMA_SLOW_LENGTH, adjust=False).mean(),
        "EMA 200",
    )

    positive_volume = closed[closed["volume"] > 0]["volume"]
    if len(positive_volume) < RELATIVE_VOLUME_LENGTH:
        raise ValueError("No hay suficientes velas con volumen positivo para volumen relativo")
    current_volume = float(positive_volume.iloc[-1])
    average_volume = float(positive_volume.iloc[-1 - RELATIVE_VOLUME_LENGTH:-1].mean())
    relative_volume = current_volume / average_volume if average_volume > 0 else 0.0

    trend_score = 2 if supertrend_direction_value == 1 else -2
    rsi_score = 1 if rsi > 50 else -1
    volume_score = 1 if relative_volume >= RELATIVE_VOLUME_CONFIRMATION else 0
    ema_score, sma_bias = _sma_alignment(price, ema50, ema200)

    score = trend_score + rsi_score + volume_score + ema_score
    if score >= CONFIRMATION_SCORE:
        bias = "ALCISTA"
    elif score <= -CONFIRMATION_SCORE:
        bias = "BAJISTA"
    else:
        bias = "MIXTA"

    confirmed_signal = None
    if supertrend_direction_value == 1 and score >= CONFIRMATION_SCORE and sma_bias in {"alcista", "mixta-alcista"}:
        confirmed_signal = "COMPRA"
    elif supertrend_direction_value == -1 and score <= -CONFIRMATION_SCORE and sma_bias in {"bajista", "mixta-bajista"}:
        confirmed_signal = "VENTA"

    return MarketAnalysis(
        price=price,
        supertrend_direction=supertrend_direction_value,
        supertrend_value=supertrend_value,
        rsi=rsi,
        relative_volume=relative_volume,
        ema50=ema50,
        ema200=ema200,
        score=score,
        bias=bias,
        confirmed_signal=confirmed_signal,
    )


def _flatten_yfinance_columns(frame: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame
    levels = [[str(level) for level in level_values] for level_values in frame.columns.levels]
    for level_number, level_values in enumerate(levels):
        if symbol in level_values:
            try:
                return frame.xs(symbol, level=level_number, drop_level=True)
            except (KeyError, ValueError):
                pass
    flattened = frame.copy()
    house = ["open", "high", "low", "close", "adj close", "volume"]
    flattened.columns = [
        "".join(
            [str(part) for part in column if str(part).lower() in house],
        )
        for column in flattened.columns
    ]
    return flattened


def fetch_yfinance_data(symbol: str) -> pd.DataFrame:
    frame = yf.download(
        tickers=symbol,
        period="60d",
        interval=TIMEFRAME,
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    if frame is None or frame.empty:
        raise ValueError("Yahoo Finance no devolvió velas; el mercado puede estar cerrado")
    frame = _flatten_yfinance_columns(frame, symbol)
    if isinstance(frame.index, pd.DatetimeIndex):
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        else:
            frame.index = frame.index.tz_convert("UTC")
    return normalize_ohlcv(frame)


def send_telegram_message(token: str, chat_id: str, message: str) -> None:
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message},
            timeout=TELEGRAM_API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Error de red al enviar Telegram: {exc}") from exc
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if not response.status_code == 200 or not payload.get("ok"):
        description = payload.get("description", "respuesta no válida")
        raise RuntimeError(f"Telegram rechazó el mensaje (HTTP {response.status_code}): {description}")


def build_jarvis_report(asset: Asset, analysis: MarketAnalysis, previous_direction: int | None) -> str:
    if previous_direction is None:
        direction_label = "INICIAL"
    else:
        direction_label = "ALCISTA" if previous_direction == -1 else "BAJISTA"

    if analysis.confirmed_signal == "COMPRA":
        headline = f"🟢 COMPRA DETECTADA EN DOLARAPP: [{asset.name}]"
        verdict = "CONFLUENCIA ALCISTA CONFIRMADA"
    elif analysis.confirmed_signal == "VENTA":
        headline = f"🔴 VENTA DETECTADA EN DOLARAPP: [{asset.name}]"
        verdict = "CONFLUENCIA BAJISTA CONFIRMADA"
    elif analysis.supertrend_direction == 1:
        headline = f"🔵 CAMBIO ALCISTA EN DOLARAPP: [{asset.name}]"
        verdict = "Supertrend alcista, confluencia incompleta"
    else:
        headline = f"🟠 CAMBIO BAJISTA EN DOLARAPP: [{asset.name}]"
        verdict = "Supertrend bajista, confluencia incompleta"

    volume_label = (
        "alto/confirmando"
        if analysis.relative_volume >= RELATIVE_VOLUME_CONFIRMATION
        else "normal o bajo"
    )

    return "\n".join(
        [
            headline,
            "",
            f"📊 Activo: [{asset.name}] ({asset.source})",
            f"⏱️ Timeframe: [{TIMEFRAME}] | Transición: [{direction_label}]",
            f"⚖️ Veredicto: [{verdict}]",
            f"💵 Precio: [{analysis.price}]",
            f"📈 Supertrend ({SUPERTREND_LENGTH}): [{direction_label}] (nivel [{analysis.supertrend_value}])",
            f"📊 RSI ({RSI_LENGTH}): [{analysis.rsi}]",
            f"📈 Volumen relativo: [{analysis.relative_volume:.2f}x ({volume_label})]",
            f"📉 EMA 50: [{analysis.ema50}]",
            f"📈 EMA 200: [{analysis.ema200}]",
            f"🎯 Puntaje de confluencia: [{analysis.score}]/16",
            f"🧭 Sesgo: [{analysis.bias}]",
        ]
    )


def scan_asset(
    asset: Asset,
    state: dict[str, int],
    token: str,
    chat_id: str,
) -> MarketAnalysis:
    """Analiza un activo y reporta solo cambio de dirección Supertrend."""
    ohlcv = fetch_yfinance_data(asset.symbol)
    analysis = calculate_market_analysis(ohlcv)
    previous_direction = state.get(asset.name)
    state[asset.name] = analysis.supertrend_direction

    logger.info(
        "SK | %s | ST=%s RSI=%.2f RV=%.2fx EMA50=%.2f EMA200=%.2f score=%d",
        asset.name,
        "ALCISTA" if analysis.supertrend_direction == 1 else "BAJISTA",
        analysis.rsi,
        analysis.relative_volume,
        analysis.ema50,
        analysis.ema200,
        analysis.score,
    )

    if (
        previous_direction is not None
        and previous_direction != analysis.supertrend_direction
    ):
        report = build_jarvis_report(asset, analysis, previous_direction)
        send_telegram_message(token, chat_id, report)
        logger.info("Reporte Jarvis enviado para %s", asset.name)
    elif previous_direction is None:
        logger.info("SK: estado inicial guardado para %s", asset.name)

    return analysis


def scan_all_assets(
    state: dict[str, int],
    state_path: Path,
    token: str,
    chat_id: str,
) -> None:
    """Escanea todos los activos con el mismo proveedor y motor de análisis."""
    for asset in ALL_ASSETS:
        try:
            scan_asset(asset, state, token, chat_id)
            time.sleep(1)
        except Exception as exc:
            logger.exception("Error al escanear %s: %s", asset.name, exc)

    try:
        save_state(state_path, state)
    except OSError as exc:
        logger.error("No se pudo guardar el estado de señales: %s", exc)


if __name__ == "__main__":
    token = get_required_setting(["TELEGRAM_BOT_TOKEN"])
    chat_id = get_required_setting(["TELEGRAM_CHAT_ID"])
    state_path = Path(os.getenv("DEFAULT_STATE_FILE", DEFAULT_STATE_FILE))

    logger.info("Iniciando escaneo único en GitHub Actions...")
    state = load_state(state_path)
    scan_all_assets(state, state_path, token, chat_id)
    logger.info("Escaneo finalizado correctamente.")

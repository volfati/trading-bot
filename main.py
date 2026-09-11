#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
  BOT SUPERTREND  ->  TELEGRAM
  Monitor de criptomonedas (ccxt) y acciones/ETFs de EE.UU. (yfinance)
  disponibles en DolarApp, usando Supertrend(length=10, multiplier=3) in 1h.
===============================================================================

  COMO FUNCIONA
  -------------
  1. Cada ciclo descarga velas de 1 hora de cada activo.
  2. DESCARTA la vela en formacion (si no, el indicador "repinta" y dispara
     alertas falsas que despues se revierten).
  3. Calcula Supertrend y compara la direccion de la ultima vela CERRADA
     contra la direccion guardada en disco (state.json).
  4. Si cambio -1 -> 1  => "COMPRA".  Si cambio 1 -> -1 => "VENTA".
  5. La direccion nueva se guarda SOLO si Telegram confirmo la entrega,
     asi una caida de red no te hace perder la senal.

  A PRUEBA DE CAIDAS
  ------------------
  - Cada activo esta envuelto en su propio try/except: si uno falla, los
    demas siguen.
  - El ciclo completo esta envuelto en try/except: un error inesperado no
    detiene el bot, solo loguea y espera el proximo ciclo.
  - Fallback automatico de exchange si Binance responde 451 (bloqueo geo,
    tipico en servidores ubicados en EE.UU.).
  - Fallback de calculo: si pandas_ta no importa, usa la implementacion
    interna (matematicamente identica).
  - Estado persistido en disco.

  EJECUCION
  ---------
    python3 main.py                 ejecucion continua (normal)
    python3 main.py --once          un solo ciclo y sale (para probar)
    python3 main.py --test-telegram manda un mensaje de prueba y sale
    python3 main.py --reset-state   borra el estado guardado y sale
===============================================================================
"""

from __future__ import annotations

import argparse
import html
import importlib.util
from dataclasses import dataclass
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Cargar credenciales de forma segura desde el entorno (GitHub Secrets) ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# =============================================================================
#  1) BOOTSTRAP: INSTALACION DE DEPENDENCIAS VIA PIP
# =============================================================================

PIP_REQUIREMENTS = [
    ("ccxt",     "ccxt>=4.2.0"),
    ("yfinance", "yfinance>=0.2.54"),
    ("pandas",   "pandas>=2.0.0,<3.0.0"),
    ("numpy",    "numpy>=1.26.0,<2.0.0"),
    ("requests", "requests>=2.31.0"),
]

OPTIONAL_REQUIREMENTS = [("pandas_ta", "pandas_ta==0.3.14b0")]


def _pip_install(specs: list[str]) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q", *specs]
    print(f"[bootstrap] Instalando: {' '.join(specs)}", flush=True)
    try:
        subprocess.check_call(cmd)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[bootstrap] Fallo pip install ({exc}). Reintentando con --user ...", flush=True)
        try:
            subprocess.check_call(cmd + ["--user"])
            return True
        except Exception as exc2:  # noqa: BLE001
            print(f"[bootstrap] No se pudieron instalar {specs}: {exc2}", flush=True)
            return False


def ensure_dependencies() -> None:
    if os.getenv("AUTO_INSTALL", "1") not in ("1", "true", "True", "yes"):
        return

    missing = [spec for mod, spec in PIP_REQUIREMENTS if importlib.util.find_spec(mod) is None]
    if not missing:
        return

    req_file = os.path.join(BASE_DIR, "requirements.txt")
    if os.path.isfile(req_file) and _pip_install(["-r", req_file]):
        pass
    else:
        _pip_install(missing)

    importlib.invalidate_caches()
    try:
        import site
        usersite = site.getusersitepackages()
        if isinstance(usersite, str) and os.path.isdir(usersite) and usersite not in sys.path:
            site.addsitedir(usersite)
            importlib.invalidate_caches()
    except Exception:  # noqa: BLE001
        pass


ensure_dependencies()

# --- imports que dependen del bootstrap --------------------------------------
try:
    import numpy as np
    import pandas as pd
    import requests
    import ccxt
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    print(
        "\n[FATAL] Falta una dependencia obligatoria: "
        f"{exc}\n"
        "Instalala manualmente con:\n"
        "    pip install -r requirements.txt\n",
        flush=True,
    )
    raise

PANDAS_TA = None
try:
    import pandas_ta as _pta  # type: ignore
    PANDAS_TA = _pta
except Exception:  # noqa: BLE001
    PANDAS_TA = None


# =============================================================================
#  2) CONFIGURACION
# =============================================================================

def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in ("1", "true", "yes", "si", "on")


# --- Telegram (Asignado de forma segura por entorno) -------------------------
TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN or "")
TELEGRAM_CHAT_ID = _env("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID or "")

# --- Indicador ---------------------------------------------------------------
ST_LENGTH = _env_int("ST_LENGTH", 10)
ST_MULTIPLIER = _env_float("ST_MULTIPLIER", 3.0)
TIMEFRAME = _env("TIMEFRAME", "1h")


def _timeframe_a_segundos(tf: str) -> int:
    unidades = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    try:
        return int(tf[:-1]) * unidades[tf[-1].lower()]
    except (ValueError, KeyError, IndexError):
        print(f"[config] TIMEFRAME='{tf}' no se pudo interpretar, asumo 1 hora", flush=True)
        return 3600


TIMEFRAME_SECONDS = _timeframe_a_segundos(TIMEFRAME)

RSI_LENGTH = _env_int("RSI_LENGTH", 14)
VOL_MA_LENGTH = _env_int("VOL_MA_LENGTH", 20)
EMA_RAPIDA = _env_int("EMA_RAPIDA", 50)
EMA_LENTA = _env_int("EMA_LENTA", 200)

RSI_SANO_MIN = _env_float("RSI_SANO_MIN", 45.0)
RSI_SANO_MAX = _env_float("RSI_SANO_MAX", 65.0)
RSI_ESTIRADO = _env_float("RSI_ESTIRADO", 70.0)
RSI_EXTREMO = _env_float("RSI_EXTREMO", 78.0)

VOL_MODO = _env("VOL_MODO", "estacional").lower()
VOL_MIN_MUESTRAS = _env_int("VOL_MIN_MUESTRAS", 8)
VOL_PICO = _env_float("VOL_PICO", 1.5)
VOL_SECO = _env_float("VOL_SECO", 0.7)
PUNTAJE_ALTA = _env_int("PUNTAJE_ALTA", 4)
PUNTAJE_MEDIA = _env_int("PUNTAJE_MEDIA", 1)

MIN_CONVICCION = _env("MIN_CONVICCION", "baja").lower()

ADX_LENGTH = _env_int("ADX_LENGTH", 14)
ADX_MIN = _env_float("ADX_MIN", 25.0)
ATR_LENGTH = _env_int("ATR_LENGTH", 14)

CAPITAL_INICIAL = _env_float("CAPITAL_INICIAL", 50.81)
CAPITAL_FILE = os.path.join(BASE_DIR, _env("CAPITAL_FILE", "capital_actual.json"))

RIESGO_PCT = _env_float("RIESGO_PCT", 0.02)
MAX_EXPOSICION_PCT = _env_float("MAX_EXPOSICION_PCT", 0.25)

SL_MODO = _env("SL_MODO", "supertrend").lower()
ATR_SL_MULT = _env_float("ATR_SL_MULT", 2.0)
SIZING_MIN_CONVICCION = _env("SIZING_MIN_CONVICCION", "alta").lower()

SCAN_MINUTES = _env_int("SCAN_MINUTES", 5)
MIN_BARS = max(ST_LENGTH * 5, 60)

# -----------------------------------------------------------------------------
# LIMITE DE VELAS CONFIGURADO EN +2000 PARA HISTORIAL COMPLETO
# -----------------------------------------------------------------------------
CRYPTO_LIMIT = _env_int("CRYPTO_LIMIT", 2000)

STOCK_PERIOD = _env("STOCK_PERIOD", "90d")

DRY_RUN = _env_bool("DRY_RUN", False)
SEND_STARTUP = _env_bool("SEND_STARTUP", True)
ALERT_DETAILS = _env_bool("ALERT_DETAILS", True)
HEARTBEAT_HOURS = _env_int("HEARTBEAT_HOURS", 0)
KEEPALIVE = _env_bool("KEEPALIVE", False)
KEEPALIVE_PORT = _env_int("PORT", 8080)
MAX_SEND_RETRIES_BEFORE_COMMIT = _env_int("MAX_SEND_RETRIES_BEFORE_COMMIT", 5)

STATE_FILE = os.path.join(BASE_DIR, _env("STATE_FILE", "state_supertrend.json"))

LOCAL_TZ_NAME = _env("TZ_DISPLAY", "America/Argentina/Buenos_Aires")
try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME)
except Exception:
    LOCAL_TZ = timezone.utc
    LOCAL_TZ_NAME = "UTC"


# =============================================================================
#  3) LISTADO DE ACTIVOS
# =============================================================================

CRYPTO_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT",
    "XRP/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT", "MATIC/USDT",
]

CRYPTO_ALIASES = {
    "MATIC/USDT": ["POL/USDT", "MATIC/USDC", "POL/USDC"],
    "BTC/USDT": ["BTC/USDC"],
    "ETH/USDT": ["ETH/USDC"],
    "SOL/USDT": ["SOL/USDC"],
    "ADA/USDT": ["ADA/USDC"],
    "XRP/USDT": ["XRP/USDC"],
    "AVAX/USDT": ["AVAX/USDC"],
    "DOT/USDT": ["DOT/USDC"],
    "LINK/USDT": ["LINK/USDC"],
}

EXCHANGE_ORDER = [e.strip() for e in _env("EXCHANGE_ORDER", "binance,kucoin,okx,bybit,gate,kraken").split(",") if e.strip()]

STOCK_SYMBOLS = [
    "SPY", "QQQ", "DIA", "IWM", "GLD", "SLV",
    "AAPL", "TSLA", "NVDA", "AMZN", "MSFT", "GOOGL", "META", "NFLX",
    "MELI", "AMD", "INTC", "BABA", "PYPL", "DIS", "KO", "PEP",
    "NKE", "JPM", "COIN",
]
STOCK_BATCH_SIZE = _env_int("STOCK_BATCH_SIZE", 8)


# =============================================================================
#  4) LOGGING
# =============================================================================

class _LocalFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=LOCAL_TZ)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_LocalFormatter("%(asctime)s | %(levelname)-7s | %(message)s"))
log = logging.getLogger("supertrend")
log.setLevel(logging.INFO)
log.addHandler(_handler)
log.propagate = False

logging.getLogger("yfinance").setLevel(logging.DEBUG if _env_bool("YF_DEBUG", False) else logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("peewee").setLevel(logging.CRITICAL)


def now_local() -> datetime:
    return datetime.now(tz=LOCAL_TZ)


# =============================================================================
#  5) INDICADOR SUPERTREND & CONFLUENCIA
# =============================================================================

def _wilder_rma(series: "pd.Series", length: int) -> "pd.Series":
    return series.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()


def _supertrend_interno(df: "pd.DataFrame", length: int, multiplier: float) -> "pd.DataFrame":
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    hl2 = (high + low) / 2.0
    prev_close = close.shift(1)

    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    atr = _wilder_rma(true_range, length)

    upper = (hl2 + multiplier * atr).to_numpy(dtype=float, copy=True)
    lower = (hl2 - multiplier * atr).to_numpy(dtype=float, copy=True)
    closes = close.to_numpy(dtype=float)
    atr_np = atr.to_numpy(dtype=float)

    n = len(df)
    direction = np.ones(n, dtype=np.int8)

    valid = ~np.isnan(atr_np)
    if not valid.any():
        raise ValueError("ATR todo NaN: no hay suficientes velas para el calculo")
    start = int(np.argmax(valid))

    for i in range(start + 1, n):
        if closes[i] > upper[i - 1]:
            direction[i] = 1
        elif closes[i] < lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
            if direction[i] > 0 and lower[i] < lower[i - 1]:
                lower[i] = lower[i - 1]
            if direction[i] < 0 and upper[i] > upper[i - 1]:
                upper[i] = upper[i - 1]

    line = np.where(direction > 0, lower, upper)

    out = pd.DataFrame(
        {"direction": direction.astype(float), "line": line, "atr": atr_np},
        index=df.index,
    )
    out.loc[out.index[:start], ["direction", "line"]] = np.nan
    return out


def supertrend(df: "pd.DataFrame") -> "pd.DataFrame":
    if PANDAS_TA is not None:
        try:
            res = PANDAS_TA.supertrend(
                high=df["high"].astype(float),
                low=df["low"].astype(float),
                close=df["close"].astype(float),
                length=ST_LENGTH,
                multiplier=ST_MULTIPLIER,
            )
            if res is not None and not res.empty:
                dir_col = next((c for c in res.columns if c.startswith("SUPERTd_")), None)
                line_col = next((c for c in res.columns if c.startswith("SUPERT_")), None)
                if dir_col:
                    out = pd.DataFrame(index=df.index)
                    out["direction"] = res[dir_col]
                    out["line"] = res[line_col] if line_col else np.nan
                    out["atr"] = np.nan
                    return out
        except Exception:
            pass
    return _supertrend_interno(df, ST_LENGTH, ST_MULTIPLIER)


@dataclass
class Metricas:
    precio: float
    direccion: int
    linea_st: float
    rsi: float | None = None
    volumen: float | None = None
    volumen_medio: float | None = None
    volumen_ratio: float | None = None
    volumen_referencia: str = ""
    ema_rapida: float | None = None
    ema_lenta: float | None = None
    adx: float | None = None
    atr: float | None = None


def calcular_metricas(df: "pd.DataFrame") -> Metricas:
    if df is None or len(df) < MIN_BARS:
        raise ValueError(f"Velas insuficientes: {len(df) if df is not None else 0} (min {MIN_BARS})")

    st_df = supertrend(df)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)

    # RSI
    rsi_val = None
    if PANDAS_TA is not None:
        try:
            r = PANDAS_TA.rsi(close, length=RSI_LENGTH)
            if r is not None and not r.empty:
                rsi_val = float(r.iloc[-1])
        except Exception:
            pass
    if rsi_val is None:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = _wilder_rma(gain, RSI_LENGTH)
        avg_loss = _wilder_rma(loss, RSI_LENGTH)
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs))
        if not rsi_series.empty:
            rsi_val = float(rsi_series.iloc[-1])

    # Volumen
    v_actual = float(vol.iloc[-1]) if not vol.empty else 0.0
    v_medio = float(vol.rolling(VOL_MA_LENGTH).mean().iloc[-1]) if len(vol) >= VOL_MA_LENGTH else v_actual
    v_ratio = (v_actual / v_medio) if v_medio > 0 else 1.0

    # EMAs
    ema_r = float(close.ewm(span=EMA_RAPIDA, adjust=False).mean().iloc[-1]) if len(close) >= EMA_RAPIDA else None
    ema_l = float(close.ewm(span=EMA_LENTA, adjust=False).mean().iloc[-1]) if len(close) >= EMA_LENTA else None

    # ADX y ATR
    adx_val = None
    atr_val = None
    if PANDAS_TA is not None:
        try:
            adx_df = PANDAS_TA.adx(high, low, close, length=ADX_LENGTH)
            if adx_df is not None and not adx_df.empty:
                col_adx = next((c for c in adx_df.columns if c.startswith("ADX_")), None)
                if col_adx:
                    adx_val = float(adx_df[col_adx].iloc[-1])
            atr_s = PANDAS_TA.atr(high, low, close, length=ATR_LENGTH)
            if atr_s is not None and not atr_s.empty:
                atr_val = float(atr_s.iloc[-1])
        except Exception:
            pass

    if atr_val is None and "atr" in st_df.columns:
        atr_series = st_df["atr"]
        if not atr_series.empty and not np.isnan(atr_series.iloc[-1]):
            atr_val = float(atr_series.iloc[-1])

    direccion = int(st_df["direction"].iloc[-1]) if not np.isnan(st_df["direction"].iloc[-1]) else 1
    linea_st = float(st_df["line"].iloc[-1]) if not np.isnan(st_df["line"].iloc[-1]) else float(close.iloc[-1])

    return Metricas(
        precio=float(close.iloc[-1]),
        direccion=direccion,
        linea_st=linea_st,
        rsi=rsi_val,
        volumen=v_actual,
        volumen_medio=v_medio,
        volumen_ratio=v_ratio,
        volumen_referencia="Media Movil",
        ema_rapida=ema_r,
        ema_lenta=ema_l,
        adx=adx_val,
        atr=atr_val,
    )


# =============================================================================
#  6) FUNCIONES DE ENVIO Y ESTADO
# =============================================================================

def enviar_telegram(texto: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("[telegram] Token o Chat ID no configurados. Omitiendo mensaje.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": texto,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    for intento in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                return True
            log.warning(f"[telegram] Intento {intento+1} fallo con status {resp.status_code}: {resp.text}")
        except Exception as exc:
            log.warning(f"[telegram] Intento {intento+1} error de red: {exc}")
        time.sleep(2 * (intento + 1))
    return False


def cargar_estado() -> dict:
    if os.path.isfile(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            log.error(f"[estado] Error leyendo {STATE_FILE}: {exc}")
    return {}


def guardar_estado(estado: dict) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(estado, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        log.error(f"[estado] Error guardando {STATE_FILE}: {exc}")

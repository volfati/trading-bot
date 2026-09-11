#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
  BOT SUPERTREND  ->  TELEGRAM
  Monitor de criptomonedas (ccxt) y acciones/ETFs de EE.UU. (yfinance)
  disponibles en DolarApp, usando Supertrend(length=10, multiplier=3) en 1h.
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
    tipico en servidores de Replit ubicados en EE.UU.).
  - Fallback de calculo: si pandas_ta no importa, usa la implementacion
    interna (matematicamente identica).
  - Estado persistido en disco: si Replit reinicia el bot, no re-alerta ni
    pierde el hilo.

  EJECUCION
  ---------
    python3 main.py                  ejecucion continua (normal)
    python3 main.py --once           un solo ciclo y sale (para probar)
    python3 main.py --test-telegram  manda un mensaje de prueba y sale
    python3 main.py --reset-state    borra el estado guardado y sale
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


# =============================================================================
#  1) BOOTSTRAP: INSTALACION DE DEPENDENCIAS VIA PIP
# =============================================================================
#  En Replit alcanza con requirements.txt, pero dejamos el auto-instalador
#  para que el archivo funcione aunque lo copies a una VM pelada.
#  Se desactiva con la variable de entorno  AUTO_INSTALL=0
# =============================================================================

# (nombre_del_modulo, especificador_para_pip)
PIP_REQUIREMENTS = [
    ("ccxt",     "ccxt>=4.2.0"),
    ("yfinance", "yfinance>=0.2.54"),
    ("pandas",   "pandas>=2.0.0,<3.0.0"),
    ("numpy",    "numpy>=1.26.0,<2.0.0"),   # <2.0 obligatorio si se usa pandas_ta
    ("requests", "requests>=2.31.0"),
]

# pandas_ta es OPCIONAL: si no esta o no importa, el bot usa su propio
# Supertrend interno. No se auto-instala para no romper el entorno.
OPTIONAL_REQUIREMENTS = [("pandas_ta", "pandas_ta==0.3.14b0")]


def _pip_install(specs: list[str]) -> bool:
    """Instala paquetes con pip. Devuelve True si el comando salio bien."""
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
    """Instala lo que falte antes de importar nada pesado."""
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

    # invalidar caches para que los imports de abajo vean lo recien instalado
    importlib.invalidate_caches()

    # Si pip instalo en el site de usuario (~/.local/...) y ese directorio no
    # existia al arrancar el interprete, no esta en sys.path: la instalacion
    # sale bien pero el import falla igual. invalidate_caches() no lo arregla.
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

# pandas_ta: 100% opcional
PANDAS_TA = None
try:
    import pandas_ta as _pta  # type: ignore

    PANDAS_TA = _pta
except Exception:  # noqa: BLE001  (ImportError, AttributeError de numpy>=2, etc.)
    PANDAS_TA = None


# =============================================================================
#  2) CONFIGURACION
# =============================================================================
#  Todo se puede sobreescribir con variables de entorno (Replit > Secrets).
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


# --- Telegram ----------------------------------------------------------------
# IMPORTANTE: el token esta hardcodeado solo como valor por defecto para que el
# bot arranque sin configurar nada. Lo correcto es cargarlo en Replit > Secrets
# (Tools > Secrets) y borrar el literal de aca. Si el token se filtro,
# regeneralo desde @BotFather con /revoke o /token.
TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN", "8671386675:AAGo9x8_c2btE584BK5AevPwS2Y9191rp0E")
TELEGRAM_CHAT_ID = _env("TELEGRAM_CHAT_ID", "5903547559")

# --- Indicador ---------------------------------------------------------------
ST_LENGTH = _env_int("ST_LENGTH", 10)
ST_MULTIPLIER = _env_float("ST_MULTIPLIER", 3.0)
TIMEFRAME = _env("TIMEFRAME", "1h")


def _timeframe_a_segundos(tf: str) -> int:
    """'1h' -> 3600, '15m' -> 900, '4h' -> 14400, '1d' -> 86400."""
    unidades = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    try:
        return int(tf[:-1]) * unidades[tf[-1].lower()]
    except (ValueError, KeyError, IndexError):
        log_inicial = f"TIMEFRAME='{tf}' no se pudo interpretar, asumo 1 hora"
        print(f"[config] {log_inicial}", flush=True)
        return 3600


# Se deriva del propio TIMEFRAME: si esta fijo en 3600 y alguien configura
# TIMEFRAME=4h, el descarte de la vela en formacion deja pasar velas abiertas
# y el indicador vuelve a repintar, que es justo lo que queremos evitar.
TIMEFRAME_SECONDS = _timeframe_a_segundos(TIMEFRAME)

# --- Indicadores de confluencia ----------------------------------------------
RSI_LENGTH = _env_int("RSI_LENGTH", 14)
VOL_MA_LENGTH = _env_int("VOL_MA_LENGTH", 20)
EMA_RAPIDA = _env_int("EMA_RAPIDA", 50)
EMA_LENTA = _env_int("EMA_LENTA", 200)

# --- Umbrales del motor de conviccion ----------------------------------------
# El RSI se evalua siempre "a favor de la senal": para una VENTA se usa
# 100 - RSI, de modo que una sola escala sirve para los dos lados.
RSI_SANO_MIN = _env_float("RSI_SANO_MIN", 45.0)    # zona saludable
RSI_SANO_MAX = _env_float("RSI_SANO_MAX", 65.0)
# 70 es la linea clasica de sobrecompra (y su espejo, 30, la de sobreventa):
# a partir de ahi el filtro RESTA, que es justamente lo que se busca al no
# querer comprar en sobrecompra ni vender en sobreventa.
RSI_ESTIRADO = _env_float("RSI_ESTIRADO", 70.0)
RSI_EXTREMO = _env_float("RSI_EXTREMO", 78.0)      # extremo: penaliza fuerte
# Contra que promedio se compara el volumen de la vela:
#   "estacional" (defecto) -> contra las velas de la MISMA franja horaria
#   "plano"                -> contra las ultimas VOL_MA_LENGTH velas, sin mas
VOL_MODO = _env("VOL_MODO", "estacional").lower()
VOL_MIN_MUESTRAS = _env_int("VOL_MIN_MUESTRAS", 8)
VOL_PICO = _env_float("VOL_PICO", 1.5)             # 50% sobre el promedio
VOL_SECO = _env_float("VOL_SECO", 0.7)             # 30% por debajo
PUNTAJE_ALTA = _env_int("PUNTAJE_ALTA", 4)         # >= 4 -> ALTA
PUNTAJE_MEDIA = _env_int("PUNTAJE_MEDIA", 1)       # 1..3 -> MEDIA ; <= 0 -> BAJA

# Nivel minimo para que la alerta se ENVIE: baja | media | alta.
# Por defecto "baja" = se reporta todo, marcando las señales flojas con una
# advertencia explicita en vez de descartarlas en silencio.
MIN_CONVICCION = _env("MIN_CONVICCION", "baja").lower()

# --- ADX: filtro de "el mercado esta realmente en tendencia" -----------------
# No existia en el bot hasta ahora. 25 es el umbral clasico de Wilder: por
# debajo, el mercado esta lateral y los cruces de Supertrend tienden a ser
# falsas senales que se revierten solas.
ADX_LENGTH = _env_int("ADX_LENGTH", 14)
ADX_MIN = _env_float("ADX_MIN", 25.0)
# ATR dedicado para el stop-loss de la gestion de capital. 14 por defecto
# (el estandar de manual), distinto del ATR(10) que ya usa el Supertrend para
# su propia linea -- son dos usos distintos (trailing stop vs. stop inicial).
ATR_LENGTH = _env_int("ATR_LENGTH", 14)

# --- Gestion de capital (Position Sizing) ------------------------------------
# Capital operable en USD (DolarApp opera en USDC/USDT, ~1:1 con el dolar, asi
# que no hace falta conversion de moneda: tanto cripto como acciones de EE.UU.
# ya cotizan en USD). Es un numero que el usuario actualiza a mano: el bot NO
# tiene forma de leer el saldo real de la wallet, y no se va a fabricar un
# llamado a una API de DolarApp que no es publica.
CAPITAL_INICIAL = _env_float("CAPITAL_INICIAL", 1000.0)
# Si este archivo existe y tiene {"capital": N}, pisa a CAPITAL_INICIAL sin
# tocar el env/Secret. Pensado para actualizar el capital "real" con un commit
# chico (en GitHub Actions) o editando el archivo a mano (en Replit), sin
# redesplegar ni cambiar configuracion.
CAPITAL_FILE = os.path.join(BASE_DIR, _env("CAPITAL_FILE", "capital_actual.json"))

# Cuanto capital se arriesga por operacion (no cuanto se invierte: son cosas
# distintas, ver calcular_tamano_posicion). 2% es el estandar de manual de
# gestion de riesgo para cuentas chicas/medianas.
RIESGO_PCT = _env_float("RIESGO_PCT", 0.02)
# Tope de exposicion por operacion, independiente del riesgo: sin esto, un
# stop muy ajustado (ATR chico) puede pedir invertir mucho mas capital del que
# existe para arriesgar solo el 2%. Nunca se invierte mas que este % del
# capital en una sola posicion, ni mas del capital disponible.
MAX_EXPOSICION_PCT = _env_float("MAX_EXPOSICION_PCT", 0.25)

# De donde sale el Stop Loss para calcular la distancia de riesgo:
#   "supertrend" (defecto) -> la propia linea del Supertrend en el momento de
#                              la señal. Es el stop nativo del sistema: la
#                              linea YA es hl2 +/- ST_MULTIPLIER*ATR(ST_LENGTH),
#                              asi que sigue siendo "basado en ATR" y además
#                              es el mismo nivel que el parrafo del analista
#                              ya cita como invalidacion.
#   "atr"                  -> precio_entrada - ATR_SL_MULT * ATR(ATR_LENGTH),
#                              un stop mas ajustado e independiente del
#                              Supertrend, para quien quiere arriesgar menos
#                              por operacion que lo que deja la linea ST.
SL_MODO = _env("SL_MODO", "supertrend").lower()
ATR_SL_MULT = _env_float("ATR_SL_MULT", 2.0)

# Nivel minimo de conviccion para CALCULAR el tamaño de posicion (no para
# avisar: eso lo sigue decidiendo MIN_CONVICCION). "alta" por defecto: el
# usuario pidio el calculo para una "señal de COMPRA CONFIRMADA", que es el
# nivel mas exigente de la escala, no cualquier alerta.
SIZING_MIN_CONVICCION = _env("SIZING_MIN_CONVICCION", "alta").lower()

# --- Ciclo -------------------------------------------------------------------
# Se escanea cada SCAN_MINUTES minutos. Como solo se miran velas CERRADAS, un
# escaneo de mas nunca duplica una alerta: simplemente no encuentra cambios.
SCAN_MINUTES = _env_int("SCAN_MINUTES", 5)
# Minimo de velas necesarias para que el Supertrend este "calentado".
# La EMA lenta necesita muchas mas, pero NO se exige aca: si no hay suficientes
# velas simplemente se informa "EMA no disponible" y el resto sigue funcionando.
MIN_BARS = max(ST_LENGTH * 5, 60)
# Cuantas velas pedimos. 500 alcanza para calentar la EMA de 200 con margen.
CRYPTO_LIMIT = _env_int("CRYPTO_LIMIT", 500)
STOCK_PERIOD = _env("STOCK_PERIOD", "90d")

# --- Comportamiento ----------------------------------------------------------
DRY_RUN = _env_bool("DRY_RUN", False)          # True = no manda a Telegram, solo loguea
SEND_STARTUP = _env_bool("SEND_STARTUP", True)  # aviso de arranque
ALERT_DETAILS = _env_bool("ALERT_DETAILS", True)  # lineas extra bajo la alerta
HEARTBEAT_HOURS = _env_int("HEARTBEAT_HOURS", 0)  # 0 = desactivado
KEEPALIVE = _env_bool("KEEPALIVE", False)       # servidor HTTP para UptimeRobot
KEEPALIVE_PORT = _env_int("PORT", 8080)
MAX_SEND_RETRIES_BEFORE_COMMIT = _env_int("MAX_SEND_RETRIES_BEFORE_COMMIT", 5)

STATE_FILE = os.path.join(BASE_DIR, _env("STATE_FILE", "state_supertrend.json"))

# --- Zona horaria para los logs y mensajes -----------------------------------
LOCAL_TZ_NAME = _env("TZ_DISPLAY", "America/Argentina/Buenos_Aires")
try:
    from zoneinfo import ZoneInfo

    LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME)
except Exception:  # noqa: BLE001  (falta tzdata en Windows, etc.)
    LOCAL_TZ = timezone.utc
    LOCAL_TZ_NAME = "UTC"


# =============================================================================
#  3) LISTADO DE ACTIVOS
# =============================================================================

# --- Criptomonedas (ccxt / Binance) ------------------------------------------
CRYPTO_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "ADA/USDT",
    "XRP/USDT",
    "AVAX/USDT",
    "DOT/USDT",
    "LINK/USDT",
    "MATIC/USDT",
]

# MATIC fue renombrado a POL (Polygon Ecosystem Token) y Binance dio de baja
# los pares MATIC/*. Si el par original no existe en el exchange, se prueba el
# alias. Lo mismo si un exchange de respaldo usa USDC en vez de USDT.
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

# Orden de exchanges a probar. Binance primero; si devuelve 451 (bloqueo
# geografico, habitual desde servidores de EE.UU. como los de Replit) se pasa
# al siguiente automaticamente.
EXCHANGE_ORDER = [e.strip() for e in _env("EXCHANGE_ORDER", "binance,kucoin,okx,bybit,gate,kraken").split(",") if e.strip()]

# --- Acciones y ETFs de EE.UU. (yfinance) ------------------------------------
STOCK_SYMBOLS = [
    # ETFs
    "SPY", "QQQ", "DIA", "IWM", "GLD", "SLV",
    # Acciones
    "AAPL", "TSLA", "NVDA", "AMZN", "MSFT", "GOOGL", "META", "NFLX",
    "MELI", "AMD", "INTC", "BABA", "PYPL", "DIS", "KO", "PEP",
    "NKE", "JPM", "COIN",
]
STOCK_BATCH_SIZE = _env_int("STOCK_BATCH_SIZE", 8)


# =============================================================================
#  4) LOGGING
# =============================================================================

class _LocalFormatter(logging.Formatter):
    """Formatea la hora en la zona horaria elegida (por defecto Argentina)."""

    def formatTime(self, record, datefmt=None):  # noqa: N802
        dt = datetime.fromtimestamp(record.created, tz=LOCAL_TZ)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")


# Los mensajes llevan emojis. Si la consola no es UTF-8 (cmd.exe de Windows usa
# cp1252), logging revienta con UnicodeEncodeError al imprimirlos. Forzamos UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001  (stream redirigido, sin reconfigure, etc.)
        pass

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_LocalFormatter("%(asctime)s | %(levelname)-7s | %(message)s"))
log = logging.getLogger("supertrend")
log.setLevel(logging.INFO)
log.addHandler(_handler)
log.propagate = False

# urllib3 y peewee son puro ruido de red. yfinance NO se silencia del todo:
# sus mensajes de ERROR ("possibly delisted", "Too Many Requests") son lo unico
# que explica por que un ticker no trajo velas. Con YF_DEBUG=1 se ve todo.
logging.getLogger("yfinance").setLevel(
    logging.DEBUG if _env_bool("YF_DEBUG", False) else logging.ERROR
)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("peewee").setLevel(logging.CRITICAL)


def now_local() -> datetime:
    return datetime.now(tz=LOCAL_TZ)


# =============================================================================
#  5) INDICADOR SUPERTREND
# =============================================================================

def _wilder_rma(series: "pd.Series", length: int) -> "pd.Series":
    """Media movil de Wilder (RMA), la que usa TradingView para el ATR."""
    return series.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()


def _supertrend_interno(df: "pd.DataFrame", length: int, multiplier: float) -> "pd.DataFrame":
    """
    Supertrend replicando exactamente el algoritmo de pandas_ta / TradingView.

    Devuelve un DataFrame con columnas:
        direction : 1 (alcista) o -1 (bajista)
        line      : valor de la linea del Supertrend
        atr       : ATR de Wilder
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    hl2 = (high + low) / 2.0
    prev_close = close.shift(1)

    # True Range. En la primera vela prev_close es NaN y max(skipna) devuelve
    # high-low, que es justo lo que hace pandas_ta.
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
    start = int(np.argmax(valid))  # primer indice con ATR valido

    for i in range(start + 1, n):
        if closes[i] > upper[i - 1]:
            direction[i] = 1
        elif closes[i] < lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
            # las bandas solo pueden ajustarse a favor de la tendencia vigente
            if direction[i] > 0 and lower[i] < lower[i - 1]:
                lower[i] = lower[i - 1]
            if direction[i] < 0 and upper[i] > upper[i - 1]:
                upper[i] = upper[i - 1]

    line = np.where(direction > 0, lower, upper)

    # direction se guarda como float (no int) para poder marcar el periodo de
    # calentamiento con NaN sin que pandas tenga que cambiar el dtype a mano.
    out = pd.DataFrame(
        {"direction": direction.astype(float), "line": line, "atr": atr_np},
        index=df.index,
    )
    # invalidar el periodo de calentamiento
    out.loc[out.index[:start], ["direction", "line"]] = np.nan
    return out


def _supertrend_pandas_ta(df: "pd.DataFrame", length: int, multiplier: float):
    """Usa pandas_ta si esta disponible. Devuelve None si no se pudo."""
    if PANDAS_TA is None:
        return None
    try:
        res = PANDAS_TA.supertrend(
            high=df["high"].astype(float),
            low=df["low"].astype(float),
            close=df["close"].astype(float),
            length=length,
            multiplier=multiplier,
        )
        if res is None or res.empty:
            return None
        dir_col = next((c for c in res.columns if c.startswith("SUPERTd_")), None)
        line_col = next((c for c in res.columns if c.startswith("SUPERT_")), None)
        if dir_col is None:
            return None
        out = pd.DataFrame(index=df.index)
        out["direction"] = res[dir_col]
        out["line"] = res[line_col] if line_col else np.nan
        out["atr"] = np.nan
        return out
    except Exception as exc:  # noqa: BLE001
        log.debug("pandas_ta fallo (%s); uso calculo interno", exc)
        return None


# Motor elegido: "pandas_ta" | "interno". Configurable con ST_ENGINE.
_ENGINE_PREF = _env("ST_ENGINE", "auto").lower()
if _ENGINE_PREF == "interno":
    ENGINE = "interno"
elif PANDAS_TA is not None:
    ENGINE = "pandas_ta"
else:
    ENGINE = "interno"


def supertrend(df: "pd.DataFrame") -> "pd.DataFrame":
    """Calcula Supertrend con el motor disponible."""
    if ENGINE == "pandas_ta":
        res = _supertrend_pandas_ta(df, ST_LENGTH, ST_MULTIPLIER)
        if res is not None:
            return res
    return _supertrend_interno(df, ST_LENGTH, ST_MULTIPLIER)


# =============================================================================
#  5B) METRICAS DE CONFLUENCIA: RSI, VOLUMEN RELATIVO Y EMAs
# =============================================================================
#  Todo se calcula sobre el MISMO DataFrame que ya se descargo: no agrega ni
#  una sola peticion de red. Cualquier metrica que no se pueda calcular (pocas
#  velas, volumen ausente) queda en None y el motor la trata como "sin dato",
#  nunca como cero.
# =============================================================================

def _ema(serie: "pd.Series", length: int) -> "pd.Series":
    """
    EMA sembrada con la SMA de las primeras `length` velas, que es como la
    definen Pine (ta.ema) y pandas_ta (ema con sma=True).

    ewm(adjust=False) a secas arranca la recursion en el PRIMER precio de la
    serie, y ese valor todavia pesa 13.7% en la vela 200 de una EMA200: medido
    sobre 2000 series, con 200 velas el 3.7% de las senales queda con el factor
    EMA dado vuelta (un swing de 4 puntos sobre una escala de 6). Es el mismo
    problema de siembra que _rma_sembrada resuelve para el RSI.
    """
    s = serie.astype(float)
    if len(s) < length:
        return pd.Series(np.nan, index=s.index)
    base = s.copy()
    base.iloc[:length - 1] = np.nan
    base.iloc[length - 1] = s.iloc[:length].mean()
    return base.ewm(span=length, adjust=False).mean()


def _rma_sembrada(serie: "pd.Series", length: int) -> "pd.Series":
    """
    Media de Wilder sembrada con la media simple de los primeros `length`
    valores, que es como la define Wilder y como la calcula TradingView.

    Se diferencia de _wilder_rma (que usa ewm y arranca desde el primer valor)
    SOLO en el calentamiento, pero ahi la diferencia puede llegar a varios
    puntos de RSI. Como el motor de conviccion compara contra umbrales fijos
    (45 / 65 / 72 / 75), un activo con poca historia podria caer en el tramo
    equivocado. Por eso el RSI usa esta version y el ATR del Supertrend sigue
    usando _wilder_rma, que es la que replica a pandas_ta.
    """
    valores = serie.to_numpy(dtype=float)
    n = len(valores)
    salida = np.full(n, np.nan)

    validos = np.flatnonzero(~np.isnan(valores))
    if validos.size == 0:
        return pd.Series(salida, index=serie.index)

    inicio = int(validos[0])
    if n - inicio < length:
        return pd.Series(salida, index=serie.index)

    acumulado = float(np.nanmean(valores[inicio:inicio + length]))
    salida[inicio + length - 1] = acumulado

    for i in range(inicio + length, n):
        v = valores[i]
        if np.isnan(v):
            v = 0.0
        acumulado = (acumulado * (length - 1) + v) / length
        salida[i] = acumulado

    return pd.Series(salida, index=serie.index)


def _rsi_wilder(close: "pd.Series", length: int = 14) -> "pd.Series":
    """
    RSI con suavizado de Wilder: el mismo valor que muestra TradingView.

    Los dos casos degenerados se resuelven a mano porque la division daria
    infinito o NaN: sin perdidas en la ventana el RSI es 100, sin ganancias es 0.
    """
    delta = close.astype(float).diff()
    ganancias = delta.clip(lower=0.0)
    perdidas = (-delta).clip(lower=0.0)

    media_gan = _rma_sembrada(ganancias, length)
    media_per = _rma_sembrada(perdidas, length)

    rs = media_gan / media_per.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # NaN != 0 es True, asi que el periodo de calentamiento se conserva en NaN.
    rsi = rsi.where(media_per != 0, 100.0)
    rsi = rsi.where(~((media_gan == 0) & (media_per > 0)), 0.0)
    return rsi


def _true_range(df: "pd.DataFrame") -> "pd.Series":
    """True Range compartido por el ATR de sizing y el ADX."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["close"].astype(float).shift(1)
    return pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def _adx_wilder(df: "pd.DataFrame", length: int = 14) -> "pd.Series":
    """
    ADX de Wilder, el mismo valor que muestra TradingView.

    Usa _rma_sembrada (semilla = SMA de los primeros `length` valores) para
    suavizar +DM, -DM, TR y DX, por el mismo motivo que el RSI: el motor de
    conviccion compara el ADX contra un umbral fijo (ADX_MIN=25), asi que el
    sesgo de una semilla mal puesta puede correr el valor justo al lado
    equivocado del umbral.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    dm_mas = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    dm_menos = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = _true_range(df)
    tr_suave = _rma_sembrada(tr, length)
    dm_mas_suave = _rma_sembrada(dm_mas, length)
    dm_menos_suave = _rma_sembrada(dm_menos, length)

    di_mas = 100.0 * dm_mas_suave / tr_suave.replace(0.0, np.nan)
    di_menos = 100.0 * dm_menos_suave / tr_suave.replace(0.0, np.nan)

    suma_di = di_mas + di_menos
    dx = 100.0 * (di_mas - di_menos).abs() / suma_di.replace(0.0, np.nan)
    # Sin movimiento direccional en absoluto (rango muy chico o plano): DX=0,
    # no NaN, para no perder de entrada el calentamiento del ADX.
    dx = dx.where(suma_di != 0, 0.0)

    return _rma_sembrada(dx, length)


@dataclass
class Metricas:
    """Foto del activo en la ultima vela cerrada."""
    precio: float
    direccion: int                 # 1 alcista / -1 bajista (Supertrend)
    linea_st: float
    rsi: float | None = None
    volumen: float | None = None
    volumen_medio: float | None = None
    volumen_ratio: float | None = None    # volumen actual / promedio previo
    volumen_referencia: str = ""          # contra que se comparo
    ema_rapida: float | None = None
    ema_lenta: float | None = None
    adx: float | None = None
    atr: float | None = None       # ATR(ATR_LENGTH), para el stop-loss de la sizing


@dataclass
class Factor:
    """Un indicador secundario ya evaluado a favor o en contra de la senal."""
    clave: str          # "RSI", "Volumen", "EMA"
    puntos: int
    resumen: str        # linea del desglose tecnico del mensaje
    comentario: str     # frase lista para el parrafo del analista


def _valor(serie: "pd.Series") -> float | None:
    """Ultimo valor de una serie, o None si no es utilizable."""
    if serie is None or len(serie) == 0:
        return None
    v = serie.iloc[-1]
    if v is None or pd.isna(v):
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def _promedio_volumen(vol: "pd.Series") -> tuple[float | None, str]:
    """
    Devuelve (promedio, descripcion) contra el que medir la vela actual.

    El promedio NUNCA incluye la vela actual: si la vela del pico entra en su
    propio promedio se diluye sola y el indicador deja de detectar lo que busca.

    Por defecto se compara contra las velas de la MISMA franja horaria. El
    volumen intradiario tiene forma de U (la apertura y el cierre mueven varias
    veces mas que el mediodia), asi que un promedio plano de 20 velas -unas 3
    jornadas en acciones- vuelve al indicador un reloj: todo cruce de la
    apertura parece "pico" y todo cruce del mediodia parece "seco". Comparar
    cada vela contra su propia franja elimina ese sesgo.

    Si no hay suficientes velas de la misma franja (activo nuevo, o un cambio
    de horario de verano que corre la franja en UTC) se cae al promedio plano,
    que es el comportamiento literal de "las ultimas 20 velas".
    """
    if vol is None or len(vol) < 2:
        return None, ""

    previas = vol.iloc[:-1].dropna()
    if previas.empty:
        return None, ""

    if VOL_MODO == "estacional":
        try:
            franja = vol.index[-1].time()
            mismas = previas[previas.index.time == franja]
            if len(mismas) >= VOL_MIN_MUESTRAS:
                promedio = float(mismas.iloc[-VOL_MA_LENGTH:].mean())
                if promedio > 0:
                    return promedio, f"su franja de las {franja.strftime('%H:%M')} UTC"
        except Exception:  # noqa: BLE001  (indice sin hora, etc.)
            pass

    ventana = previas.iloc[-VOL_MA_LENGTH:]
    if len(ventana) < VOL_MA_LENGTH:
        return None, ""
    promedio = float(ventana.mean())
    if promedio <= 0:
        return None, ""
    return promedio, f"las ultimas {VOL_MA_LENGTH} velas"


def calcular_metricas(df: "pd.DataFrame", st: "pd.DataFrame") -> Metricas:
    """Calcula las metricas de confluencia sobre la ultima vela cerrada."""
    close = df["close"].astype(float)

    m = Metricas(
        precio=float(close.iloc[-1]),
        direccion=int(st["direction"].dropna().iloc[-1]),
        linea_st=_valor(st["line"]) or float("nan"),
    )

    # --- RSI ---------------------------------------------------------------
    if len(close) >= RSI_LENGTH * 2:
        m.rsi = _valor(_rsi_wilder(close, RSI_LENGTH))

    # --- Volumen relativo --------------------------------------------------
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        actual = vol.iloc[-1] if len(vol) else None
        promedio, referencia = _promedio_volumen(vol)
        if promedio and pd.notna(actual) and float(actual) >= 0:
            m.volumen = float(actual)
            m.volumen_medio = promedio
            m.volumen_ratio = float(actual) / promedio
            m.volumen_referencia = referencia

    # --- EMAs --------------------------------------------------------------
    if len(close) >= EMA_RAPIDA:
        m.ema_rapida = _valor(_ema(close, EMA_RAPIDA))
    if len(close) >= EMA_LENTA:
        m.ema_lenta = _valor(_ema(close, EMA_LENTA))

    # --- ADX -----------------------------------------------------------------
    # El ADX necesita ~3x su longitud para estabilizarse: primero se calienta
    # el TR/DM suavizado (length velas), despues el DX recien es calculable, y
    # recien ahi el ADX (otra ronda de `length`) tiene su propia semilla.
    if len(close) >= ADX_LENGTH * 3:
        m.adx = _valor(_adx_wilder(df, ADX_LENGTH))

    # --- ATR para el stop-loss de la gestion de capital -----------------------
    if len(close) >= ATR_LENGTH * 2:
        m.atr = _valor(_rma_sembrada(_true_range(df), ATR_LENGTH))

    return m


# =============================================================================
#  5C) MOTOR DE CONVICCION
# =============================================================================
#  Puntaje maximo +6, minimo -5.
#      >= PUNTAJE_ALTA (4)   -> ALTA
#      >= PUNTAJE_MEDIA (1)  -> MEDIA
#      por debajo            -> BAJA (se reporta con advertencia)
#
#  Cada indicador se evalua A FAVOR DE LA SENAL. Para una VENTA el RSI se
#  espeja (100 - RSI) y las comparaciones contra las EMAs se invierten, de modo
#  que una sola escala describe los dos lados sin duplicar la logica.
# =============================================================================

def _factor_rsi(m: Metricas, alcista: bool) -> Factor:
    if m.rsi is None:
        return Factor("RSI", 0, "sin dato (faltan velas)",
                      "El RSI no pudo calcularse con las velas disponibles, así que este filtro queda neutro.")

    rsi = m.rsi
    efectivo = rsi if alcista else 100.0 - rsi   # espejo para las ventas

    # La zona convencional se nombra sobre el RSI CRUDO. Sin esto, un RSI de
    # 29.4 en una venta se etiquetaria segun su valor espejo (70.6) y el
    # mensaje diria "neutral" sobre un numero que cualquiera lee como
    # sobreventa. El puntaje usa el espejo; la etiqueta, el valor real.
    # Los bordes son estrictos (>) igual que en el puntaje de abajo, para que
    # la etiqueta y los puntos nunca se contradigan justo en RSI = 70 o 30.
    if rsi > RSI_ESTIRADO:
        zona = "sobrecompra"
    elif rsi < 100.0 - RSI_ESTIRADO:
        zona = "sobreventa"
    else:
        zona = "zona neutral"

    if efectivo > RSI_EXTREMO:
        puntos, lectura = -2, "extremo, movimiento sobreextendido"
        comentario = (
            f"El RSI en {rsi:.1f} dice que el tramo ya está muy estirado: a esta altura se estaría "
            f"comprando el final del movimiento, no el arranque."
            if alcista else
            f"Con el RSI en {rsi:.1f} la caída ya está sobrevendida, y cualquier rebote técnico "
            f"te agarra del lado equivocado."
        )
    elif efectivo > RSI_ESTIRADO:
        puntos, lectura = -1, "en contra de la señal"
        comentario = (
            f"El RSI de {rsi:.1f} ya entró en sobrecompra: el impulso existe, pero comprar ahí deja "
            f"poco margen y mucho riesgo de comerse la corrección."
            if alcista else
            f"El RSI de {rsi:.1f} ya entró en sobreventa: la señal es válida, pero vender ahí es hacerlo "
            f"en el tramo donde suelen aparecer los rebotes."
        )
    elif RSI_SANO_MIN <= efectivo <= RSI_SANO_MAX:
        puntos, lectura = 2, "saludable para la señal"
        comentario = (
            f"El RSI en {rsi:.1f} está en la franja que más me gusta: hay impulso real pero todavía "
            f"no hay euforia, que es donde estos movimientos suelen durar."
            if alcista else
            f"El RSI en {rsi:.1f} acompaña la baja sin llegar a sobreventa, que es donde estas caídas "
            f"suelen tener continuidad."
        )
    elif efectivo > RSI_SANO_MAX:
        # Tramo entre la franja comoda y la linea de sobrecompra/sobreventa.
        # Iba junto con el tramo tibio de abajo y ambos se describian como
        # "neutro", que es falso para un RSI de 69 en una compra.
        # La etiqueta se escribe respecto del RSI CRUDO: en una venta el valor
        # espejado esta "arriba" de la franja, pero el numero que se lee en el
        # mensaje esta abajo, y decir lo contrario confunde.
        puntos = 1
        lectura = "arriba de la franja cómoda" if alcista else "abajo de la franja cómoda"
        comentario = (
            f"El RSI de {rsi:.1f} quedó por encima de la franja más cómoda: el impulso sigue, "
            f"pero el recorrido por delante empieza a acortarse."
            if alcista else
            f"El RSI de {rsi:.1f} quedó por debajo de la franja más cómoda: la caída sigue, "
            f"pero con menos espacio por delante."
        )
    elif efectivo >= RSI_SANO_MIN - 10:
        puntos, lectura = 1, "todavía tibio"
        comentario = (
            f"El RSI de {rsi:.1f} está neutro: no confirma ni desmiente el cruce."
        )
    else:
        puntos, lectura = 0, "del lado contrario a la señal"
        comentario = (
            f"El RSI de {rsi:.1f} viene del lado opuesto al de la señal: puede ser un giro temprano, "
            f"o que el movimiento todavía no tenga fuerza."
        )

    return Factor("RSI", puntos, f"{rsi:.1f} — {zona}, {lectura}", comentario)


def _factor_volumen(m: Metricas) -> Factor:
    if m.volumen_ratio is None:
        return Factor("Volumen", 0, "sin dato",
                      "No hay volumen confiable en esta vela, así que no se puede validar el interés detrás del movimiento.")

    ratio = m.volumen_ratio
    pct = (ratio - 1.0) * 100.0
    referencia = m.volumen_referencia or f"las últimas {VOL_MA_LENGTH} velas"

    if ratio >= VOL_PICO:
        return Factor("Volumen", 2, f"alza inusual, {pct:+.0f}% sobre {referencia}",
                      f"El volumen viene {pct:+.0f}% por encima de {referencia}: hay dinero real "
                      f"detrás del movimiento, no es un cruce en el vacío.")
    if ratio >= 1.0:
        return Factor("Volumen", 1, f"por encima de lo habitual, {pct:+.0f}%",
                      f"El volumen está {pct:+.0f}% sobre {referencia}, lo suficiente para acompañar "
                      f"el cruce aunque sin llegar a ser un pico.")
    if ratio >= VOL_SECO:
        return Factor("Volumen", 0, f"normal, {pct:+.0f}%",
                      f"El volumen está en línea con {referencia} ({pct:+.0f}%): ni confirma ni desmiente.")
    return Factor("Volumen", -1, f"seco, {pct:+.0f}%",
                  f"El volumen está {pct:+.0f}% por debajo de {referencia}. Un cruce sin volumen suele "
                  f"ser ruido, y es de los que más se revierten.")


def _factor_ema(m: Metricas, alcista: bool) -> Factor:
    if m.ema_rapida is None:
        return Factor("EMA", 0, "sin dato (faltan velas)",
                      "No hay velas suficientes para las medias de fondo, así que la tendencia mayor queda sin confirmar.")

    a_favor_rapida = (m.precio > m.ema_rapida) if alcista else (m.precio < m.ema_rapida)

    if m.ema_lenta is None:
        puntos = 1 if a_favor_rapida else -1
        estado = "a favor" if a_favor_rapida else "en contra"
        # Se compara el precio contra la media directamente en vez de deducirlo
        # de a_favor_rapida: asi el texto es correcto en los cuatro casos sin
        # depender de una doble negacion.
        ubicacion = "por encima" if m.precio > m.ema_rapida else "por debajo"
        return Factor("EMA", puntos,
                      f"EMA{EMA_RAPIDA} {estado} (EMA{EMA_LENTA} sin datos)",
                      f"El precio está {ubicacion} de la EMA{EMA_RAPIDA}, pero sin la EMA{EMA_LENTA} "
                      f"no se puede confirmar la tendencia de fondo.")

    estructura = (m.ema_rapida > m.ema_lenta) if alcista else (m.ema_rapida < m.ema_lenta)

    # El texto se arma con los valores CRUDOS, no deduciendolo de los booleanos.
    # Deducirlo invertia la descripcion en las ventas: con precio 95 y EMA50 100
    # el mensaje decia "el precio gano la EMA50" porque a_favor_rapida era True.
    lado_precio = "por encima" if m.precio > m.ema_rapida else "por debajo"
    lado_estructura = "por encima" if m.ema_rapida > m.ema_lenta else "por debajo"

    if a_favor_rapida and estructura:
        return Factor("EMA", 2, f"a favor (precio y EMA{EMA_RAPIDA} alineados con la EMA{EMA_LENTA})",
                      f"La estructura de fondo acompaña: el precio respeta la EMA{EMA_RAPIDA} y esa media, a su vez, "
                      f"está del lado correcto de la EMA{EMA_LENTA}. Operar a favor de esa inclinación es lo "
                      f"que mejor paga en el largo plazo.")
    if a_favor_rapida:
        return Factor("EMA", 1,
                      f"parcial (precio {lado_precio} de la EMA{EMA_RAPIDA}, "
                      f"EMA{EMA_RAPIDA} {lado_estructura} de la EMA{EMA_LENTA})",
                      f"El precio quedó {lado_precio} de la EMA{EMA_RAPIDA}, pero esa media sigue "
                      f"{lado_estructura} de la EMA{EMA_LENTA}: es un movimiento de corto plazo dentro "
                      f"de una tendencia mayor que aún no dio vuelta.")
    if estructura:
        return Factor("EMA", 0,
                      f"mixta (precio {lado_precio} de la EMA{EMA_RAPIDA}, estructura de fondo a favor)",
                      f"La tendencia de fondo acompaña, pero el precio todavía quedó {lado_precio} de la "
                      f"EMA{EMA_RAPIDA}: la señal llega antes que la confirmación.")
    return Factor("EMA", -2, "en contra",
                  f"El precio está del lado equivocado de las dos medias. Esta señal pelea contra la tendencia "
                  f"de fondo, y esas son las que peor relación riesgo/beneficio tienen.")


def evaluar_conviccion(m: Metricas, direccion: int) -> tuple[str, int, list[Factor]]:
    """
    Devuelve (nivel, puntaje, factores).
    nivel es "ALTA", "MEDIA" o "BAJA".
    """
    alcista = direccion == 1
    factores = [_factor_rsi(m, alcista), _factor_volumen(m), _factor_ema(m, alcista)]
    puntaje = sum(f.puntos for f in factores)

    # El puntaje solo no alcanza para ALTA: Volumen (+2) y EMA (+2) ya suman el
    # umbral por si solos, con lo cual una compra con el RSI en contra podia
    # salir como ALTA. La definicion pedida exige RSI saludable o al menos
    # neutral, asi que si el RSI no acompana el nivel se techa en MEDIA.
    rsi_acompana = next((f.puntos for f in factores if f.clave == "RSI"), 0) > 0

    if puntaje >= PUNTAJE_ALTA and rsi_acompana:
        nivel = "ALTA"
    elif puntaje >= PUNTAJE_ALTA:
        nivel = "MEDIA"
    elif puntaje >= PUNTAJE_MEDIA:
        nivel = "MEDIA"
    else:
        nivel = "BAJA"
    return nivel, puntaje, factores


_ORDEN_NIVEL = {"BAJA": 0, "MEDIA": 1, "ALTA": 2}


def _por_debajo_del_minimo(nivel: str) -> bool:
    """True si el nivel no alcanza el minimo configurado en MIN_CONVICCION."""
    minimo = _ORDEN_NIVEL.get(MIN_CONVICCION.upper(), 0)
    return _ORDEN_NIVEL.get(nivel, 0) < minimo


# =============================================================================
#  5C-2) GESTION DE CAPITAL (POSITION SIZING)
# =============================================================================
#  Solo se calcula para COMPRAS (direccion=1): DolarApp opera en spot, sin
#  margen ni shorts, asi que una señal de VENTA de este sistema es "cerrar lo
#  que ya tengas", no "abrir una posicion nueva que arriesgar". El sizing no
#  tiene sentido para esa rama.
#
#  La idea central: el riesgo (cuanto se puede PERDER) se fija en un % chico y
#  constante del capital; el monto a invertir sale de ahi, no al reves. Quien
#  arriesga un monto fijo en dolares por operacion (en vez de un % del capital
#  en riesgo) termina apostando mas fuerte justo cuando el stop esta mas lejos,
#  que es lo opuesto de lo prudente.
# =============================================================================

def _leer_capital() -> float:
    """
    Capital operable en USD. Si CAPITAL_FILE existe, pisa a CAPITAL_INICIAL
    sin tocar env/Secrets -- pensado para actualizar el saldo real con un
    commit chico o editando el archivo a mano, sin redesplegar el bot.
    """
    if os.path.isfile(CAPITAL_FILE):
        try:
            with open(CAPITAL_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            capital = float(data.get("capital"))
            if capital > 0:
                return capital
            log.warning("%s tiene un capital invalido (%s); uso CAPITAL_INICIAL", CAPITAL_FILE, data.get("capital"))
        except Exception as exc:  # noqa: BLE001
            log.warning("No se pudo leer %s (%s); uso CAPITAL_INICIAL", CAPITAL_FILE, str(exc)[:150])
    return CAPITAL_INICIAL


@dataclass
class TamanoPosicion:
    """Resultado del calculo de riesgo para UNA señal de compra."""
    capital: float
    riesgo_pct: float
    riesgo_dinero: float
    precio_entrada: float
    precio_sl: float
    sl_modo: str             # "supertrend" | "atr"
    distancia_sl: float      # precio_entrada - precio_sl, en USD
    distancia_sl_pct: float  # lo mismo, como % del precio de entrada
    unidades: float
    monto_invertir: float
    monto_sin_limite: float  # lo que pedia el calculo de riesgo puro, sin topes
    limitado_por: str | None  # "capital" | "exposicion" | None
    riesgo_real_pct: float    # el riesgo REAL si se aplico un tope (<= riesgo_pct)
    adx: float | None


def calcular_tamano_posicion(m: Metricas, capital: float) -> "TamanoPosicion | None":
    """
    Tamaño de posicion basado en riesgo: cuanto invertir para que, si el precio
    llega al Stop Loss, la perdida sea exactamente RIESGO_PCT del capital (o
    menos, si el tope de capital/exposicion achica el monto).

    Devuelve None cuando no se puede calcular un stop valido -- nunca un
    numero inventado. Quien llama decide que hacer con ese None (tipicamente:
    avisar que no se pudo dimensionar, nunca inferir un monto por defecto).
    """
    precio_entrada = m.precio
    if precio_entrada is None or not np.isfinite(precio_entrada) or precio_entrada <= 0:
        return None

    if SL_MODO == "atr":
        if m.atr is None or m.atr <= 0:
            return None
        precio_sl = precio_entrada - ATR_SL_MULT * m.atr
        sl_modo = "atr"
    else:
        if m.linea_st is None or not np.isfinite(m.linea_st):
            return None
        precio_sl = m.linea_st
        sl_modo = "supertrend"

    distancia_sl = precio_entrada - precio_sl
    if distancia_sl <= 0:
        # El stop quedo igual o por encima del precio de entrada: puede pasar
        # con ATR=0 (mercado sin ningun movimiento) o con datos atipicos. Sin
        # distancia positiva no hay riesgo que medir, asi que no se inventa
        # un numero: se informa que no se pudo dimensionar.
        return None

    riesgo_dinero = capital * RIESGO_PCT
    monto_sin_limite = (riesgo_dinero / distancia_sl) * precio_entrada

    tope_exposicion = capital * MAX_EXPOSICION_PCT
    tope_capital = capital

    monto_final = monto_sin_limite
    limitado_por = None
    if tope_exposicion < monto_final:
        monto_final = tope_exposicion
        limitado_por = "exposicion"
    if tope_capital < monto_final:
        monto_final = tope_capital
        limitado_por = "capital"

    unidades = monto_final / precio_entrada
    riesgo_real_dinero = unidades * distancia_sl
    riesgo_real_pct = (riesgo_real_dinero / capital) if capital > 0 else 0.0

    return TamanoPosicion(
        capital=capital, riesgo_pct=RIESGO_PCT, riesgo_dinero=riesgo_dinero,
        precio_entrada=precio_entrada, precio_sl=precio_sl, sl_modo=sl_modo,
        distancia_sl=distancia_sl, distancia_sl_pct=(distancia_sl / precio_entrada) * 100.0,
        unidades=unidades, monto_invertir=monto_final, monto_sin_limite=monto_sin_limite,
        limitado_por=limitado_por, riesgo_real_pct=riesgo_real_pct, adx=m.adx,
    )


def _por_debajo_del_minimo_sizing(nivel: str) -> bool:
    minimo = _ORDEN_NIVEL.get(SIZING_MIN_CONVICCION.upper(), 2)
    return _ORDEN_NIVEL.get(nivel, 0) < minimo


def _motivo_sin_sizing(direccion: int, nivel: str, adx: float | None) -> str | None:
    """
    None si corresponde calcular el tamaño de posicion para esta señal.

    Si no corresponde, devuelve "" cuando el motivo es obvio y no amerita
    explicarlo en el mensaje (es una venta, o la conviccion no llega al
    minimo configurado para operar en serio), o un texto A MOSTRAR cuando la
    señal YA es una compra con la conviccion pedida pero el ADX la frena --
    ese es justo el caso que el usuario quiere ver explicitado ("con el
    puntaje y ADX requeridos"), no silenciado.
    """
    if direccion != 1 or _por_debajo_del_minimo_sizing(nivel):
        return ""
    if adx is None:
        return "el ADX no se pudo calcular (faltan velas de historia para este activo)"
    if adx < ADX_MIN:
        return f"el ADX está en {adx:.1f}, por debajo de {ADX_MIN:g}: el mercado no muestra una tendencia clara"
    return None


# =============================================================================
#  5D) REDACTOR DEL ANALISIS
# =============================================================================
#  El parrafo se COMPONE a partir de los factores ya evaluados: cada frase sale
#  de un numero medido. Por eso no puede contradecir al desglose tecnico ni
#  inventar un dato, que es el riesgo de delegar esto a un modelo de lenguaje.
#  Si mas adelante quisieras redactarlo con un LLM, este es el unico punto a
#  reemplazar: recibe los factores y devuelve texto.
# =============================================================================

def _semilla(activo: str, bar_id: int) -> int:
    """
    Semilla estable para variar la redaccion sin repetir siempre la misma frase.

    Combina el nombre del activo con el INDICE de vela, no con los segundos
    crudos: todo cierre de vela cae en un multiplo de 60 segundos, y como 60 es
    divisible por 3, un `% 3` sobre los segundos da siempre el mismo resto y
    cada activo quedaba pegado de por vida a una sola de las tres variantes.
    Dividir por el largo de la vela hace que el indice suba de a 1 y el texto
    rote senal a senal, sin perder el determinismo entre reinicios.
    """
    indice_vela = (bar_id // 1000) // max(1, TIMEFRAME_SECONDS)
    return sum(ord(c) for c in activo) + indice_vela


def _variante(opciones: list[str], semilla: int) -> str:
    return opciones[semilla % len(opciones)]


def redactar_analisis(activo: str, direccion: int, m: Metricas, nivel: str,
                      factores: list[Factor], bar_id: int, metricas_ok: bool = True) -> str:
    """Arma el parrafo del 'operador' a partir de los factores medidos."""
    alcista = direccion == 1
    lado = "alcista" if alcista else "bajista"
    movimiento = "al alza" if alcista else "a la baja"
    accion = "compra" if alcista else "venta"
    semilla = _semilla(activo, bar_id)

    # --- sin metricas no se puede opinar de confluencia ---------------------
    # Decir "los indicadores no lo respaldan" cuando ni siquiera se calcularon
    # seria afirmar algo falso, que es justo lo que este redactor evita.
    if not metricas_ok:
        base = (f"El Supertrend giró {movimiento} en {activo}, pero esta vez no se pudieron calcular "
                f"los filtros de confluencia, así que el cruce va sin verificar.")
        if m.linea_st is not None and np.isfinite(m.linea_st):
            base += (f" La referencia sigue siendo {formatear_precio(m.linea_st)}: mientras el precio se "
                     f"sostenga {'por encima' if alcista else 'por debajo'} de ese nivel la señal está viva.")
        return base + " Conviene mirar el gráfico antes de operarla."

    # --- apertura, segun el nivel de conviccion -----------------------------
    if nivel == "ALTA":
        apertura = _variante([
            f"El giro {lado} de {activo} llega con los filtros alineados, que es la configuración "
            f"más confiable de este sistema.",
            f"Cruce {lado} limpio en {activo}: el Supertrend da vuelta y ningún indicador secundario lo contradice.",
            f"{activo} muestra confluencia completa en el giro {lado}, no es un cambio de color suelto.",
        ], semilla)
    elif nivel == "MEDIA":
        apertura = _variante([
            f"El Supertrend giró {movimiento} en {activo}, pero no todos los filtros acompañan: es una "
            f"señal para seguir de cerca, no para entrar de una.",
            f"Hay cruce {lado} en {activo} con sustento parcial, aunque el cuadro no termina de cerrar del todo.",
            f"{activo} da señal de {accion}, con los indicadores secundarios repartidos entre a favor y neutro.",
        ], semilla)
    else:
        apertura = _variante([
            f"Atención con {activo}: el Supertrend marca {accion}, pero el resto de los indicadores no lo respalda.",
            f"Señal de {accion} en {activo} de baja calidad: el cruce está, el respaldo técnico no.",
            f"{activo} cruzó {movimiento}, aunque la confluencia juega en contra y eso la vuelve poco confiable.",
        ], semilla)

    # --- el factor que mas suma y el que mas resta --------------------------
    ordenados = sorted(factores, key=lambda f: f.puntos)
    peor, mejor = ordenados[0], ordenados[-1]

    cuerpo = [apertura]
    if mejor.puntos > 0:
        cuerpo.append(mejor.comentario)
    if peor.puntos <= 0 and peor.clave != mejor.clave:
        conector = "Lo que hay que vigilar: " if peor.puntos < 0 else "El punto flojo: "
        cuerpo.append(conector + peor.comentario[0].lower() + peor.comentario[1:])

    # --- cierre operativo: la linea del Supertrend es la invalidacion -------
    if m.linea_st is not None and np.isfinite(m.linea_st):
        referencia = formatear_precio(m.linea_st)
        lado_precio = "por encima" if alcista else "por debajo"
        cuerpo.append(
            f"Mientras el precio se sostenga {lado_precio} de {referencia}, la señal sigue vigente; "
            f"un cierre de {TIMEFRAME} del otro lado la invalida."
        )
    if nivel == "BAJA":
        cuerpo.append("Con esta puntuación la tomaría como aviso para mirar el gráfico, no como entrada.")

    return " ".join(cuerpo)


# =============================================================================
#  6) TELEGRAM
# =============================================================================

_session = requests.Session()
_session.headers.update({"User-Agent": "bot-supertrend/1.0"})
_telegram_lock = threading.Lock()
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# Evento global de apagado. Se usa en lugar de time.sleep() en todas las esperas
# para que un SIGTERM de Replit no tenga que aguantar un backoff completo.
_parar = threading.Event()

# Telegram tolera aproximadamente 1 mensaje por segundo al mismo chat. Si el
# mercado gira en bloque (los 9 pares USDT en un derrumbe de BTC, o todos los
# indices en la misma vela) el bot dispararia 10-20 mensajes de golpe y se
# comeria un 429. Este throttle los espacia.
_ultimo_envio = 0.0
SEGUNDOS_ENTRE_MENSAJES = _env_float("SEGUNDOS_ENTRE_MENSAJES", 1.05)


def _largo_telegram(texto: str) -> int:
    """Telegram mide en unidades UTF-16: cada emoji fuera del BMP cuenta 2."""
    return len(texto.encode("utf-16-le")) // 2


def _recortar_para_telegram(texto: str, limite: int = 4000) -> str:
    """Recorta respetando el limite real y sin partir una etiqueta HTML al medio."""
    if _largo_telegram(texto) <= limite:
        return texto
    bajo, alto = 0, len(texto)
    while bajo < alto:
        medio = (bajo + alto + 1) // 2
        if _largo_telegram(texto[:medio]) <= limite:
            bajo = medio
        else:
            alto = medio - 1
    recortado = texto[:bajo]
    salto = recortado.rfind("\n")  # cortar en un fin de linea, nunca dentro de <b>
    if salto > 0:
        recortado = recortado[:salto]
    return recortado + "\n[...]"


def telegram_send(text: str, retries: int = 4) -> bool:
    """
    Envia un mensaje a Telegram. Devuelve True si Telegram confirmo la entrega.
    Nunca lanza excepcion: ante cualquier error loguea y devuelve False.
    """
    global _ultimo_envio

    if DRY_RUN:
        log.info("[DRY_RUN] Mensaje NO enviado:\n%s", text)
        return True

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.error("Faltan TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID")
        return False

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": _recortar_para_telegram(text),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    with _telegram_lock:
        # --- throttle: nunca dos mensajes seguidos en menos de ~1 segundo ----
        desde_el_ultimo = time.time() - _ultimo_envio
        if desde_el_ultimo < SEGUNDOS_ENTRE_MENSAJES:
            _parar.wait(SEGUNDOS_ENTRE_MENSAJES - desde_el_ultimo)

        esperas_flood = 0
        intento = 0
        while intento < retries:
            intento += 1
            if _parar.is_set():
                log.warning("Apagado en curso: abandono el envio a Telegram")
                return False
            try:
                r = _session.post(f"{TELEGRAM_API}/sendMessage", data=payload, timeout=20)
                _ultimo_envio = time.time()

                if r.status_code == 200:
                    return True

                # --- 429: flood wait. Telegram dice exactamente cuanto esperar;
                # reintentar antes de tiempo ALARGA el bloqueo. Se respeta el
                # valor completo y no se gasta un reintento en la espera.
                if r.status_code == 429:
                    try:
                        espera = int(r.json().get("parameters", {}).get("retry_after", 5))
                    except Exception:  # noqa: BLE001
                        espera = 5
                    esperas_flood += 1
                    if esperas_flood > 3:
                        log.error("Telegram sigue limitando tras 3 esperas. Abandono este mensaje.")
                        return False
                    log.warning("Telegram 429: espero los %ss que pide", espera)
                    intento -= 1
                    _parar.wait(min(espera + 1, 300))
                    continue

                # --- 400 por HTML mal formado: vale un reintento en texto plano
                # antes de dar la alerta por perdida.
                if r.status_code == 400 and payload.get("parse_mode"):
                    detalle = r.text.lower()
                    if "parse" in detalle or "too long" in detalle or "entities" in detalle:
                        log.warning("Telegram rechazo el formato HTML; reintento en texto plano")
                        payload.pop("parse_mode", None)
                        plano = text
                        for etiqueta in ("<b>", "</b>", "<i>", "</i>"):
                            plano = plano.replace(etiqueta, "")
                        payload["text"] = _recortar_para_telegram(plano)
                        continue

                # --- errores de configuracion: reintentar no sirve de nada
                if r.status_code in (400, 401, 403, 404):
                    log.error(
                        "Telegram rechazo el mensaje (HTTP %s): %s. "
                        "Revisa el token, el chat_id y que le hayas mandado /start al bot.",
                        r.status_code, r.text[:300],
                    )
                    return False

                log.warning("Telegram HTTP %s: %s", r.status_code, r.text[:200])

            except requests.RequestException as exc:
                log.warning("Error de red hacia Telegram (intento %s/%s): %s", intento, retries, exc)
            except Exception as exc:  # noqa: BLE001
                log.warning("Error inesperado enviando a Telegram: %s", exc)

            # backoff solo si todavia queda otro intento por delante
            if intento < retries:
                _parar.wait(min(2 ** intento, 30))

    log.error("No se pudo entregar el mensaje a Telegram tras %s intentos", retries)
    return False


def esc(texto: str) -> str:
    """Escapa el texto para parse_mode=HTML."""
    return html.escape(str(texto), quote=False)


# =============================================================================
#  7) PERSISTENCIA DEL ESTADO
# =============================================================================
#  { "BTC/USDT": {"dir": 1, "bar": 1712345678000, "fails": 0}, ... }
# =============================================================================

def load_state() -> dict:
    if not os.path.isfile(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudo leer %s (%s). Arranco con estado vacio.", STATE_FILE, exc)
        return {}


def save_state(state: dict) -> None:
    """Escritura atomica: primero .tmp, despues replace."""
    tmp = STATE_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, STATE_FILE)
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudo guardar el estado: %s", exc)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


# =============================================================================
#  8) UTILIDADES DE VELAS
# =============================================================================

def normalizar_ohlc(df: "pd.DataFrame") -> "pd.DataFrame":
    """Deja el DataFrame con columnas open/high/low/close en minuscula, sin NaN."""
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()

    # yfinance puede devolver columnas MultiIndex incluso con un solo ticker
    if isinstance(df.columns, pd.MultiIndex):
        for nivel in range(df.columns.nlevels):
            valores = df.columns.get_level_values(nivel)
            if any(str(v).lower() in ("open", "high", "low", "close") for v in valores):
                df.columns = valores
                break
        else:
            df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # nombres alternativos
    renombres = {"adj_close": "adj_close", "vol": "volume", "price": "close"}
    df = df.rename(columns=renombres)

    requeridas = ("open", "high", "low", "close")
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"faltan columnas {faltantes} (recibi {list(df.columns)})")

    df = df[[c for c in ("open", "high", "low", "close", "volume") if c in df.columns]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=list(requeridas))
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def indice_utc(df: "pd.DataFrame") -> "pd.DataFrame":
    """Fuerza el indice a DatetimeIndex en UTC."""
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df = df.copy()
    df.index = idx
    return df


def descartar_vela_en_formacion(df: "pd.DataFrame", tf_seconds: int = TIMEFRAME_SECONDS) -> "pd.DataFrame":
    """
    Elimina la ultima vela si todavia no cerro.

    Es EL punto critico del bot: el Supertrend sobre una vela abierta puede
    cambiar de direccion y volver atras varias veces dentro de la misma hora.
    Operando solo sobre velas cerradas, la senal es definitiva.
    """
    if df.empty:
        return df
    ahora = pd.Timestamp.now(tz="UTC")
    cierre_ultima = df.index[-1] + pd.Timedelta(seconds=tf_seconds)
    if cierre_ultima > ahora:
        return df.iloc[:-1]
    return df


# =============================================================================
#  9) FUENTE DE DATOS: CRIPTO (ccxt)
# =============================================================================

class ProveedorCripto:
    """Envuelve ccxt con fallback de exchange y resolucion de alias."""

    def __init__(self) -> None:
        self.exchange = None
        self.exchange_id = None
        self._mapa_simbolos: dict[str, str] = {}
        # exchanges que ya fallaron en este ciclo; se saltean al reconectar
        self._descartados: set[str] = set()

    def descartar_actual(self) -> None:
        """
        Marca el exchange vigente como inservible y fuerza la reconexion.

        Sin esto, conectar() volveria a empezar por EXCHANGE_ORDER[0] y se
        reconectaria al mismo exchange roto una vez por activo: 9 load_markets()
        por ciclo y el fallback a kucoin nunca llegaria a usarse.
        """
        if self.exchange_id:
            self._descartados.add(self.exchange_id)
            log.warning("Descarto %s por esta ronda y paso al siguiente exchange", self.exchange_id)
        self.exchange = None
        self.exchange_id = None
        self._mapa_simbolos = {}

    # -- conexion -------------------------------------------------------------
    def conectar(self, forzar: bool = False) -> bool:
        if self.exchange is not None and not forzar:
            return True

        forzado = _env("EXCHANGE_ID", "")
        base = [forzado] if forzado else EXCHANGE_ORDER
        orden = [n for n in base if n not in self._descartados]
        if not orden:
            # se agotaron todos: se limpia la lista negra y se vuelve a intentar
            # desde el principio en el proximo ciclo.
            log.warning("Todos los exchanges fallaron. Limpio la lista negra y reintento.")
            self._descartados.clear()
            orden = base

        for nombre in orden:
            if not hasattr(ccxt, nombre):
                log.warning("ccxt no conoce el exchange '%s', lo salteo", nombre)
                continue
            try:
                log.info("Conectando a %s ...", nombre)
                ex = getattr(ccxt, nombre)({
                    "enableRateLimit": True,
                    "timeout": 20000,
                    "options": {"defaultType": "spot"},
                })
                ex.load_markets()
                self.exchange = ex
                self.exchange_id = nombre
                self._mapa_simbolos = {}
                log.info("Conectado a %s (%s mercados)", nombre, len(ex.markets))
                return True
            except Exception as exc:  # noqa: BLE001
                texto = str(exc)
                if "451" in texto or "restricted location" in texto.lower():
                    log.warning(
                        "%s bloquea esta IP (HTTP 451 - restriccion geografica). Pruebo el siguiente.",
                        nombre,
                    )
                else:
                    log.warning("No pude conectar a %s: %s", nombre, texto[:200])

        log.error("Ningun exchange respondio. Reintento en el proximo ciclo.")
        return False

    # -- resolucion de simbolos ----------------------------------------------
    def resolver(self, simbolo: str) -> str | None:
        """Devuelve el simbolo real que existe en el exchange, o None."""
        if simbolo in self._mapa_simbolos:
            return self._mapa_simbolos[simbolo]

        if self.exchange is None:
            return None

        candidatos = [simbolo] + CRYPTO_ALIASES.get(simbolo, [])
        for cand in candidatos:
            mercado = self.exchange.markets.get(cand)
            # Se compara contra False explicito, no por truthiness: algunos
            # exchanges dejan 'active'/'spot' en None (desconocido) y el default
            # de dict.get no protege contra eso, porque la clave SI existe.
            if mercado and mercado.get("active") is not False and mercado.get("spot") is not False:
                if cand != simbolo:
                    log.info("%s no existe en %s -> uso %s", simbolo, self.exchange_id, cand)
                self._mapa_simbolos[simbolo] = cand
                return cand

        log.warning("%s no esta disponible en %s (ni sus alias)", simbolo, self.exchange_id)
        self._mapa_simbolos[simbolo] = ""  # cachear el fallo para no repetir el log
        return None

    # -- descarga de velas ----------------------------------------------------
    def velas(self, simbolo: str) -> "pd.DataFrame":
        if not self.conectar():
            raise RuntimeError("sin conexion a ningun exchange")

        real = self.resolver(simbolo)
        if not real:
            raise ValueError(f"simbolo no disponible en {self.exchange_id}")

        crudo = self.exchange.fetch_ohlcv(real, timeframe=TIMEFRAME, limit=CRYPTO_LIMIT)
        if not crudo:
            raise ValueError("el exchange devolvio 0 velas")

        df = pd.DataFrame(crudo, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts")
        df = normalizar_ohlc(df)
        df = indice_utc(df)
        return descartar_vela_en_formacion(df)


# =============================================================================
#  10) FUENTE DE DATOS: ACCIONES Y ETFs (yfinance)
# =============================================================================

def _extraer_ticker(bloque: "pd.DataFrame", ticker: str) -> "pd.DataFrame | None":
    """Saca el sub-DataFrame de un ticker desde una descarga agrupada."""
    if bloque is None or bloque.empty:
        return None
    if not isinstance(bloque.columns, pd.MultiIndex):
        sub = bloque.dropna(how="all")
        return sub if not sub.empty else None
    for nivel in range(bloque.columns.nlevels):
        if ticker in bloque.columns.get_level_values(nivel):
            sub = bloque.xs(ticker, axis=1, level=nivel, drop_level=True)
            # Cuando un ticker del lote falla, yfinance NO lo omite: lo devuelve
            # con todas las filas de los tickers buenos y los valores en NaN.
            # Ese frame no es .empty, asi que hay que mirar el contenido real.
            sub = sub.dropna(how="all")
            return sub if not sub.empty else None
    return None


def descargar_acciones(tickers: list[str]) -> dict[str, "pd.DataFrame"]:
    """
    Descarga velas de 1h para varios tickers.

    Estrategia: descarga en lotes (una sola request por lote, mucho mas amable
    con el rate limit de Yahoo que 25 pedidos sueltos). Si un lote falla, cae a
    descarga individual ticker por ticker.
    """
    resultado: dict[str, pd.DataFrame] = {}

    for inicio in range(0, len(tickers), STOCK_BATCH_SIZE):
        lote = tickers[inicio:inicio + STOCK_BATCH_SIZE]
        bloque = None

        for intento in range(1, 4):
            try:
                bloque = yf.download(
                    tickers=lote,
                    period=STOCK_PERIOD,
                    interval=TIMEFRAME,
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    threads=False,
                    group_by="ticker",
                )
                # yf.download NO propaga excepciones: atrapa internamente los
                # 429, los cortes de red y los tickers invalidos, y devuelve un
                # DataFrame vacio. Sin este raise, el except de abajo seria
                # codigo muerto y no habria ni reintento ni backoff.
                if bloque is None or bloque.empty:
                    raise RuntimeError("Yahoo devolvio un bloque vacio")
                break
            except Exception as exc:  # noqa: BLE001
                bloque = None
                if intento == 3:
                    log.error("Lote %s sin datos tras 3 intentos: %s", lote, str(exc)[:150])
                    break
                espera = 3 * intento
                log.warning(
                    "Fallo la descarga del lote %s (intento %s/3): %s. Reintento en %ss",
                    lote, intento, str(exc)[:150], espera,
                )
                _parar.wait(espera)

        # Si se cayo el lote ENTERO es limite de Yahoo o un problema de la
        # fuente, no tickers invalidos: pedirlos de a uno solo agrava el 429.
        lote_caido = bloque is None

        for ticker in lote:
            if _parar.is_set():
                return resultado
            try:
                sub = _extraer_ticker(bloque, ticker) if bloque is not None else None
                df = normalizar_ohlc(sub)

                # El fallback se decide sobre filas utilizables, no sobre .empty:
                # un ticker fallido dentro de un lote bueno vuelve lleno de NaN.
                if df.empty and not lote_caido:
                    alterno = _descarga_individual(ticker)
                    df = normalizar_ohlc(alterno)
                    _parar.wait(1.0)

                if df.empty:
                    log.warning(
                        "%s: sin velas utilizables%s", ticker,
                        " (se cayo el lote entero, probable limite de Yahoo)" if lote_caido
                        else " (mercado cerrado o ticker invalido)",
                    )
                    continue

                df = indice_utc(df)
                df = descartar_vela_en_formacion(df)
                if not df.empty:
                    resultado[ticker] = df
            except Exception as exc:  # noqa: BLE001
                log.warning("%s: error procesando datos -> %s", ticker, str(exc)[:150])

        _parar.wait(1.0)  # respiro entre lotes para no comerse un 429 de Yahoo

    return resultado


def _descarga_individual(ticker: str) -> "pd.DataFrame | None":
    """Plan B: pedir un solo ticker."""
    try:
        df = yf.Ticker(ticker).history(
            period=STOCK_PERIOD,
            interval=TIMEFRAME,
            auto_adjust=False,
            actions=False,
        )
        return df if df is not None and not df.empty else None
    except Exception as exc:  # noqa: BLE001
        log.warning("%s: descarga individual fallida -> %s", ticker, str(exc)[:150])
        return None


# =============================================================================
#  11) LOGICA DE ALERTAS
# =============================================================================

def formatear_precio(valor: float) -> str:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "-"
    if valor >= 1000:
        return f"{valor:,.2f}"
    if valor >= 1:
        return f"{valor:,.4f}".rstrip("0").rstrip(".")
    return f"{valor:.8f}".rstrip("0").rstrip(".")


def _bloque_gestion_capital(tamano: "TamanoPosicion | None", motivo_sin_sizing: str) -> list[str]:
    """Lineas del bloque '💰 Gestion de Capital', o [] si no corresponde."""
    if tamano is None:
        if not motivo_sin_sizing:  # "" = venta o conviccion baja: ni se menciona
            return []
        return ["", "💰 <b>Gestión de Capital:</b>",
                f"No se calculó tamaño de posición: {esc(motivo_sin_sizing)}."]

    lineas = [
        "", "💰 <b>Gestión de Capital (Position Sizing):</b>",
        f"• Capital considerado: ${formatear_precio(tamano.capital)}",
        f"• Riesgo objetivo: {tamano.riesgo_pct*100:g}% (${formatear_precio(tamano.riesgo_dinero)})",
        f"• Stop Loss ({'línea Supertrend' if tamano.sl_modo == 'supertrend' else f'ATR x{ATR_SL_MULT:g}'}): "
        f"${formatear_precio(tamano.precio_sl)} "
        f"(-{tamano.distancia_sl_pct:.2f}% desde la entrada)",
        f"• Cantidad sugerida: {formatear_precio(tamano.unidades)} unidades",
        f"• Monto a invertir: <b>${formatear_precio(tamano.monto_invertir)}</b>",
    ]
    if tamano.adx is not None:
        lineas.append(f"• ADX({ADX_LENGTH}): {tamano.adx:.1f} (tendencia {'fuerte' if tamano.adx >= ADX_MIN*1.4 else 'confirmada'})")

    if tamano.limitado_por:
        motivo = ("tu capital disponible" if tamano.limitado_por == "capital"
                 else f"el tope de exposición por operación ({MAX_EXPOSICION_PCT*100:g}% del capital)")
        lineas.append(
            f"• ⚠️ El riesgo puro pedía ${formatear_precio(tamano.monto_sin_limite)}; se limitó por {motivo}. "
            f"Riesgo real si toca el stop: {tamano.riesgo_real_pct*100:.2f}% (no el {tamano.riesgo_pct*100:g}% objetivo)."
        )
    lineas.append(
        "• <i>Cálculo para esta señal sola. Si ya tenés otras posiciones abiertas, "
        "restá su valor de tu capital antes de usar este monto.</i>"
    )
    return lineas


def construir_alerta(activo: str, direccion: int, m: Metricas, nivel: str, puntaje: int,
                     factores: list[Factor], cierre_vela, mercado: str, bar_id: int,
                     metricas_ok: bool = True, tamano: "TamanoPosicion | None" = None,
                     motivo_sin_sizing: str = "") -> str:
    """
    Reporte de oportunidad con el formato JARVIS.
    Con ALERT_DETAILS=0 se reduce a la cabecera y el nivel de conviccion.
    """
    accion = "COMPRA" if direccion == 1 else "VENTA"
    emoji = "🟢" if direccion == 1 else "🔴"

    if not ALERT_DETAILS:
        sufijo = f" — invertir ${formatear_precio(tamano.monto_invertir)}" if tamano else ""
        return (f"{emoji} <b>{accion}: {esc(activo)}</b> — conviccion {nivel} "
                f"@ {formatear_precio(m.precio)}{sufijo}")

    por_clave = {f.clave: f for f in factores}
    hora_local = cierre_vela.tz_convert(LOCAL_TZ).strftime("%d/%m/%Y %H:%M")
    estado_st = "alcista" if direccion == 1 else "bajista"

    partes = [
        "🧠 <b>JARVIS TRADING ASSISTANT — ANÁLISIS DE OPORTUNIDAD</b>",
        f"{emoji} <b>ACCIÓN SUGERIDA: {accion}</b>",
        f"📌 <b>Activo:</b> {esc(activo)}",
        f"💲 <b>Precio Actual:</b> ${formatear_precio(m.precio)}",
        f"⭐️ <b>Nivel de Convicción:</b> {nivel} ({puntaje:+d} de 6)",
        "",
        "📊 <b>Desglose Técnico:</b>",
        f"• <b>Supertrend:</b> cruce {estado_st} confirmado en el cierre "
        f"(línea en {formatear_precio(m.linea_st)})",
        f"• <b>RSI ({RSI_LENGTH}):</b> {esc(por_clave['RSI'].resumen)}",
        f"• <b>Volumen:</b> {esc(por_clave['Volumen'].resumen)}",
        f"• <b>Tendencia (EMA):</b> {esc(por_clave['EMA'].resumen)}",
        "",
        "💡 <b>Análisis del Operador:</b>",
        f"<i>{esc(redactar_analisis(activo, direccion, m, nivel, factores, bar_id, metricas_ok))}</i>",
    ]

    if not metricas_ok:
        partes += ["", "⚠️ <b>Sin filtros de confluencia:</b> no se pudieron calcular RSI, volumen ni EMAs "
                       "para esta vela. El cruce del Supertrend es válido; el resto quedó sin verificar."]
    elif nivel == "BAJA":
        partes += ["", "⚠️ <b>Señal de baja convicción:</b> se informa por transparencia, "
                       "pero la confluencia no la respalda."]

    partes += _bloque_gestion_capital(tamano, motivo_sin_sizing)

    partes += [
        "",
        f"<i>{esc(mercado)} · vela de {esc(TIMEFRAME)} cerrada {hora_local} ({esc(LOCAL_TZ_NAME)})</i>",
    ]
    return "\n".join(partes)


def evaluar_activo(activo: str, df: "pd.DataFrame", estado: dict, mercado: str) -> bool:
    """
    Calcula el Supertrend, compara con el estado previo y alerta si hubo giro.
    Devuelve True si se envio una alerta.
    """
    if len(df) < MIN_BARS:
        log.warning("%s: solo %s velas cerradas (necesito %s). Salteo.", activo, len(df), MIN_BARS)
        return False

    st = supertrend(df)
    serie_dir = st["direction"].dropna()
    if len(serie_dir) < 2:
        log.warning("%s: el Supertrend no tiene suficientes valores validos", activo)
        return False

    dir_actual = int(serie_dir.iloc[-1])
    cierre_vela = serie_dir.index[-1]
    bar_id = int(cierre_vela.timestamp() * 1000)
    precio = float(df["close"].iloc[-1])
    linea_st = float(st["line"].iloc[-1]) if not pd.isna(st["line"].iloc[-1]) else float("nan")

    previo = estado.get(activo) or {}
    dir_guardada = previo.get("dir")
    bar_guardada = previo.get("bar")
    fallos = int(previo.get("fails", 0))

    # --- Primera vez que vemos este activo: sembramos sin alertar ------------
    if dir_guardada is None:
        estado[activo] = {"dir": dir_actual, "bar": bar_id, "fails": 0}
        log.info(
            "%s: estado inicial sembrado (%s) @ %s",
            activo, "ALCISTA" if dir_actual == 1 else "BAJISTA", formatear_precio(precio),
        )
        return False

    # --- Vela ya evaluada: idempotencia por vela ----------------------------
    # La direccion NO forma parte de la condicion a proposito. Si la fuente
    # revisa una vela ya cerrada (yfinance consolida la ultima barra de la
    # sesion, los exchanges tambien ajustan) y el indicador da otra direccion
    # sobre la misma vela, incluirla dejaria pasar una segunda alerta
    # contradictoria sobre esa misma vela.
    # El reintento por fallo de envio no queda bloqueado: esa rama guarda la
    # vela VIEJA justamente para poder volver a entrar aca.
    if bar_guardada is not None and bar_id == bar_guardada:
        return False

    # --- Sin cambio de direccion: solo actualizamos el puntero de vela -------
    if dir_actual == int(dir_guardada):
        estado[activo] = {"dir": dir_actual, "bar": bar_id, "fails": 0}
        return False

    # --- HUBO GIRO: se evalua la confluencia antes de avisar ----------------
    metricas_ok = True
    try:
        metricas = calcular_metricas(df, st)
    except Exception as exc:  # noqa: BLE001
        # Si algo falla al calcular las metricas secundarias NO se pierde la
        # senal: se sigue con lo que hay y los filtros quedan como "sin dato".
        log.warning("%s: no se pudieron calcular las metricas de confluencia (%s)", activo, str(exc)[:150])
        metricas = Metricas(precio=precio, direccion=dir_actual, linea_st=linea_st)
        metricas_ok = False

    nivel, puntaje, factores = evaluar_conviccion(metricas, dir_actual)

    log.info(
        "%s: GIRO %s -> %s | conviccion %s (%+d) | RSI %s | vol %s | EMA %s | ADX %s",
        activo, dir_guardada, dir_actual, nivel, puntaje,
        f"{metricas.rsi:.1f}" if metricas.rsi is not None else "-",
        f"x{metricas.volumen_ratio:.2f}" if metricas.volumen_ratio is not None else "-",
        f"{factores[2].puntos:+d}",
        f"{metricas.adx:.1f}" if metricas.adx is not None else "-",
    )

    # --- Filtro por nivel minimo -------------------------------------------
    # Si las metricas no se pudieron calcular, TODOS los factores valen 0 y el
    # nivel sale BAJA por falta de datos, no por confluencia en contra. Filtrar
    # ahi perderia la senal en silencio, que es lo contrario de lo que promete
    # el except de arriba: por eso el filtro solo se aplica a senales medidas.
    if metricas_ok and _por_debajo_del_minimo(nivel):
        log.info("%s: conviccion %s por debajo de MIN_CONVICCION=%s. No se avisa.",
                 activo, nivel, MIN_CONVICCION)
        # El cambio se confirma igual: el giro ocurrio y ya fue evaluado. Si no
        # se confirmara, la misma senal vieja volveria a evaluarse vela tras
        # vela y podria dispararse tarde, cuando ya no tiene sentido operarla.
        estado[activo] = {"dir": dir_actual, "bar": bar_id, "fails": 0}
        save_state(estado)
        return False

    # --- Gestion de capital: solo para compras que pasan puntaje y ADX ------
    tamano = None
    motivo_sin_sizing = _motivo_sin_sizing(dir_actual, nivel, metricas.adx if metricas_ok else None) \
        if metricas_ok else ""
    if motivo_sin_sizing is None:
        tamano = calcular_tamano_posicion(metricas, _leer_capital())
        if tamano is None:
            motivo_sin_sizing = "no se pudo calcular un Stop Loss válido para esta vela"

    mensaje = construir_alerta(activo, dir_actual, metricas, nivel, puntaje,
                               factores, cierre_vela, mercado, bar_id, metricas_ok,
                               tamano, motivo_sin_sizing)
    entregado = telegram_send(mensaje)

    if entregado:
        estado[activo] = {"dir": dir_actual, "bar": bar_id, "fails": 0}
        # Se persiste EN EL ACTO, no al final del ciclo. Un escaneo completo
        # tarda entre 20 s y varios minutos; si Replit recicla el contenedor en
        # esa ventana, el estado en disco todavia diria la direccion vieja y el
        # bot volveria a mandar una alerta ya entregada.
        save_state(estado)
        return True

    # No se entrego: NO confirmamos el cambio, asi el proximo ciclo reintenta.
    fallos += 1
    if fallos >= MAX_SEND_RETRIES_BEFORE_COMMIT:
        log.error(
            "%s: %s ciclos sin poder avisar. Confirmo el cambio igual para no quedar trabado.",
            activo, fallos,
        )
        estado[activo] = {"dir": dir_actual, "bar": bar_id, "fails": 0}
    else:
        estado[activo] = {"dir": dir_guardada, "bar": bar_guardada, "fails": fallos}
        log.warning("%s: alerta no entregada, reintento en el proximo ciclo (%s)", activo, fallos)
    return False


# =============================================================================
#  12) CICLO DE ESCANEO
# =============================================================================

_cripto = ProveedorCripto()
# _parar (evento de apagado) se define arriba, junto a la seccion de Telegram,
# porque telegram_send ya lo necesita para sus esperas.


def escanear_cripto(estado: dict) -> tuple[int, int, int]:
    """Devuelve (ok, errores, alertas)."""
    ok = errores = alertas = 0
    log.info("--- Criptomonedas (%s) ---", _cripto.exchange_id or "conectando")

    for simbolo in CRYPTO_SYMBOLS:
        if _parar.is_set():
            break
        try:
            df = _cripto.velas(simbolo)
            if evaluar_activo(simbolo, df, estado, mercado=f"Cripto / {_cripto.exchange_id}"):
                alertas += 1
            ok += 1
        except Exception as exc:  # noqa: BLE001
            errores += 1
            log.warning("%s: %s", simbolo, str(exc)[:200])
            # Si el fallo es del exchange (y no de un simbolo puntual), lo
            # descartamos y seguimos con el siguiente de la lista.
            # Se chequea por TIPO y no por substring: el texto de
            # ccxt.RequestTimeout dice "timed out", no "timeout", asi que el
            # match por texto nunca lo detectaba; y un '451' dentro de un precio
            # ("2451.30") disparaba una reconexion falsa.
            es_fallo_de_red = isinstance(exc, ccxt.NetworkError)
            es_bloqueo_geo = isinstance(exc, ccxt.ExchangeError) and "451" in str(exc)
            if es_fallo_de_red or es_bloqueo_geo:
                _cripto.descartar_actual()

    return ok, errores, alertas


def escanear_acciones(estado: dict) -> tuple[int, int, int]:
    """Devuelve (ok, errores, alertas)."""
    ok = errores = alertas = 0
    log.info("--- Acciones y ETFs de EE.UU. ---")

    try:
        datos = descargar_acciones(STOCK_SYMBOLS)
    except Exception as exc:  # noqa: BLE001
        log.error("Fallo general de yfinance: %s", str(exc)[:200])
        return 0, len(STOCK_SYMBOLS), 0

    for ticker in STOCK_SYMBOLS:
        if _parar.is_set():
            break
        df = datos.get(ticker)
        if df is None or df.empty:
            errores += 1
            continue
        try:
            if evaluar_activo(ticker, df, estado, mercado="Acciones EE.UU. / NASDAQ-NYSE"):
                alertas += 1
            ok += 1
        except Exception as exc:  # noqa: BLE001
            errores += 1
            log.warning("%s: %s", ticker, str(exc)[:200])

    return ok, errores, alertas


def ciclo(estado: dict) -> None:
    inicio = time.time()
    log.info("=" * 62)
    log.info("Escaneo iniciado  |  %s", now_local().strftime("%d/%m/%Y %H:%M:%S"))

    ok_c, err_c, alert_c = escanear_cripto(estado)
    ok_a, err_a, alert_a = escanear_acciones(estado)

    save_state(estado)

    log.info(
        "Escaneo terminado en %.1fs  |  OK: %s  |  fallos: %s  |  alertas: %s",
        time.time() - inicio, ok_c + ok_a, err_c + err_a, alert_c + alert_a,
    )


def dormir_hasta_proximo_ciclo() -> None:
    """
    Espera hasta el proximo multiplo de SCAN_MINUTES, con un colchon de 20s
    para darle tiempo al exchange a publicar la vela recien cerrada.
    Se interrumpe si llega SIGINT/SIGTERM.
    """
    paso = max(1, SCAN_MINUTES) * 60
    ahora = time.time()
    objetivo = (int(ahora // paso) + 1) * paso + 20
    espera = max(5.0, objetivo - ahora)
    log.info("Proximo escaneo en %s (%.0f s)\n", (now_local() + timedelta(seconds=espera)).strftime("%H:%M:%S"), espera)
    _parar.wait(espera)


# =============================================================================
#  13) EXTRAS DE ENTORNO (Replit)
# =============================================================================

def arrancar_keepalive() -> None:
    """
    Mini servidor HTTP para que un pinger externo (UptimeRobot) mantenga
    despierto el Repl. Solo necesario en planes sin 'Always On' / Reserved VM.
    """
    if not KEEPALIVE:
        return
    try:
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class Ping(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"bot-supertrend OK")

            def log_message(self, *args):  # silenciar el log de cada ping
                return

        servidor = HTTPServer(("0.0.0.0", KEEPALIVE_PORT), Ping)
        threading.Thread(target=servidor.serve_forever, daemon=True).start()
        log.info("Keep-alive HTTP escuchando en el puerto %s", KEEPALIVE_PORT)
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudo levantar el keep-alive: %s", exc)


def manejar_senal(signum, frame):  # noqa: ARG001
    log.info("Senal %s recibida: cerrando ordenadamente ...", signum)
    _parar.set()


# =============================================================================
#  14) MAIN
# =============================================================================

def mensaje_arranque() -> str:
    return "\n".join([
        "🧠 <b>JARVIS TRADING ASSISTANT activo</b>",
        "",
        f"Señal: Supertrend({ST_LENGTH}, {ST_MULTIPLIER:g}) en {esc(TIMEFRAME)}",
        f"Confluencia: RSI({RSI_LENGTH}) · Volumen vs MA{VOL_MA_LENGTH} · EMA{EMA_RAPIDA}/{EMA_LENTA} · ADX({ADX_LENGTH})",
        f"Motor de calculo: {esc(ENGINE)}",
        f"Cripto: {len(CRYPTO_SYMBOLS)} pares ({esc(_cripto.exchange_id or 'sin conexion')})",
        f"Acciones/ETFs: {len(STOCK_SYMBOLS)} tickers (Yahoo Finance)",
        f"Escaneo cada {SCAN_MINUTES} min sobre velas cerradas",
        f"Convicción mínima para avisar: {esc(MIN_CONVICCION.upper())}",
        f"Gestión de capital: ${formatear_precio(_leer_capital())} · riesgo {RIESGO_PCT*100:g}% · "
        f"tope {MAX_EXPOSICION_PCT*100:g}% por operación · SL por {esc(SL_MODO)} · "
        f"requiere {esc(SIZING_MIN_CONVICCION.upper())} + ADX≥{ADX_MIN:g}",
        "",
        f"<i>{now_local().strftime('%d/%m/%Y %H:%M')} ({esc(LOCAL_TZ_NAME)})</i>",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Bot Supertrend -> Telegram")
    parser.add_argument("--once", action="store_true", help="ejecuta un solo ciclo y sale")
    parser.add_argument("--test-telegram", action="store_true", help="manda un mensaje de prueba y sale")
    parser.add_argument("--reset-state", action="store_true", help="borra el estado guardado y sale")
    args = parser.parse_args()

    if args.reset_state:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
            print(f"Estado borrado: {STATE_FILE}")
        else:
            print("No habia estado guardado.")
        return 0

    if args.test_telegram:
        ok = telegram_send("✅ Prueba de conexion del bot Supertrend. Si lees esto, Telegram esta OK.")
        print("Enviado correctamente." if ok else "FALLO el envio. Revisa token y chat_id.")
        return 0 if ok else 1

    signal.signal(signal.SIGINT, manejar_senal)
    try:
        signal.signal(signal.SIGTERM, manejar_senal)
    except (AttributeError, ValueError):
        pass  # Windows / hilos secundarios

    log.info("=" * 62)
    log.info("JARVIS TRADING ASSISTANT -> TELEGRAM")
    log.info("Supertrend(%s, %g) | timeframe %s | escaneo cada %s min",
             ST_LENGTH, ST_MULTIPLIER, TIMEFRAME, SCAN_MINUTES)
    log.info("Confluencia: RSI(%s) | Volumen vs MA%s | EMA%s/EMA%s | ADX(%s)",
             RSI_LENGTH, VOL_MA_LENGTH, EMA_RAPIDA, EMA_LENTA, ADX_LENGTH)
    log.info("Conviccion minima para avisar: %s", MIN_CONVICCION.upper())
    log.info(
        "Gestion de capital: $%s | riesgo %g%% | tope %g%% por operacion | SL por %s | "
        "sizing requiere conviccion >= %s y ADX >= %g",
        formatear_precio(_leer_capital()), RIESGO_PCT * 100, MAX_EXPOSICION_PCT * 100,
        SL_MODO, SIZING_MIN_CONVICCION.upper(), ADX_MIN,
    )
    log.info("Motor de calculo: %s%s", ENGINE,
             "" if PANDAS_TA is not None else "  (pandas_ta no disponible)")
    log.info("Activos: %s cripto + %s acciones/ETFs", len(CRYPTO_SYMBOLS), len(STOCK_SYMBOLS))
    log.info("Estado: %s", STATE_FILE)
    if DRY_RUN:
        log.warning("DRY_RUN activo: NO se enviara nada a Telegram")
    log.info("=" * 62)

    arrancar_keepalive()
    _cripto.conectar()

    estado = load_state()
    if estado:
        log.info("Estado previo cargado: %s activos ya monitoreados", len(estado))
    else:
        log.info("Sin estado previo: el primer ciclo solo siembra direcciones (sin alertas)")

    if SEND_STARTUP:
        telegram_send(mensaje_arranque())

    ultimo_heartbeat = time.time()

    while not _parar.is_set():
        try:
            ciclo(estado)
        except Exception as exc:  # noqa: BLE001
            # Red de seguridad final: pase lo que pase, el bot no se detiene.
            log.exception("Error no controlado en el ciclo: %s", exc)
            try:
                save_state(estado)
            except Exception:  # noqa: BLE001
                pass

        if HEARTBEAT_HOURS > 0 and (time.time() - ultimo_heartbeat) >= HEARTBEAT_HOURS * 3600:
            alcistas = sum(1 for v in estado.values() if v.get("dir") == 1)
            telegram_send(
                f"💓 Bot operativo. {len(estado)} activos monitoreados "
                f"({alcistas} en tendencia alcista, {len(estado) - alcistas} bajista)."
            )
            ultimo_heartbeat = time.time()

        if args.once:
            log.info("Modo --once: salgo despues del primer ciclo.")
            break

        dormir_hasta_proximo_ciclo()

    save_state(estado)
    log.info("Bot detenido.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
        sys.exit(0)

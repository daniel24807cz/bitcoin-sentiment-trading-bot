"""
Algotradingový bot pro obchodování BTC/USDC na Binance (Margin / Spot kompatibilní s EU MiCA)
na základě sentiment analýzy zpráv (Bitcoin RSS kanály + CoinMarketCap Free API tržní kontext)
pomocí modelu Google Gemini.
Obsahuje ochranný Stop-Loss (2 %), automatický Take-Profit (2 %) a živý monitoring otevřených pozic a PnL.
"""

import os
import re
import time
import logging
import warnings
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple

import ccxt
import requests

warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai

# Nastavení logování
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Konstanty bota (BTC/USDC je plně povolený pár pro EU/ČR účty na Binance dle MiCA)
SYMBOL = "BTC/USDC"
STOP_LOSS_PCT = 0.013           # 1.3 % ochranný Stop-Loss proti směru obchodu (menší ztráta při špatném vstupu)
TAKE_PROFIT_PCT = 0.02          # 2 % Take-Profit ve směru obchodu (~8 % zisk na kolaterál při páce 4x)
TRAILING_ACTIVATION_PCT = 0.008 # Po dosažení +0.8 % zisku se SL posune do zisku (pokryje i poplatky)
PROFIT_LOCK_PCT = 0.003         # Minimální zamčený zisk (+0.3 % pokrývá ~0.2 % poplatky burzy + čistý zisk)
TRAILING_DISTANCE_PCT = 0.005   # Posuvný Stop-Loss drží odstup 0.5 % od nejlepší dosažené ceny
SL_UPDATE_MIN_STEP_PCT = 0.002  # Příkaz na Binance přepíšeme, až když se SL zlepší alespoň o 0.2 %
MIN_SENTIMENT_SCORE = 7         # Minimální síla zprávy (1-10) pro otevření obchodu
RSI_PERIOD = 14                 # Perioda pro technický indikátor RSI
RSI_TIMEFRAME = "15m"           # Časový rámec svíček z Binance pro RSI filtr
RSI_OVERBOUGHT = 65.0           # Nad 65 neotvíráme LONG (standardní signál 7-8/10)
RSI_OVERSOLD = 38.0             # Pod 38 neotvíráme SHORT (standardní signál 7-8/10)
STRONG_SIGNAL_SCORE = 9         # Od skóre 9-10/10 povolíme širší RSI pásmo
RSI_STRONG_OVERBOUGHT = 72.0    # Nad 72 neotvíráme LONG ani u silných zpráv (9-10/10)
RSI_STRONG_OVERSOLD = 28.0      # Pod 28 neotvíráme SHORT ani u silných zpráv (9-10/10)
EMA_PERIOD = 50                 # Perioda pro trendový filtr EMA 50
EMA_TIMEFRAME = "1h"            # Časový rámec svíček z Binance pro hlavní trend (1h EMA 50)
TRADE_INVESTMENT_USDT = 21.0    # Výchozí velikost investice, pokud nelze načíst zůstatek
TRADE_CAPITAL_RATIO = 0.90      # Dynamicky použijeme 90 % volných USDC (efekt složeného úročení)
MIN_TRADE_USDC = 10.0           # Minimální volný zůstatek pro dynamický výpočet investice
POLL_INTERVAL_SECONDS = 60      # Interval kontroly pozic a zpráv (60 sekund)
CMC_CACHE_TTL_SECONDS = 300     # Cache pro CoinMarketCap API (5 minut = max 288 kreditů/den)

_cmc_cache: Dict[str, Any] = {"timestamp": 0.0, "context": ""}
_position_state: Dict[str, Any] = {
    "short_entry_price": None,
    "short_amount": None,
    "short_best_price": None,
    "short_trailing_sl": None,
    "long_entry_price": None,
    "long_amount": None,
    "long_best_price": None,
    "long_trailing_sl": None,
    "idle_since": None,
    "last_article_title": None,
    "last_sentiment": None,
}

# Zdroje zpráv a tržních dat (100% zdarma a bez blokací pro cloudové IP)
BTC_RSS_FEEDS = [
    "https://cointelegraph.com/rss/tag/bitcoin",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://decrypt.co/feed",
]
CMC_QUOTES_API_URL = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest"

FALLBACK_GEMINI_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-flash-latest",
)

SENTIMENT_PROMPT = (
    "Analyzuj tuto zprávu o Bitcoinu a odpověz striktně pouze jedním slovem "
    "s ohledem na vliv na jeho cenu: POSITIVE, NEGATIVE, nebo NEUTRAL."
)

SENTIMENT_SCORE_PROMPT = (
    "Analyzuj tuto zprávu o Bitcoinu z pohledu reálného dopadu na jeho cenu a odpověz "
    "přesně ve formátu 'SENTIMENT SKORE' (např. 'POSITIVE 8', 'NEGATIVE 7', 'NEUTRAL 3'). "
    "SENTIMENT je POSITIVE, NEGATIVE nebo NEUTRAL; SKORE je celé číslo 1 až 10 "
    "(1 = obecná/bezvýznamná zpráva, 10 = zásadní tržní katalyzátor)."
)


def load_api_keys() -> Dict[str, str]:
    """
    Bezpečně načte všechny potřebné API klíče z proměnných prostředí (os.environ).
    Pro CoinMarketCap načítá CMC_API_KEY (případně COINMARKETCAP_API_KEY nebo NEWS_API_KEY).
    Vyhodí EnvironmentError, pokud některý klíč chybí.
    """
    keys: Dict[str, str] = {}
    missing_keys = []

    for key in ("BINANCE_API_KEY", "BINANCE_SECRET", "GEMINI_API_KEY"):
        val = os.environ.get(key)
        if not val:
            missing_keys.append(key)
        else:
            keys[key] = val

    cmc_key = (
        os.environ.get("CMC_API_KEY")
        or os.environ.get("COINMARKETCAP_API_KEY")
        or os.environ.get("NEWS_API_KEY")
    )
    if not cmc_key:
        missing_keys.append("CMC_API_KEY")
    else:
        keys["CMC_API_KEY"] = cmc_key

    if missing_keys:
        raise EnvironmentError(
            f"Chybí povinné proměnné prostředí: {', '.join(missing_keys)}"
        )

    return keys


def create_exchange(api_key: str, api_secret: str) -> ccxt.binance:
    """
    Inicializuje a nakonfiguruje připojení k burze Binance.
    Výchozí režim je 'margin' (umožňuje LONG i SHORT na EU účtech s párem BTC/USDC).
    """
    exchange = ccxt.binance(
        {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        }
    )
    exchange.options["defaultType"] = "margin"
    return exchange


def get_margin_btc_borrowed_and_free(balance: Dict[str, Any]) -> Dict[str, float]:
    """
    Vyčte z odpovědi Binance Cross Margin přesný stav půjčených (borrowed) a držených (free + locked) BTC a volných USDC.
    """
    usdc_free = float(balance.get("USDC", {}).get("free", 0.0) or 0.0)
    btc_obj = balance.get("BTC", {}) if isinstance(balance.get("BTC"), dict) else {}
    btc_free = float(btc_obj.get("total", btc_obj.get("free", 0.0)) or 0.0)
    btc_borrowed = 0.0

    user_assets = balance.get("info", {}).get("userAssets", [])
    if isinstance(user_assets, list):
        for asset in user_assets:
            if asset.get("asset") == "BTC":
                btc_borrowed = float(asset.get("borrowed", 0.0) or 0.0)
                btc_free = float(asset.get("free", 0.0) or 0.0) + float(asset.get("locked", 0.0) or 0.0)
            elif asset.get("asset") == "USDC":
                usdc_free = float(asset.get("free", usdc_free) or 0.0)

    return {
        "usdc_free": usdc_free,
        "btc_free": btc_free,
        "btc_borrowed": btc_borrowed,
    }


def verify_binance_account(exchange: ccxt.binance) -> Dict[str, float]:
    """
    Ověří připojení k Binance a vypíše aktuální stav Cross Margin / Spot účtu.
    """
    try:
        balance = exchange.fetch_balance({"type": "margin"})
        stats = get_margin_btc_borrowed_and_free(balance)
        exchange.options["defaultType"] = "margin"
        logger.info(
            "Připojení k Binance (režim MARGIN - LONG i SHORT) je aktivní! "
            "Volné USDC: %.2f USDC | Volné BTC: %.6f BTC | Půjčené BTC (SHORT): %.6f BTC.",
            stats["usdc_free"],
            stats["btc_free"],
            stats["btc_borrowed"],
        )
        return stats
    except Exception:
        try:
            exchange.options["defaultType"] = "spot"
            balance = exchange.fetch_balance({"type": "spot"})
            usdc_free = float(balance.get("USDC", {}).get("free", 0.0) or 0.0)
            btc_free = float(balance.get("BTC", {}).get("free", 0.0) or 0.0)
            logger.info(
                "Připojení k Binance (režim SPOT) je plně funkční! "
                "Dostupný zůstatek ve Spot peněžence: %.2f USDC, %.6f BTC.",
                usdc_free,
                btc_free,
            )
            return {"usdc_free": usdc_free, "btc_free": btc_free, "btc_borrowed": 0.0}
        except Exception as spot_exc:
            logger.warning("Nelze načíst zůstatek z Binance: %s", spot_exc)
            return {"usdc_free": 0.0, "btc_free": 0.0, "btc_borrowed": 0.0}


def configure_gemini(gemini_api_key: str) -> genai.GenerativeModel:
    """
    Nakonfiguruje Google Generative AI SDK a vrátí instanci modelu Gemini.
    """
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    return model


def strip_html_tags(text: str) -> str:
    """Odstraní případné HTML značky z RSS popisu článku."""
    if not text:
        return ""
    return re.sub(r"<[^>]+>", " ", text).strip()


def fetch_cmc_market_context(cmc_api_key: str) -> str:
    """
    Stáhne aktuální tržní statistiky Bitcoinu z bezplatného CoinMarketCap endpointu
    (/v2/cryptocurrency/quotes/latest) s 5minutovou cache a zformátuje je jako kontext ke zprávě.
    """
    if not cmc_api_key:
        return ""

    now = time.time()
    if _cmc_cache["context"] and (now - float(_cmc_cache["timestamp"])) < CMC_CACHE_TTL_SECONDS:
        return str(_cmc_cache["context"])

    headers = {
        "X-CMC_PRO_API_KEY": cmc_api_key,
        "Accept": "application/json",
    }
    params = {
        "symbol": "BTC",
        "convert": "USD",
    }
    try:
        response = requests.get(CMC_QUOTES_API_URL, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
        btc_data = payload.get("data", {}).get("BTC", [])
        if isinstance(btc_data, list) and btc_data:
            btc_item = btc_data[0]
        elif isinstance(btc_data, dict):
            btc_item = btc_data
        else:
            return ""

        usd_quote = btc_item.get("quote", {}).get("USD", {})
        price = usd_quote.get("price")
        chg_1h = usd_quote.get("percent_change_1h") or 0.0
        chg_24h = usd_quote.get("percent_change_24h") or 0.0
        vol_chg_24h = usd_quote.get("volume_change_24h") or 0.0

        if price is None:
            return ""

        context = (
            f"Tržní kontext BTC (CoinMarketCap): cena {price:.2f} USD, "
            f"změna 1h: {chg_1h:+.2f}%, změna 24h: {chg_24h:+.2f}%, "
            f"změna objemu 24h: {vol_chg_24h:+.2f}%."
        )
        _cmc_cache["timestamp"] = now
        _cmc_cache["context"] = context
        return context
    except Exception as exc:
        logger.warning("Nepodařilo se načíst tržní kontext z CoinMarketCap: %s", exc)
        return str(_cmc_cache["context"]) if _cmc_cache["context"] else ""


def _is_bitcoin_relevant(title: str, desc: str) -> bool:
    """Ověří, zda se článek týká Bitcoinu (šetří kvótu Gemini API na obecných krypto feedech)."""
    combined = f"{title} {desc}"
    return bool(re.search(r"\b(bitcoin|btc)\b", combined, re.IGNORECASE))


def fetch_bitcoin_news(cmc_api_key: str = "", limit: int = 5) -> List[Dict[str, Any]]:
    """
    Stáhne nejnovější články o Bitcoinu ze všech veřejných RSS kanálů (Cointelegraph, CoinDesk, Decrypt)
    a obohatí je o aktuální tržní kontext z CoinMarketCap Free API.
    """
    articles: List[Dict[str, Any]] = []
    seen_links: set = set()
    req_headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BTCTradingBot/1.0)"
    }

    for feed_url in BTC_RSS_FEEDS:
        try:
            response = requests.get(feed_url, headers=req_headers, timeout=15)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            feed_count = 0
            is_btc_only_feed = "tag/bitcoin" in feed_url
            for item in root.findall(".//item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or item.findtext("guid") or title).strip()
                desc = strip_html_tags(item.findtext("description") or "")
                if not title or link in seen_links:
                    continue
                if not is_btc_only_feed and not _is_bitcoin_relevant(title, desc):
                    continue
                seen_links.add(link)
                articles.append(
                    {
                        "id": link,
                        "title": title,
                        "body": desc,
                        "url": link,
                    }
                )
                feed_count += 1
                if limit and feed_count >= limit:
                    break
        except Exception as exc:
            logger.warning("Nelze načíst RSS kanál %s: %s", feed_url, exc)

    market_context = fetch_cmc_market_context(cmc_api_key) if cmc_api_key else ""

    if market_context:
        if articles:
            for article in articles:
                article["market_context"] = market_context
        else:
            articles.append(
                {
                    "id": f"cmc_{int(time.time() // 300)}",
                    "title": market_context,
                    "body": "",
                    "market_context": "",
                }
            )

    return articles


def analyze_sentiment(model: genai.GenerativeModel, news_text: str) -> str:
    """
    Odešle text zprávy do modelu Gemini a vrátí vyhodnocený sentiment:
    POSITIVE, NEGATIVE, nebo NEUTRAL.
    """
    full_prompt = f"{SENTIMENT_PROMPT}\n\nZpráva:\n{news_text}"
    response = None

    try:
        response = model.generate_content(full_prompt)
    except Exception as primary_exc:
        logger.warning(
            "Primární model Gemini vrátil chybu (%s), zkouším záložní model Gemini...",
            primary_exc,
        )
        for fallback_name in FALLBACK_GEMINI_MODELS:
            try:
                fallback_model = genai.GenerativeModel(fallback_name)
                response = fallback_model.generate_content(full_prompt)
                logger.info("Úspěšně použit model %s.", fallback_name)
                break
            except Exception:
                continue

    if response is None:
        logger.error("Žádný z modelů Gemini neodpověděl, nastavuji NEUTRAL.")
        return "NEUTRAL"

    raw_answer = (response.text or "").strip().upper()

    # Očištění případné interpunkce kolem jednoho slova
    cleaned = raw_answer.strip(".,!?;:\"' \n\t")
    if cleaned in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
        return cleaned

    for valid in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
        if valid in cleaned.split():
            return valid

    logger.warning("Neočekávaná odpověď modelu Gemini ('%s'), nastavuji NEUTRAL.", raw_answer)
    return "NEUTRAL"


def analyze_sentiment_with_score(model: genai.GenerativeModel, news_text: str) -> Tuple[str, int]:
    """
    Odešle zprávu do Gemini a vrátí dvojici (SENTIMENT, SKÓRE 1-10).
    Pokud model vrátí jen slovo bez čísla, použije se výchozí skóre 10.
    """
    full_prompt = f"{SENTIMENT_SCORE_PROMPT}\n\nZpráva:\n{news_text}"
    response = None

    try:
        response = model.generate_content(full_prompt)
    except Exception as primary_exc:
        logger.warning(
            "Primární model Gemini vrátil chybu (%s), zkouším záložní model Gemini...",
            primary_exc,
        )
        for fallback_name in FALLBACK_GEMINI_MODELS:
            try:
                fallback_model = genai.GenerativeModel(fallback_name)
                response = fallback_model.generate_content(full_prompt)
                logger.info("Úspěšně použit model %s.", fallback_name)
                break
            except Exception:
                continue

    if response is None:
        logger.error("Žádný z modelů Gemini neodpověděl, nastavuji NEUTRAL (0/10).")
        return "NEUTRAL", 0

    raw_answer = (response.text or "").strip().upper()
    sentiment = "NEUTRAL"
    for valid in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
        if re.search(rf"\b{valid}\b", raw_answer):
            sentiment = valid
            break

    score_match = re.search(r"\b(10|[1-9])\b", raw_answer)
    score = int(score_match.group(1)) if score_match else (10 if sentiment in ("POSITIVE", "NEGATIVE") else 5)
    score = max(1, min(10, score))

    if sentiment == "NEUTRAL" and not re.search(r"\b(POSITIVE|NEGATIVE|NEUTRAL)\b", raw_answer):
        logger.warning("Neočekávaná odpověď modelu Gemini ('%s'), nastavuji NEUTRAL.", raw_answer)

    return sentiment, score


def calculate_rsi(closes: List[float], period: int = RSI_PERIOD) -> Optional[float]:
    """
    Vypočítá indikátor RSI (Relative Strength Index) z uzavíracích cen svíček.
    """
    if len(closes) < period + 1:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas[:period]]
    losses = [max(-d, 0.0) for d in deltas[:period]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for d in deltas[period:]:
        gain = max(d, 0.0)
        loss = max(-d, 0.0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def fetch_btc_rsi(
    exchange: ccxt.binance,
    symbol: str = SYMBOL,
    timeframe: str = RSI_TIMEFRAME,
    period: int = RSI_PERIOD,
) -> Optional[float]:
    """
    Stáhne poslední svíčky z Binance a vrátí aktuální hodnotu RSI(14).
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=period + 20)
        if not isinstance(ohlcv, list) or len(ohlcv) < period + 1:
            return None
        closes = [float(candle[4]) for candle in ohlcv if isinstance(candle, (list, tuple)) and len(candle) >= 5]
        return calculate_rsi(closes, period=period)
    except Exception as exc:
        logger.warning("Nelze načíst %s RSI z Binance (%s) -> pokračuji bez RSI filtru.", timeframe, exc)
        return None


def calculate_ema(closes: List[float], period: int = EMA_PERIOD) -> Optional[float]:
    """
    Vypočítá exponenciální klouzavý průměr (EMA) z uzavíracích cen svíček.
    """
    if len(closes) < period:
        return None

    sma = sum(closes[:period]) / period
    ema = sma
    multiplier = 2.0 / (period + 1)
    for price in closes[period:]:
        ema = (price - ema) * multiplier + ema
    return round(ema, 2)


def fetch_btc_ema(
    exchange: ccxt.binance,
    symbol: str = SYMBOL,
    timeframe: str = EMA_TIMEFRAME,
    period: int = EMA_PERIOD,
) -> Optional[float]:
    """
    Stáhne poslední svíčky z Binance a vrátí aktuální hodnotu EMA(50) pro určení hlavního trendu.
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=period + 30)
        if not isinstance(ohlcv, list) or len(ohlcv) < period:
            return None
        closes = [float(candle[4]) for candle in ohlcv if isinstance(candle, (list, tuple)) and len(candle) >= 5]
        return calculate_ema(closes, period=period)
    except Exception as exc:
        logger.warning("Nelze načíst %s EMA%d z Binance (%s) -> pokračuji bez trend filtru.", timeframe, period, exc)
        return None


def calculate_stop_loss_price(entry_price: float, side: str, stop_loss_pct: float = STOP_LOSS_PCT) -> float:
    """
    Vypočítá ochrannou Stop-Loss cenu (2 % proti směru obchodu).
    - Pro LONG ('buy'): Stop-Loss je o 2 % níže než vstupní cena.
    - Pro SHORT ('sell'): Stop-Loss je o 2 % výše než vstupní cena.
    """
    if side == "buy":
        return round(entry_price * (1.0 - stop_loss_pct), 2)
    elif side == "sell":
        return round(entry_price * (1.0 + stop_loss_pct), 2)
    else:
        raise ValueError(f"Neplatný směr obchodu: {side}")


def calculate_take_profit_price(entry_price: float, side: str, take_profit_pct: float = TAKE_PROFIT_PCT) -> float:
    """
    Vypočítá cílovou Take-Profit cenu (2 % ve směru obchodu).
    - Pro LONG ('buy'): Take-Profit je o 2 % výše než vstupní cena.
    - Pro SHORT ('sell'): Take-Profit je o 2 % níže než vstupní cena.
    """
    if side == "buy":
        return round(entry_price * (1.0 + take_profit_pct), 2)
    elif side == "sell":
        return round(entry_price * (1.0 - take_profit_pct), 2)
    else:
        raise ValueError(f"Neplatný směr obchodu: {side}")


def calculate_trailing_stop_price(
    entry_price: float,
    best_price: float,
    position_side: str,
    current_trailing_sl: Optional[float] = None,
) -> Optional[float]:
    """
    Vypočítá posuvný Stop-Loss (ochranu zisku), jakmile obchod dosáhne alespoň +0.8 % zisku.
    - Minimální zamčený Stop-Loss je +0.3 % v zisku (pokryje ~0.2 % poplatky burzy + čistý zisk).
    - Při dalším příznivém pohybu drží odstup 0.5 % od nejlepší dosažené ceny a nikdy se nevrací zpět.
    """
    if entry_price <= 0:
        return None

    if position_side == "short":
        profit_pct = (entry_price - best_price) / entry_price
        if profit_pct < TRAILING_ACTIVATION_PCT:
            return None
        lock_sl = round(entry_price * (1.0 - PROFIT_LOCK_PCT), 2)
        trail_sl = round(best_price * (1.0 + TRAILING_DISTANCE_PCT), 2)
        candidate_sl = min(lock_sl, trail_sl)
        if current_trailing_sl is not None:
            candidate_sl = min(candidate_sl, current_trailing_sl)
        return candidate_sl

    if position_side == "long":
        profit_pct = (best_price - entry_price) / entry_price
        if profit_pct < TRAILING_ACTIVATION_PCT:
            return None
        lock_sl = round(entry_price * (1.0 + PROFIT_LOCK_PCT), 2)
        trail_sl = round(best_price * (1.0 - TRAILING_DISTANCE_PCT), 2)
        candidate_sl = max(lock_sl, trail_sl)
        if current_trailing_sl is not None:
            candidate_sl = max(candidate_sl, current_trailing_sl)
        return candidate_sl

    raise ValueError(f"Neplatný směr pozice: {position_side}")


def _reset_position_state(position_side: str) -> None:
    """Vynuluje interní stav sledované pozice po jejím uzavření."""
    _position_state[f"{position_side}_entry_price"] = None
    _position_state[f"{position_side}_amount"] = None
    _position_state[f"{position_side}_best_price"] = None
    _position_state[f"{position_side}_trailing_sl"] = None


def _log_exchange_closed_position(
    exchange: ccxt.binance,
    symbol: str,
    position_side: str,
    current_price: float,
    usdc_free: float,
) -> None:
    """Zapíše do logu výsledek pozice, kterou mezitím uzavřela přímo burza Binance (Stop-Loss / Trailing SL)."""
    entry_price = _position_state.get(f"{position_side}_entry_price")
    amount = _position_state.get(f"{position_side}_amount")
    last_sl = _position_state.get(f"{position_side}_trailing_sl")
    if not entry_price or not amount:
        return

    is_short = position_side == "short"
    label = "SHORT" if is_short else "LONG"
    close_trade_side = "buy" if is_short else "sell"
    exit_price = float(last_sl) if last_sl else current_price

    try:
        trades = exchange.fetch_my_trades(symbol, limit=5, params={"marginMode": "cross"})
        if isinstance(trades, list):
            for trade in reversed(trades):
                if trade.get("side") == close_trade_side and float(trade.get("price") or 0.0) > 0:
                    exit_price = float(trade["price"])
                    break
    except Exception:
        pass

    pnl_usdc = ((float(entry_price) - exit_price) if is_short else (exit_price - float(entry_price))) * float(amount)
    pnl_pct = (((float(entry_price) - exit_price) if is_short else (exit_price - float(entry_price))) / float(entry_price)) * 100.0
    reason = "Trailing Stop-Loss v zisku" if last_sl else "Stop-Loss"

    logger.info(
        "🏁 POZICE UZAVŘENA (burza Binance - %s): %s %.5f BTC | Vstup: ~%.2f -> Výstup: ~%.2f USDC | "
        "Výsledek: %+.2f USDC (%+.2f%%) | Nový volný zůstatek: %.2f USDC",
        reason,
        label,
        float(amount),
        float(entry_price),
        exit_price,
        pnl_usdc,
        pnl_pct,
        usdc_free,
    )


def _estimate_entry_price(
    exchange: ccxt.binance,
    symbol: str,
    position_side: str,
    position_amount: float,
    sl_orders: List[Dict[str, Any]],
    current_price: float,
) -> float:
    """
    Zjistí skutečnou průměrnou vstupní cenu otevřené pozice z historie obchodů na Binance
    (případně z původních Stop-Loss objednávek) a uloží ji, aby ji posun Stop-Lossu nezkreslil.
    """
    cached_entry = _position_state.get(f"{position_side}_entry_price")
    if cached_entry and float(cached_entry) > 0:
        return float(cached_entry)

    target_trade_side = "sell" if position_side == "short" else "buy"
    try:
        trades = exchange.fetch_my_trades(symbol, limit=20, params={"marginMode": "cross"})
        if isinstance(trades, list) and trades:
            accum_amt = 0.0
            total_cost = 0.0
            for trade in reversed(trades):
                if trade.get("side") != target_trade_side:
                    continue
                amt = float(trade.get("amount") or 0.0)
                price = float(trade.get("price") or 0.0)
                if amt <= 0 or price <= 0:
                    continue
                needed = max(0.0, position_amount - accum_amt)
                take_amt = min(amt, needed) if accum_amt + amt > position_amount * 1.02 else amt
                accum_amt += take_amt
                total_cost += take_amt * price
                if accum_amt >= position_amount * 0.98:
                    break
            if accum_amt > 0:
                est_entry = round(total_cost / accum_amt, 2)
                _position_state[f"{position_side}_entry_price"] = est_entry
                return est_entry
    except Exception:
        pass

    sl_prices = [
        float(o.get("stopPrice") or o.get("price") or 0.0)
        for o in sl_orders
        if float(o.get("stopPrice") or o.get("price") or 0.0) > 0
    ]
    if sl_prices:
        avg_sl = sum(sl_prices) / len(sl_prices)
        divisor = (1.0 + STOP_LOSS_PCT) if position_side == "short" else (1.0 - STOP_LOSS_PCT)
        est_entry = round(avg_sl / divisor, 2)
        _position_state[f"{position_side}_entry_price"] = est_entry
        return est_entry

    return current_price


def _close_margin_position(
    exchange: ccxt.binance,
    symbol: str,
    close_side: str,
    amount: float,
    open_orders: List[Dict[str, Any]],
) -> None:
    """Zruší otevřené Stop-Loss příkazy a uzavře pozici na Cross Marginu tržním příkazem."""
    for order in open_orders:
        try:
            exchange.cancel_order(order["id"], symbol, params={"marginMode": "cross"})
        except Exception:
            pass

    close_amount = float(exchange.amount_to_precision(symbol, amount))
    exchange.create_order(
        symbol=symbol,
        type="market",
        side=close_side,
        amount=close_amount,
        params={
            "marginMode": "cross",
            "sideEffectType": "AUTO_BORROW_REPAY",
        },
    )


def _replace_stop_loss_order(
    exchange: ccxt.binance,
    symbol: str,
    sl_side: str,
    amount: float,
    new_sl_price: float,
    old_sl_orders: List[Dict[str, Any]],
) -> None:
    """Nahradí stávající Stop-Loss objednávky na Binance novým utaženým Stop-Lossem v zisku."""
    for order in old_sl_orders:
        try:
            exchange.cancel_order(order["id"], symbol, params={"marginMode": "cross"})
        except Exception:
            pass

    close_amount = float(exchange.amount_to_precision(symbol, amount))
    formatted_price = round(new_sl_price, 2)
    try:
        precision_val = exchange.price_to_precision(symbol, formatted_price)
        if isinstance(precision_val, (str, int, float)):
            formatted_price = float(precision_val)
    except Exception:
        pass

    exchange.create_order(
        symbol=symbol,
        type="STOP_LOSS_LIMIT",
        side=sl_side,
        amount=close_amount,
        price=formatted_price,
        params={
            "stopPrice": formatted_price,
            "marginMode": "cross",
            "sideEffectType": "AUTO_BORROW_REPAY",
        },
    )


def _monitor_single_direction(
    exchange: ccxt.binance,
    symbol: str,
    position_side: str,
    position_amount: float,
    current_price: float,
    open_orders: List[Dict[str, Any]],
) -> None:
    """Sleduje jednu otevřenou pozici (SHORT nebo LONG), hlídá Take-Profit i posuvný Stop-Loss v zisku."""
    is_short = position_side == "short"
    close_side = "buy" if is_short else "sell"
    entry_side = "sell" if is_short else "buy"
    label = "SHORT" if is_short else "LONG"

    sl_orders = [
        o for o in open_orders
        if o.get("side") == close_side and float(o.get("stopPrice") or o.get("price") or 0.0) > 0
    ]
    est_entry = _estimate_entry_price(exchange, symbol, position_side, position_amount, sl_orders, current_price)
    _position_state[f"{position_side}_amount"] = position_amount
    _position_state["idle_since"] = None

    prev_best = _position_state.get(f"{position_side}_best_price")
    if prev_best is None:
        best_price = current_price
    else:
        best_price = min(float(prev_best), current_price) if is_short else max(float(prev_best), current_price)
    _position_state[f"{position_side}_best_price"] = best_price

    tp_price = calculate_take_profit_price(est_entry, entry_side, TAKE_PROFIT_PCT)
    new_trailing_sl = calculate_trailing_stop_price(
        est_entry,
        best_price,
        position_side,
        _position_state.get(f"{position_side}_trailing_sl"),
    )
    if new_trailing_sl is not None:
        _position_state[f"{position_side}_trailing_sl"] = new_trailing_sl

    sl_prices = [float(o.get("stopPrice") or o.get("price") or 0.0) for o in sl_orders]
    active_sl = (
        new_trailing_sl
        if new_trailing_sl is not None
        else ((sum(sl_prices) / len(sl_prices)) if sl_prices else calculate_stop_loss_price(est_entry, entry_side))
    )

    pnl_usdc = ((est_entry - current_price) if is_short else (current_price - est_entry)) * position_amount
    pnl_pct = (((est_entry - current_price) if is_short else (current_price - est_entry)) / est_entry) * 100.0 if est_entry else 0.0
    lock_tag = " [ZISK ZAMČEN 🛡️]" if new_trailing_sl is not None else ""

    logger.info(
        "📊 STAV POZICE: %s %.5f BTC (%.2f USDC) | Vstup: ~%.2f USDC | "
        "Aktuální cena: %.2f USDC | PnL: %+.2f USDC (%+.2f%%) | "
        "Take-Profit cíl: %.2f USDC | Stop-Loss: %.2f USDC%s",
        label,
        position_amount,
        position_amount * current_price,
        est_entry,
        current_price,
        pnl_usdc,
        pnl_pct,
        tp_price,
        active_sl,
        lock_tag,
    )

    tp_hit = (current_price <= tp_price) if is_short else (current_price >= tp_price)
    if tp_hit:
        logger.info(
            "🎯 TAKE-PROFIT DOSAŽEN (aktuální cena %.2f vs cíl %.2f USDC)! "
            "Uzavírám %s pozici (%.5f BTC) a uzamykám zisk %+.2f USDC...",
            current_price,
            tp_price,
            label,
            position_amount,
            pnl_usdc,
        )
        _close_margin_position(exchange, symbol, close_side, position_amount, open_orders)
        _reset_position_state(position_side)
        logger.info(
            "🏁 POZICE UZAVŘENA (Take-Profit): %s %.5f BTC | Vstup: ~%.2f -> Výstup: %.2f USDC | Výsledek: %+.2f USDC (%+.2f%%)",
            label,
            position_amount,
            est_entry,
            current_price,
            pnl_usdc,
            pnl_pct,
        )
        return

    if new_trailing_sl is not None:
        trailing_hit = (current_price >= new_trailing_sl) if is_short else (current_price <= new_trailing_sl)
        if trailing_hit:
            logger.info(
                "🛡️ TRAILING STOP-LOSS V ZISKU DOSAŽEN (aktuální cena %.2f vs SL %.2f USDC)! "
                "Uzavírám %s pozici (%.5f BTC) a uzamykám zisk %+.2f USDC...",
                current_price,
                new_trailing_sl,
                label,
                position_amount,
                pnl_usdc,
            )
            _close_margin_position(exchange, symbol, close_side, position_amount, open_orders)
            _reset_position_state(position_side)
            logger.info(
                "🏁 POZICE UZAVŘENA (Trailing Stop-Loss v zisku): %s %.5f BTC | Vstup: ~%.2f -> Výstup: %.2f USDC | Výsledek: %+.2f USDC (%+.2f%%)",
                label,
                position_amount,
                est_entry,
                current_price,
                pnl_usdc,
                pnl_pct,
            )
            return

        should_update_exchange_sl = False
        if not sl_prices:
            should_update_exchange_sl = True
        elif is_short and new_trailing_sl <= min(sl_prices) * (1.0 - SL_UPDATE_MIN_STEP_PCT):
            should_update_exchange_sl = True
        elif (not is_short) and new_trailing_sl >= max(sl_prices) * (1.0 + SL_UPDATE_MIN_STEP_PCT):
            should_update_exchange_sl = True

        if should_update_exchange_sl:
            old_sl_display = min(sl_prices) if (is_short and sl_prices) else (max(sl_prices) if sl_prices else active_sl)
            logger.info(
                "🛡️ OCHRANA ZISKU: Posouvám Stop-Loss na Binance z %.2f na %.2f USDC (nejlepší cena: %.2f USDC)!",
                old_sl_display,
                new_trailing_sl,
                best_price,
            )
            _replace_stop_loss_order(exchange, symbol, close_side, position_amount, new_trailing_sl, sl_orders)


def monitor_open_positions_and_take_profit(
    exchange: ccxt.binance,
    symbol: str = SYMBOL,
) -> None:
    """
    Zkontroluje otevřené pozice na Binance Cross Margin, vypíše živý report o zisku/ztrátě (PnL),
    posouvá Stop-Loss do zisku (Trailing Stop) a při dosažení Take-Profitu pozici uzavře.
    """
    if exchange.options.get("defaultType") != "margin":
        return

    try:
        exchange.load_markets()
        balance = exchange.fetch_balance({"type": "margin"})
        stats = get_margin_btc_borrowed_and_free(balance)
        btc_borrowed = stats["btc_borrowed"]
        btc_free = stats["btc_free"]

        ticker = exchange.fetch_ticker(symbol)
        current_price = float(ticker["last"])
        open_orders = exchange.fetch_open_orders(symbol, params={"marginMode": "cross"})

        if btc_borrowed >= 0.0001:
            _monitor_single_direction(exchange, symbol, "short", btc_borrowed, current_price, open_orders)
        else:
            if _position_state.get("short_entry_price"):
                _log_exchange_closed_position(exchange, symbol, "short", current_price, stats["usdc_free"])
            _reset_position_state("short")

        if btc_free >= 0.0001 and btc_borrowed < 0.0001:
            _monitor_single_direction(exchange, symbol, "long", btc_free, current_price, open_orders)
        else:
            if _position_state.get("long_entry_price"):
                _log_exchange_closed_position(exchange, symbol, "long", current_price, stats["usdc_free"])
            _reset_position_state("long")

        if btc_borrowed < 0.0001 and btc_free < 0.0001:
            if _position_state.get("idle_since") is None:
                _position_state["idle_since"] = time.time()
            idle_mins = int((time.time() - float(_position_state["idle_since"])) // 60)
            last_title = _position_state.get("last_article_title")
            last_sent = _position_state.get("last_sentiment")
            last_info = f" | Poslední zpráva: '{last_title}' -> {last_sent}" if last_title else ""
            logger.info(
                "📊 STAV ÚČTU: Žádná otevřená pozice (čekám %d min) | Volné USDC: %.2f | "
                "Cena BTC: %.2f USDC | Čekám na nový článek z RSS se signálem POSITIVE (LONG) nebo NEGATIVE (SHORT)%s",
                idle_mins,
                stats["usdc_free"],
                current_price,
                last_info,
            )
    except Exception as exc:
        logger.warning("Chyba při kontrole otevřených pozic / Take-Profitu: %s", exc)


def has_open_btc_position(exchange: ccxt.binance) -> bool:
    """
    Zjistí, zda už na Margin/Spot účtu běží otevřená LONG nebo SHORT pozice v BTC (>= 0.0001 BTC).
    """
    try:
        balance = exchange.fetch_balance()
    except Exception:
        return False

    if not isinstance(balance, dict):
        return False

    stats = get_margin_btc_borrowed_and_free(balance)
    return stats["btc_borrowed"] >= 0.0001 or stats["btc_free"] >= 0.0001


def calculate_dynamic_investment(
    exchange: ccxt.binance,
    fallback_usdc: float = TRADE_INVESTMENT_USDT,
) -> float:
    """
    Dynamicky spočítá velikost investice jako 90 % volných USDC na účtu (složené úročení).
    Pokud zůstatek nelze načíst, vrátí bezpečně výchozí hodnotu fallback_usdc.
    """
    try:
        balance = exchange.fetch_balance()
    except Exception:
        return fallback_usdc

    if not isinstance(balance, dict):
        return fallback_usdc

    usdc_free = get_margin_btc_borrowed_and_free(balance).get("usdc_free", 0.0)
    if usdc_free >= MIN_TRADE_USDC:
        return round(usdc_free * TRADE_CAPITAL_RATIO, 2)
    return fallback_usdc


def execute_trade(
    exchange: ccxt.binance,
    sentiment: str,
    investment_usdt: Optional[float] = None,
    symbol: str = SYMBOL,
    signal_score: int = 8,
) -> Optional[Dict[str, Any]]:
    """
    Exekuční logika pro BTC/USDC na základě vyhodnocení sentimentu:
    - Před každým obchodem načte trhy (exchange.load_markets()), zjistí aktuální cenu (ticker['last']),
      dynamicky vypočítá množství BTC jako TRADE_INVESTMENT_USDT / ticker['last']
      a zaokrouhlí jej přes exchange.amount_to_precision(symbol, amount).
    - Pokud již běží otevřená pozice (>= 0.0001 BTC), nový obchod neotevře (max. 1 pozice najednou).
    - POSITIVE -> otevře Market Buy (LONG) pozici + ochranný Stop-Loss (-2 %) + cílový Take-Profit (+2 %)
    - NEGATIVE -> otevře Market Sell (SHORT) pozici + ochranný Stop-Loss (+2 %) + cílový Take-Profit (-2 %)
    - NEUTRAL  -> nedělá nic
    """
    sentiment = sentiment.upper().strip()

    if sentiment == "NEUTRAL":
        logger.info("Sentiment je NEUTRAL -> neprovádím žádnou obchodní akci.")
        return None

    if sentiment == "POSITIVE":
        entry_side = "buy"
        sl_side = "sell"
        position_label = "LONG"
    elif sentiment == "NEGATIVE":
        entry_side = "sell"
        sl_side = "buy"
        position_label = "SHORT"
    else:
        logger.warning("Neznámý sentiment '%s' -> neprovádím žádnou akci.", sentiment)
        return None

    if has_open_btc_position(exchange):
        logger.info(
            "⏸️ Na účtu již běží otevřená pozice -> nový %s obchod (%s) neotevírám (max. 1 pozice najednou).",
            position_label,
            symbol,
        )
        return None

    max_rsi = RSI_STRONG_OVERBOUGHT if signal_score >= STRONG_SIGNAL_SCORE else RSI_OVERBOUGHT
    min_rsi = RSI_STRONG_OVERSOLD if signal_score >= STRONG_SIGNAL_SCORE else RSI_OVERSOLD

    rsi = fetch_btc_rsi(exchange, symbol=symbol)
    if rsi is not None:
        if position_label == "LONG" and rsi > max_rsi:
            logger.info(
                "⏸️ RSI filtr: 15m RSI je %.1f (> %.1f pro sílu %d/10, překoupený trh) -> LONG neotevírám.",
                rsi,
                max_rsi,
                signal_score,
            )
            return None
        if position_label == "SHORT" and rsi < min_rsi:
            logger.info(
                "⏸️ RSI filtr: 15m RSI je %.1f (< %.1f pro sílu %d/10, přeprodaný trh) -> SHORT neotevírám.",
                rsi,
                min_rsi,
                signal_score,
            )
            return None
        logger.info("📈 RSI filtr OK: 15m RSI = %.1f (povoleno pro %s, síla %d/10).", rsi, position_label, signal_score)

    if investment_usdt is None:
        investment_usdt = calculate_dynamic_investment(exchange, TRADE_INVESTMENT_USDT)

    # 1. Načtení informací o trzích a zjištění aktuální ceny aktiva
    exchange.load_markets()
    ticker = exchange.fetch_ticker(symbol)
    current_price = float(ticker["last"])

    ema_50 = fetch_btc_ema(exchange, symbol=symbol)
    if ema_50 is not None:
        if position_label == "LONG" and current_price < ema_50:
            logger.info(
                "⏸️ Trend filtr: cena %.2f USDC je pod 1h EMA50 (%.2f USDC, klesající trend) -> LONG neotevírám.",
                current_price,
                ema_50,
            )
            return None
        if position_label == "SHORT" and current_price > ema_50:
            logger.info(
                "⏸️ Trend filtr: cena %.2f USDC je nad 1h EMA50 (%.2f USDC, rostoucí trend) -> SHORT neotevírám.",
                current_price,
                ema_50,
            )
            return None
        logger.info(
            "🧭 Trend filtr OK: cena %.2f USDC vs 1h EMA50 %.2f USDC (povoleno pro %s).",
            current_price,
            ema_50,
            position_label,
        )

    # Dynamický výpočet množství BTC a striktní zaokrouhlení dle pravidel burzy Binance
    raw_amount = investment_usdt / current_price
    amount = float(exchange.amount_to_precision(symbol, raw_amount))

    # 2. Otevření tržní (Market) pozice
    logger.info(
        "Otevírám %s pozici (Market %s) pro %s (investice: %.2f USDC, cena: %.2f USDC, množství: %s BTC)...",
        position_label,
        entry_side.upper(),
        symbol,
        investment_usdt,
        current_price,
        amount,
    )

    order_params: Dict[str, Any] = {}
    if exchange.options.get("defaultType") == "margin":
        order_params["marginMode"] = "cross"
        order_params["sideEffectType"] = "AUTO_BORROW_REPAY"

    try:
        market_order = exchange.create_order(
            symbol=symbol,
            type="market",
            side=entry_side,
            amount=amount,
            params=order_params,
        )
    except ccxt.BaseError as exc:
        if "-3003" in str(exc) and exchange.options.get("defaultType") == "margin":
            logger.info("Margin účet na Binance ještě není aktivován (-3003), přepínám automaticky na SPOT účet...")
            exchange.options["defaultType"] = "spot"
            market_order = exchange.create_order(
                symbol=symbol,
                type="market",
                side=entry_side,
                amount=amount,
            )
        else:
            raise

    # Zjištění skutečné vstupní ceny z objednávky nebo použití ticker['last']
    entry_price = float(market_order.get("average") or market_order.get("price") or current_price)

    # 3. Výpočet ochranného Stop-Loss příkazu (2 % proti směru obchodu) a Take-Profit cíle (2 % ve směru obchodu)
    stop_price = calculate_stop_loss_price(entry_price, entry_side, STOP_LOSS_PCT)
    take_profit_price = calculate_take_profit_price(entry_price, entry_side, TAKE_PROFIT_PCT)

    logger.info(
        "Zadávám ochranný Stop-Loss (%s) na ceně %.2f USDC a aktivuji hlídání Take-Profit cíle na %.2f USDC (vstup: %.2f USDC)...",
        sl_side.upper(),
        stop_price,
        take_profit_price,
        entry_price,
    )
    sl_params: Dict[str, Any] = {"stopPrice": stop_price}
    if exchange.options.get("defaultType") == "future":
        sl_params["closePosition"] = False
        sl_params["reduceOnly"] = True
    elif exchange.options.get("defaultType") == "margin":
        sl_params["marginMode"] = "cross"
        sl_params["sideEffectType"] = "AUTO_BORROW_REPAY"

    stop_loss_order = exchange.create_order(
        symbol=symbol,
        type="STOP_LOSS_LIMIT" if exchange.options.get("defaultType") in ("spot", "margin") else "STOP_MARKET",
        side=sl_side,
        amount=amount,
        price=stop_price if exchange.options.get("defaultType") in ("spot", "margin") else None,
        params=sl_params,
    )

    _reset_position_state("short" if entry_side == "sell" else "long")
    _position_state["idle_since"] = None
    logger.info(
        "🚀 POZICE OTEVŘENA: %s %.5f BTC (~%.2f USDC) | Vstup: %.2f USDC | Stop-Loss: %.2f USDC | Take-Profit cíl: %.2f USDC",
        position_label,
        amount,
        amount * entry_price,
        entry_price,
        stop_price,
        take_profit_price,
    )

    return {
        "sentiment": sentiment,
        "position": position_label,
        "symbol": symbol,
        "amount": amount,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "take_profit_price": take_profit_price,
        "market_order": market_order,
        "stop_loss_order": stop_loss_order,
    }


def process_latest_news(
    exchange: ccxt.binance,
    model: genai.GenerativeModel,
    cmc_api_key: str,
    seen_ids: set,
    investment_usdt: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Stáhne nejnovější zprávy o Bitcoinu + tržní kontext z CoinMarketCapu,
    každou novou zprávu vyhodnotí pomocí Gemini a provede příslušnou
    obchodní akci na páru BTC/USDC.
    """
    articles = fetch_bitcoin_news(cmc_api_key)
    executed_trades: List[Dict[str, Any]] = []

    for article in articles:
        article_id = str(
            article.get("id")
            or article.get("url")
            or article.get("source_url")
            or article.get("title", "")
        )
        if not article_id or article_id in seen_ids:
            continue

        seen_ids.add(article_id)
        title = (article.get("title") or "").strip()
        body = (article.get("body") or article.get("subtitle") or article.get("description") or "").strip()
        market_context = (article.get("market_context") or "").strip()

        if not title and not body:
            continue

        # Pokud už běží otevřená pozice, jen označíme článek jako přečtený (vyhneme se
        # zbytečnému volání Gemini API i otevírání další pozice).
        if has_open_btc_position(exchange):
            continue

        short_body = body[:500]
        parts = [p for p in (title, short_body, market_context) if p]
        news_text = "\n".join(parts)

        logger.info("Analyzuji novou zprávu: %s", title)
        sentiment, score = analyze_sentiment_with_score(model, news_text)
        logger.info("Výsledek analýzy Gemini: %s (síla: %d/10)", sentiment, score)
        _position_state["last_article_title"] = title[:80]
        _position_state["last_sentiment"] = f"{sentiment} ({score}/10)"

        if sentiment in ("POSITIVE", "NEGATIVE") and score < MIN_SENTIMENT_SCORE:
            logger.info(
                "⏭️ Zpráva má nízkou sílu signálu (%s %d/10 < %d/10) -> neobchoduji.",
                sentiment,
                score,
                MIN_SENTIMENT_SCORE,
            )
            continue

        try:
            trade_result = execute_trade(
                exchange,
                sentiment,
                investment_usdt=investment_usdt,
                symbol=SYMBOL,
                signal_score=score,
            )
            if trade_result:
                executed_trades.append(trade_result)
        except ccxt.InsufficientFunds as exc:
            logger.info(
                "✅ Obchodní příkaz (%s na %s) je připraven, "
                "ale volný kolaterál je již plně využit v otevřených pozicích (%s).",
                sentiment,
                SYMBOL,
                exc,
            )
        except ccxt.BaseError as exc:
            msg = str(exc)
            if any(k in msg.lower() for k in ("insufficient", "-2010", "-3006", "borrow", "balance", "margin")):
                logger.info(
                    "ℹ️ Kapacita Margin účtu je již plně využita stávajícími otevřenými pozicemi (%s).",
                    msg,
                )
            else:
                raise

    return executed_trades


def run_bot(poll_interval: int = POLL_INTERVAL_SECONDS) -> None:
    """
    Hlavní smyčka algotradingového bota.
    """
    logger.info("Spouštím BTC/USDC Sentiment Algotrading Bota (Stop-Loss 2%% + Take-Profit 2%%)...")
    keys = load_api_keys()
    exchange = create_exchange(keys["BINANCE_API_KEY"], keys["BINANCE_SECRET"])
    account_stats = verify_binance_account(exchange)
    model = configure_gemini(keys["GEMINI_API_KEY"])

    seen_ids: set = set()

    # Při startu vždy označíme stávající články v RSS jako přečtené,
    # aby restart kontejneru neotevíral obchody zpětně na staré zprávy.
    try:
        initial_articles = fetch_bitcoin_news("")
        for art in initial_articles:
            aid = str(art.get("id") or art.get("url") or art.get("source_url") or art.get("title", ""))
            if aid:
                seen_ids.add(aid)
        logger.info(
            "Inicializace RSS: označeno %d stávajících článků jako přečtených (obchoduji až nově příchozí zprávy).",
            len(seen_ids),
        )
    except Exception:
        pass

    while True:
        try:
            # 1. Kontrola otevřených pozic, výpis PnL a případné uzavření na Take-Profitu
            monitor_open_positions_and_take_profit(exchange, symbol=SYMBOL)

            # 2. Kontrola nových zpráv
            process_latest_news(
                exchange=exchange,
                model=model,
                cmc_api_key=keys["CMC_API_KEY"],
                seen_ids=seen_ids,
            )
        except requests.RequestException as exc:
            logger.error("Chyba při stahování zpráv / tržních dat: %s", exc)
        except ccxt.BaseError as exc:
            logger.error("Chyba při komunikaci s burzou Binance: %s", exc)
        except Exception as exc:
            logger.error("Neočekávaná chyba v hlavní smyčce bota: %s", exc)

        logger.info("Čekám %d sekund do další kontroly pozic a zpráv...", poll_interval)
        time.sleep(poll_interval)


if __name__ == "__main__":
    run_bot()

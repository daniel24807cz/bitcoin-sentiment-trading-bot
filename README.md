# Bitcoin Sentiment Trading Bot (`BTC/USDC`)

An algorithmic trading bot in Python for automated **`BTC/USDC`** trading on **Binance (Cross Margin)** powered by **Google Gemini** news sentiment analysis, enriched with **CoinMarketCap API** market context and technical filters (**RSI** and **EMA**).

> [!WARNING]
> **Risk Disclaimer:** This project is provided for educational and experimental purposes only. Cryptocurrency trading (especially with margin/leverage) carries significant financial risk. The author assumes no responsibility for any financial losses.

---

## Key Features

- **EU MiCA Compatible:** Trades `BTC/USDC` via Binance Cross Margin (`ccxt.binance` with `defaultType = "margin"`), supporting both **LONG** and **SHORT** positions.
- **Bitcoin RSS News Feeds:** Continuously fetches Bitcoin news from public RSS sources:
  - Cointelegraph (`https://cointelegraph.com/rss/tag/bitcoin`)
  - CoinDesk (`https://www.coindesk.com/arc/outboundfeeds/rss/`)
  - Decrypt (`https://decrypt.co/feed`)
- **CoinMarketCap Market Context:** Attaches live BTC price, 1h/24h percentage change, and 24h volume change to each news item (with a 5-minute cache `CMC_CACHE_TTL_SECONDS = 300` to conserve API credits).
- **AI Sentiment Analysis (Google Gemini):** Evaluates news impact on BTC price (`POSITIVE`, `NEGATIVE`, `NEUTRAL`) and assigns a signal strength score `1–10` (default entry threshold: `MIN_SENTIMENT_SCORE = 7`). Uses `gemini-2.5-flash` with automatic fallback models.
- **Technical Entry Filters:**
  - **15m RSI (14):** Standard signals (`7–8/10`) are filtered out when overbought (`RSI > 65`) for LONG or oversold (`RSI < 38`) for SHORT. Strong signals (`9–10/10`) allow a wider band (`28–72`).
  - **1h EMA (50):** Trend filter — LONG entries are only allowed above `1h EMA 50`, SHORT entries only below `1h EMA 50`.
- **Risk & Position Management:**
  - Maximum **1 open position** at a time.
  - **Dynamic position sizing & leverage:** Uses `90%` of free USDC balance (`TRADE_CAPITAL_RATIO = 0.90`, fallback `21.0 USDC` when balance `< 10 USDC`). Currently runs with **~1x effective leverage** (no leverage multiplier is applied on top of free USDC; Binance Cross Margin is used solely to enable both LONG and SHORT positions).
  - **Stop-Loss:** `1.3%` against the trade direction (`STOP_LOSS_PCT = 0.013`).
  - **Take-Profit:** `2.0%` in the trade direction (`TAKE_PROFIT_PCT = 0.02`).
  - **Trailing Stop-Loss:** Activates at `+0.8%` profit (`TRAILING_ACTIVATION_PCT = 0.008`), locks in at least `+0.3%` profit (`PROFIT_LOCK_PCT = 0.003`), and trails `0.5%` behind the best price (exchange order updates when SL improves by at least `0.2%`).
  - Live position and PnL logging every `60` seconds (`POLL_INTERVAL_SECONDS = 60`).

---

## Live Trading Performance (Production Logs)

> [!NOTE]
> **Test Capital & Leverage Context:** For live testing purposes, the bot was funded with a small test balance of only **~24–28 USDC** (position sizes ranged between **~20.38 and ~25.83 USDC**, using `90%` of free USDC per trade) and **~1x effective leverage** (no leverage multiplier). Because of the small test capital and conservative `~1x` sizing, absolute USDC profits are naturally modest, while the percentage return and `78.6%` win rate demonstrate the strategy working as intended.

Below is the verified summary of all **14 closed trades** recorded in Railway production logs (`2026-11-17` – `2026-11-23`):

| Metric | Value |
| :--- | :--- |
| **Total Closed Trades** | `14` (`10` SHORT, `4` LONG) |
| **Winning Trades** | `11` (`78.6%` win rate) |
| **Losing Trades** | `3` (`21.4%`) |
| **Exit Breakdown** | `2` Take-Profit, `9` Trailing Stop-Loss (in profit), `3` Stop-Loss |
| **Gross Profit / Gross Loss** | `+2.37 USDC` / `-1.25 USDC` |
| **Net Realized PnL** | **`+1.12 USDC`** |

<details>
<summary>View full closed trade history (14 trades)</summary>

| Date (UTC) | Side | Size (BTC) | Entry → Exit (USDC) | Exit Reason | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `2026-11-17 09:40` | SHORT | `0.00031` | `~64,676.22 → ~64,469.48` | Trailing Stop-Loss (profit) | `+0.06 USDC (+0.32%)` |
| `2026-11-17 15:56` | SHORT | `0.00032` | `~64,193.04 → 63,724.44` | Trailing Stop-Loss (profit) | `+0.15 USDC (+0.73%)` |
| `2026-11-17 23:19` | SHORT | `0.00032` | `~63,698.34 → ~64,972.31` | Stop-Loss | `-0.41 USDC (-2.00%)` |
| `2026-11-18 14:39` | LONG | `0.00037` | `~64,522.23 → ~63,156.09` | Stop-Loss | `-0.51 USDC (-2.12%)` |
| `2026-11-18 22:37` | SHORT | `0.00034` | `~63,174.87 → 62,563.44` | Trailing Stop-Loss (profit) | `+0.20 USDC (+0.97%)` |
| `2026-11-19 16:11` | SHORT | `0.00034` | `~63,604.79 → 63,130.40` | Trailing Stop-Loss (profit) | `+0.16 USDC (+0.75%)` |
| `2026-11-19 20:47` | SHORT | `0.00034` | `~62,695.52 → 62,121.60` | Trailing Stop-Loss (profit) | `+0.19 USDC (+0.92%)` |
| `2026-11-20 05:25` | SHORT | `0.00034` | `~62,002.94 → 61,657.16` | Trailing Stop-Loss (profit) | `+0.12 USDC (+0.56%)` |
| `2026-11-20 12:27` | SHORT | `0.00034` | `~61,869.01 → 61,596.30` | Trailing Stop-Loss (profit) | `+0.09 USDC (+0.44%)` |
| `2026-11-20 15:22` | SHORT | `0.00035` | `~61,378.50 → 60,138.44` | Take-Profit | `+0.43 USDC (+2.02%)` |
| `2026-11-22 04:01` | SHORT | `0.00038` | `~59,973.36 → 59,510.91` | Trailing Stop-Loss (profit) | `+0.17 USDC (+0.77%)` |
| `2026-11-23 13:57` | LONG | `0.00041` | `~61,186.95 → 62,929.69` | Take-Profit | `+0.72 USDC (+2.85%)` |
| `2026-11-23 15:33` | LONG | `0.00041` | `~62,994.34 → ~63,183.59` | Trailing Stop-Loss (profit) | `+0.08 USDC (+0.30%)` |
| `2026-11-23 17:08` | LONG | `0.00041` | `~62,544.16 → ~61,731.09` | Stop-Loss | `-0.33 USDC (-1.30%)` |

</details>

---

## Project Structure

```text
bitcoin-sentiment-trading-bot/
├── .github/
│   └── workflows/
│       └── tests.yml       # GitHub Actions workflow for automated unit tests
├── .env.example            # Environment variable template (API keys)
├── .gitignore              # Ignored files (venv, .env, backup, __pycache__)
├── CONTRIBUTING.md         # Contribution guidelines
├── LICENSE                 # MIT License
├── Procfile                # Process configuration for Railway / PaaS
├── README.md               # Project documentation
├── SECURITY.md             # Security policy and best practices
├── bot.py                  # Core trading bot logic
├── railway.json            # Railway build and deployment configuration
├── requirements.txt        # Python dependencies
└── test_bot.py             # Unit and integration tests with mocking
```

---

## Requirements & Installation

- **Python 3.11+**
- A **Binance** account (with Cross Margin enabled and an API key)
- A **Google Gemini** API key
- A **CoinMarketCap** API key (Free tier is sufficient)

### Local Setup

1. Clone the repository and enter the project directory:
   ```bash
   git clone <REPOSITORY_URL>
   cd bitcoin-sentiment-trading-bot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # Linux / macOS:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your environment variables using `.env.example` as a reference.

5. Run the bot:
   ```bash
   python bot.py
   ```

---

## Environment Variables

| Variable | Description |
| :--- | :--- |
| `BINANCE_API_KEY` | Binance API key (recommended without withdrawal permissions) |
| `BINANCE_SECRET` | Binance API secret |
| `CMC_API_KEY` | CoinMarketCap API key (also supports `COINMARKETCAP_API_KEY` or `NEWS_API_KEY`) |
| `GEMINI_API_KEY` | Google Gemini API key for sentiment analysis |

---

## Deploying to Railway

The project is ready for 24/7 deployment on **Railway**:
- `railway.json` configures `NIXPACKS` build, start command `python bot.py`, and `ON_FAILURE` restart policy (max 10 retries).
- `Procfile` defines `worker: python bot.py`.
- Set the 4 environment variables above in your Railway service **Variables** tab.

---

## Testing

The repository includes 21 unit and integration tests in `test_bot.py` with full mocking of external services (Binance, Gemini, RSS feeds, CoinMarketCap).

Run the test suite:
```bash
python -m unittest test_bot.py -v
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

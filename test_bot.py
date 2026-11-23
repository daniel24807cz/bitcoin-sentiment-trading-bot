"""
Testovací sada (unit a integrační testy s mockováním) pro ověření
všech komponent algotradingového bota (bot.py) s Bitcoin RSS + CoinMarketCap Free API,
párem BTC/USDC, Stop-Loss (2 %), Take-Profit (2 %) a dynamickým výpočtem velikosti pozice z TRADE_INVESTMENT_USDT = 21.0.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import bot


class TestTradingBot(unittest.TestCase):
    """Testy pro BTC/USDC Sentiment Algotrading Bota."""

    def setUp(self):
        bot._cmc_cache = {"timestamp": 0.0, "context": ""}
        bot._reset_position_state("short")
        bot._reset_position_state("long")
        bot._position_state["idle_since"] = None
        bot._position_state["last_article_title"] = None
        bot._position_state["last_sentiment"] = None
        self.env_vars = {
            "BINANCE_API_KEY": "test_binance_key",
            "BINANCE_SECRET": "test_binance_secret",
            "CMC_API_KEY": "test_cmc_key",
            "GEMINI_API_KEY": "test_gemini_key",
        }

    def test_load_api_keys_success(self):
        """Ověří bezpečné načtení všech 4 API klíčů z os.environ."""
        with patch.dict(os.environ, self.env_vars, clear=True):
            keys = bot.load_api_keys()
            self.assertEqual(keys["BINANCE_API_KEY"], "test_binance_key")
            self.assertEqual(keys["BINANCE_SECRET"], "test_binance_secret")
            self.assertEqual(keys["CMC_API_KEY"], "test_cmc_key")
            self.assertEqual(keys["GEMINI_API_KEY"], "test_gemini_key")

    def test_load_api_keys_missing_raises(self):
        """Ověří vyhození výjimky, pokud některý z klíčů v os.environ chybí."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(EnvironmentError):
                bot.load_api_keys()

    def test_create_exchange_config(self):
        """Ověří nastavení ccxt.binance, pár BTC/USDC a TRADE_INVESTMENT_USDT = 21.0."""
        exchange = bot.create_exchange("key123", "secret123")
        self.assertEqual(exchange.options.get("defaultType"), "margin")
        self.assertEqual(bot.SYMBOL, "BTC/USDC")
        self.assertEqual(bot.TRADE_INVESTMENT_USDT, 21.0)

    @patch("bot.BTC_RSS_FEEDS", ["https://cointelegraph.com/rss/tag/bitcoin"])
    @patch("bot.requests.get")
    def test_fetch_bitcoin_news_and_cmc_context(self, mock_get):
        """Ověří stažení zpráv z Bitcoin RSS a tržního kontextu z CoinMarketCap Free API."""
        sample_rss = b"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>Bitcoin hits new high as institutional adoption grows</title>
              <link>https://cointelegraph.com/news/btc-high</link>
              <description>&lt;p&gt;Major funds increase BTC holdings significantly.&lt;/p&gt;</description>
            </item>
          </channel>
        </rss>"""

        mock_rss_resp = MagicMock()
        mock_rss_resp.content = sample_rss

        mock_cmc_resp = MagicMock()
        mock_cmc_resp.json.return_value = {
            "data": {
                "BTC": [
                    {
                        "quote": {
                            "USD": {
                                "price": 92500.0,
                                "percent_change_1h": 0.75,
                                "percent_change_24h": 3.20,
                                "volume_change_24h": 12.50,
                            }
                        }
                    }
                ]
            }
        }

        mock_get.side_effect = [mock_rss_resp, mock_cmc_resp]

        articles = bot.fetch_bitcoin_news("test_cmc_key", limit=5)
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["title"], "Bitcoin hits new high as institutional adoption grows")
        self.assertIn("92500.00 USD", articles[0]["market_context"])

    @patch("bot.BTC_RSS_FEEDS", [
        "https://cointelegraph.com/rss/tag/bitcoin",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    ])
    @patch("bot.requests.get")
    def test_fetch_bitcoin_news_aggregates_and_filters_btc(self, mock_get):
        """Ověří sloučení více RSS zdrojů a odfiltrování článků mimo Bitcoin na obecných feedech."""
        ct_rss = b"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0"><channel><item>
          <title>Bitcoin miners expand capacity</title>
          <link>https://cointelegraph.com/news/ct-1</link>
          <description>Miners add power.</description>
        </item></channel></rss>"""
        cd_rss = b"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0"><channel>
          <item>
            <title>Dogecoin meme rally fades</title>
            <link>https://coindesk.com/doge-1</link>
            <description>Altcoin news.</description>
          </item>
          <item>
            <title>Institutional BTC inflows accelerate</title>
            <link>https://coindesk.com/btc-2</link>
            <description>Spot ETF demand rises.</description>
          </item>
        </channel></rss>"""

        r1, r2 = MagicMock(), MagicMock()
        r1.content = ct_rss
        r2.content = cd_rss
        mock_get.side_effect = [r1, r2]

        articles = bot.fetch_bitcoin_news("", limit=5)
        titles = [a["title"] for a in articles]
        self.assertEqual(titles, [
            "Bitcoin miners expand capacity",
            "Institutional BTC inflows accelerate",
        ])

    def test_analyze_sentiment_prompt_and_outputs(self):
        """Ověří přesné znění promptu pro Gemini a parsování POSITIVE, NEGATIVE, NEUTRAL."""
        expected_prompt_prefix = (
            "Analyzuj tuto zprávu o Bitcoinu a odpověz striktně pouze jedním slovem "
            "s ohledem na vliv na jeho cenu: POSITIVE, NEGATIVE, nebo NEUTRAL."
        )
        self.assertEqual(bot.SENTIMENT_PROMPT, expected_prompt_prefix)

        mock_model = MagicMock()

        for expected in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
            mock_model.generate_content.return_value = MagicMock(text=f"  {expected}\n")
            result = bot.analyze_sentiment(mock_model, "Testovací zpráva o Bitcoinu")
            self.assertEqual(result, expected)
            called_prompt = mock_model.generate_content.call_args.args[0]
            self.assertTrue(called_prompt.startswith(expected_prompt_prefix))

    def test_calculate_stop_loss_and_take_profit_price(self):
        """Ověří výpočet 1.3% Stop-Loss a 2% Take-Profit ceny."""
        self.assertEqual(bot.calculate_stop_loss_price(100000.0, "buy"), 98700.0)
        self.assertEqual(bot.calculate_stop_loss_price(100000.0, "sell"), 101300.0)
        self.assertEqual(bot.calculate_take_profit_price(100000.0, "buy"), 102000.0)
        self.assertEqual(bot.calculate_take_profit_price(100000.0, "sell"), 98000.0)

    def test_execute_trade_positive_long(self):
        """Ověří dynamický výpočet množství (21 USDC / cena), amount_to_precision, Market Buy (LONG), SL (-1.3 %) a TP (+2 %)."""
        mock_exchange = MagicMock()
        mock_exchange.options = {"defaultType": "margin"}
        mock_exchange.fetch_ticker.return_value = {"last": 70000.0}
        mock_exchange.amount_to_precision.return_value = "0.0003"
        mock_exchange.create_order.side_effect = [
            {"id": "order_1", "average": 70000.0},
            {"id": "sl_1"},
        ]

        result = bot.execute_trade(mock_exchange, "POSITIVE")
        self.assertIsNotNone(result)
        self.assertEqual(result["position"], "LONG")
        self.assertEqual(result["amount"], 0.0003)
        self.assertEqual(result["entry_price"], 70000.0)
        self.assertEqual(result["stop_price"], 69090.0)
        self.assertEqual(result["take_profit_price"], 71400.0)

        mock_exchange.load_markets.assert_called_once()
        mock_exchange.fetch_ticker.assert_called_once_with("BTC/USDC")
        mock_exchange.amount_to_precision.assert_called_once_with("BTC/USDC", 21.0 / 70000.0)

    def test_execute_trade_negative_short(self):
        """Ověří dynamický výpočet množství (21 USDC / cena), amount_to_precision, Market Sell (SHORT), SL (+1.3 %) a TP (-2 %)."""
        mock_exchange = MagicMock()
        mock_exchange.options = {"defaultType": "margin"}
        mock_exchange.fetch_ticker.return_value = {"last": 84000.0}
        mock_exchange.amount_to_precision.return_value = "0.00025"
        mock_exchange.create_order.side_effect = [
            {"id": "order_2", "average": 84000.0},
            {"id": "sl_2"},
        ]

        result = bot.execute_trade(mock_exchange, "NEGATIVE")
        self.assertIsNotNone(result)
        self.assertEqual(result["position"], "SHORT")
        self.assertEqual(result["amount"], 0.00025)
        self.assertEqual(result["entry_price"], 84000.0)
        self.assertEqual(result["stop_price"], 85092.0)
        self.assertEqual(result["take_profit_price"], 82320.0)

    def test_execute_trade_neutral_does_nothing(self):
        """Ověří, že při NEUTRAL sentimentu bot neotevírá žádný obchod."""
        mock_exchange = MagicMock()
        result = bot.execute_trade(mock_exchange, "NEUTRAL")
        self.assertIsNone(result)
        mock_exchange.load_markets.assert_not_called()
        mock_exchange.create_order.assert_not_called()

    def test_monitor_open_positions_triggers_take_profit(self):
        """Ověří, že při poklesu ceny o >= 2 % u SHORT pozice bot automaticky uzavře pozici v zisku."""
        mock_exchange = MagicMock()
        mock_exchange.options = {"defaultType": "margin"}
        mock_exchange.fetch_balance.return_value = {
            "info": {
                "userAssets": [
                    {"asset": "BTC", "borrowed": "0.00128", "free": "0.0"},
                    {"asset": "USDC", "borrowed": "0.0", "free": "23.13"},
                ]
            }
        }
        # Vstup ~65490 -> SL 66800 -> TP 64180; aktuální cena 64000 (pod TP)
        mock_exchange.fetch_ticker.return_value = {"last": 64000.0}
        mock_exchange.fetch_open_orders.return_value = [
            {"id": "sl_1", "side": "buy", "stopPrice": 66800.0}
        ]
        mock_exchange.amount_to_precision.return_value = "0.00128"

        bot.monitor_open_positions_and_take_profit(mock_exchange)

        mock_exchange.cancel_order.assert_called_once_with(
            "sl_1", "BTC/USDC", params={"marginMode": "cross"}
        )
        mock_exchange.create_order.assert_called_once_with(
            symbol="BTC/USDC",
            type="market",
            side="buy",
            amount=0.00128,
            params={"marginMode": "cross", "sideEffectType": "AUTO_BORROW_REPAY"},
        )

    @patch("bot.requests.get")
    def test_fetch_cmc_market_context_uses_5min_cache(self, mock_get):
        """Ověří, že CoinMarketCap API se volá maximálně jednou za 5 minut díky cache."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {"BTC": [{"quote": {"USD": {"price": 65000.0, "percent_change_1h": 0.1, "percent_change_24h": 1.0, "volume_change_24h": 5.0}}}]}
        }
        mock_get.return_value = mock_resp

        ctx1 = bot.fetch_cmc_market_context("test_cmc_key")
        ctx2 = bot.fetch_cmc_market_context("test_cmc_key")

        self.assertEqual(ctx1, ctx2)
        self.assertIn("65000.00 USD", ctx1)
        mock_get.assert_called_once()

    def test_trailing_stop_locks_profit_and_closes_on_reversal(self):
        """Ověří, že po dosažení +0.8 % zisku se Stop-Loss posune do zisku a při obratu pozici uzavře v zisku."""
        mock_exchange = MagicMock()
        mock_exchange.options = {"defaultType": "margin"}
        mock_exchange.fetch_balance.return_value = {
            "info": {
                "userAssets": [
                    {"asset": "BTC", "borrowed": "0.00128", "free": "0.0"},
                    {"asset": "USDC", "borrowed": "0.0", "free": "21.38"},
                ]
            }
        }
        mock_exchange.fetch_my_trades.return_value = [
            {"side": "sell", "amount": 0.00128, "price": 65500.0}
        ]
        mock_exchange.amount_to_precision.return_value = "0.00128"
        mock_exchange.price_to_precision.side_effect = lambda _sym, p: f"{p:.2f}"

        # Krok 1: Cena klesla na 64500 (+1.53 % zisk) -> posune SL z 66810 na 64822.50
        mock_exchange.fetch_ticker.return_value = {"last": 64500.0}
        mock_exchange.fetch_open_orders.return_value = [
            {"id": "sl_old", "side": "buy", "stopPrice": 66810.0}
        ]
        bot.monitor_open_positions_and_take_profit(mock_exchange)

        mock_exchange.cancel_order.assert_called_once_with(
            "sl_old", "BTC/USDC", params={"marginMode": "cross"}
        )
        mock_exchange.create_order.assert_called_once_with(
            symbol="BTC/USDC",
            type="STOP_LOSS_LIMIT",
            side="buy",
            amount=0.00128,
            price=64822.5,
            params={
                "stopPrice": 64822.5,
                "marginMode": "cross",
                "sideEffectType": "AUTO_BORROW_REPAY",
            },
        )
        self.assertEqual(bot._position_state["short_entry_price"], 65500.0)
        self.assertEqual(bot._position_state["short_trailing_sl"], 64822.5)

        # Krok 2: Cena se otočí zpět nahoru na 64850 -> bot okamžitě zavře v zisku (místo čekání na -2 % SL)
        mock_exchange.reset_mock()
        mock_exchange.fetch_ticker.return_value = {"last": 64850.0}
        mock_exchange.fetch_open_orders.return_value = [
            {"id": "sl_new", "side": "buy", "stopPrice": 64822.5}
        ]
        bot.monitor_open_positions_and_take_profit(mock_exchange)

        mock_exchange.cancel_order.assert_called_once_with(
            "sl_new", "BTC/USDC", params={"marginMode": "cross"}
        )
        mock_exchange.create_order.assert_called_once_with(
            symbol="BTC/USDC",
            type="market",
            side="buy",
            amount=0.00128,
            params={"marginMode": "cross", "sideEffectType": "AUTO_BORROW_REPAY"},
        )
        self.assertIsNone(bot._position_state["short_entry_price"])

    @patch("bot.fetch_bitcoin_news")
    def test_process_latest_news_pipeline(self, mock_fetch):
        """Integrační test celého toku: stažení zpráv + CMC kontextu -> Gemini -> exekuce obchodu + deduplikace."""
        mock_fetch.return_value = [
            {
                "id": "rss_2001",
                "title": "Bitcoin ETF sees record inflows",
                "body": "Demand surges across institutional investors.",
                "market_context": "Tržní kontext BTC (CoinMarketCap): cena 95000.00 USD, změna 24h: +4.10%.",
                "url": "https://cointelegraph.com/news/btc-etf",
            },
            {
                "id": "rss_2002",
                "title": "Bitcoin network difficulty adjustment unchanged",
                "body": "Routine update.",
                "market_context": "Tržní kontext BTC (CoinMarketCap): cena 95000.00 USD, změna 24h: +4.10%.",
                "url": "https://cointelegraph.com/news/btc-diff",
            },
        ]

        mock_exchange = MagicMock()
        mock_exchange.options = {"defaultType": "margin"}
        mock_exchange.fetch_ticker.return_value = {"last": 95000.0}
        mock_exchange.amount_to_precision.return_value = "0.00022"
        mock_exchange.create_order.side_effect = [
            {"id": "mkt_1", "average": 95000.0},
            {"id": "sl_1"},
        ]

        mock_model = MagicMock()
        mock_model.generate_content.side_effect = [
            MagicMock(text="POSITIVE"),
            MagicMock(text="NEUTRAL"),
        ]

        seen_ids = set()
        trades = bot.process_latest_news(
            exchange=mock_exchange,
            model=mock_model,
            cmc_api_key="test_cmc_key",
            seen_ids=seen_ids,
        )

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["position"], "LONG")
        self.assertEqual(trades[0]["amount"], 0.00022)
        self.assertEqual(trades[0]["stop_price"], 93765.0)
        self.assertEqual(trades[0]["take_profit_price"], 96900.0)
        self.assertEqual(len(seen_ids), 2)

    def test_execute_trade_skips_when_position_already_open(self):
        """Ověří, že při už otevřené SHORT nebo LONG pozici bot neotevře další obchod."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_balance.return_value = {
            "info": {"userAssets": [{"asset": "BTC", "free": "0.0", "borrowed": "0.00032"}]}
        }
        result = bot.execute_trade(mock_exchange, "NEGATIVE")
        self.assertIsNone(result)
        mock_exchange.create_order.assert_not_called()

    @patch("bot.fetch_bitcoin_news")
    def test_process_latest_news_skips_gemini_when_position_open(self, mock_fetch):
        """Ověří, že při otevřené pozici bot označí články jako přečtené, ale nevolá Gemini ani neotevírá další obchod."""
        mock_fetch.return_value = [
            {
                "id": "rss_open_1",
                "title": "Bitcoin drops further",
                "body": "Selloff continues.",
                "url": "https://cointelegraph.com/news/btc-drop",
            }
        ]
        mock_exchange = MagicMock()
        mock_exchange.fetch_balance.return_value = {
            "info": {"userAssets": [{"asset": "BTC", "free": "0.00032", "borrowed": "0.0"}]}
        }
        mock_model = MagicMock()
        seen_ids = set()

        trades = bot.process_latest_news(
            exchange=mock_exchange,
            model=mock_model,
            cmc_api_key="test_cmc_key",
            seen_ids=seen_ids,
        )

        self.assertEqual(trades, [])
        self.assertIn("rss_open_1", seen_ids)
        mock_model.generate_content.assert_not_called()
        mock_exchange.create_order.assert_not_called()

    def test_locked_btc_in_stop_loss_detected_as_open_long(self):
        """Ověří, že BTC zamčené v Stop-Loss SELL příkazu (locked) se stále detekuje jako otevřená LONG pozice."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_balance.return_value = {
            "info": {"userAssets": [{"asset": "BTC", "free": "0.0", "locked": "0.00032", "borrowed": "0.0"}]}
        }
        self.assertTrue(bot.has_open_btc_position(mock_exchange))
        result = bot.execute_trade(mock_exchange, "NEGATIVE")
        self.assertIsNone(result)
        mock_exchange.create_order.assert_not_called()

    @patch("bot.fetch_bitcoin_news")
    def test_analyze_sentiment_with_score_skips_weak_signal(self, mock_fetch):
        """Ověří, že slabý signál se skóre pod MIN_SENTIMENT_SCORE (např. NEGATIVE 4) neotevře obchod."""
        mock_fetch.return_value = [
            {
                "id": "rss_weak_1",
                "title": "Bitcoin holds steady in quiet trading",
                "body": "Routine market recap.",
                "url": "https://cointelegraph.com/news/quiet",
            }
        ]
        mock_exchange = MagicMock()
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text="NEGATIVE 4")
        seen_ids = set()

        trades = bot.process_latest_news(
            exchange=mock_exchange,
            model=mock_model,
            cmc_api_key="test_cmc_key",
            seen_ids=seen_ids,
        )

        self.assertEqual(trades, [])
        mock_exchange.create_order.assert_not_called()

    def test_rsi_filter_blocks_overbought_long_and_oversold_short(self):
        """Ověří, že RSI filtr zablokuje LONG při RSI > 72 a SHORT při RSI < 28."""
        # Silný růst -> RSI bude ~100 (> 72)
        up_candles = [[0, 0, 0, 0, 60000.0 + i * 200.0] for i in range(25)]
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = up_candles
        self.assertIsNone(bot.execute_trade(mock_exchange, "POSITIVE"))
        mock_exchange.create_order.assert_not_called()

        # Silný pokles -> RSI bude ~0 (< 38)
        down_candles = [[0, 0, 0, 0, 70000.0 - i * 200.0] for i in range(25)]
        mock_exchange.fetch_ohlcv.return_value = down_candles
        self.assertIsNone(bot.execute_trade(mock_exchange, "NEGATIVE"))
        mock_exchange.create_order.assert_not_called()

    def test_ema_trend_filter_blocks_counter_trend_trades(self):
        """Ověří, že trend filtr (1h EMA50) zablokuje LONG pod EMA50 a SHORT nad EMA50."""
        # Neutrální 15m svíčky (RSI ~ 50) + 1h svíčky kolem 65000 (EMA50 = 65000)
        neutral_15m = [[0, 0, 0, 0, 65000.0 + (10.0 if i % 2 == 0 else -10.0)] for i in range(25)]
        flat_1h = [[0, 0, 0, 0, 65000.0] for _ in range(60)]

        mock_exchange = MagicMock()
        mock_exchange.options = {"defaultType": "margin"}

        # 1. Aktuální cena 64000 (< EMA50 65000) -> LONG proti klesajícímu trendu zablokován
        mock_exchange.fetch_ohlcv.side_effect = [neutral_15m, flat_1h]
        mock_exchange.fetch_ticker.return_value = {"last": 64000.0}
        self.assertIsNone(bot.execute_trade(mock_exchange, "POSITIVE"))
        mock_exchange.create_order.assert_not_called()

        # 2. Aktuální cena 66000 (> EMA50 65000) -> SHORT proti rostoucímu trendu zablokován
        mock_exchange.fetch_ohlcv.side_effect = [neutral_15m, flat_1h]
        mock_exchange.fetch_ticker.return_value = {"last": 66000.0}
        self.assertIsNone(bot.execute_trade(mock_exchange, "NEGATIVE"))
        mock_exchange.create_order.assert_not_called()

    @patch("bot.fetch_btc_rsi", return_value=33.0)
    def test_strong_signal_allows_wider_rsi_and_dynamic_investment(self, _mock_rsi):
        """Ověří širší RSI pásmo pro silný signál (9/10) a dynamickou velikost obchodu 90 % volných USDC."""
        mock_exchange = MagicMock()
        mock_exchange.options = {"defaultType": "margin"}
        mock_exchange.fetch_balance.return_value = {
            "info": {"userAssets": [{"asset": "BTC", "free": "0.0", "borrowed": "0.0"}, {"asset": "USDC", "free": "25.00", "borrowed": "0.0"}]}
        }
        mock_exchange.fetch_ticker.return_value = {"last": 60000.0}
        mock_exchange.amount_to_precision.return_value = "0.00037"
        mock_exchange.create_order.side_effect = [
            {"id": "mkt_dyn", "average": 60000.0},
            {"id": "sl_dyn"},
        ]

        # Standardní síla 8/10 -> RSI < 38 zablokuje
        self.assertIsNone(bot.execute_trade(mock_exchange, "NEGATIVE", signal_score=8))
        mock_exchange.create_order.assert_not_called()

        # Silná zpráva 9/10 -> projde a investuje 90 % z 25.00 = 22.50 USDC
        result = bot.execute_trade(mock_exchange, "NEGATIVE", signal_score=9)
        self.assertIsNotNone(result)
        mock_exchange.amount_to_precision.assert_called_once_with("BTC/USDC", 22.50 / 60000.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

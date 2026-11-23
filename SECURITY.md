# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this repository (such as potential credential exposure or a flaw in trading logic), please do **not** open a public Issue. Instead, contact the repository maintainer privately or use GitHub Security Advisories.

## Best Practices for Users

- **Never commit your `.env` file** or any API keys (`BINANCE_API_KEY`, `BINANCE_SECRET`, `CMC_API_KEY`, `GEMINI_API_KEY`).
- On Binance, restrict API key permissions to reading and spot/margin trading only (disable withdrawals), and configure an IP whitelist whenever possible.
- Always verify behavior with unit tests and small capital amounts before running with live funds.

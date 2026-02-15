"""
Kalshi Market Scanner: Fetches prediction markets from Kalshi.

Kalshi is a US-regulated prediction market (CFTC-regulated).
Public market data endpoints require NO authentication.
Trading requires API keys + RSA signature auth.

API base: https://api.elections.kalshi.com/trade-api/v2
(Despite "elections" in URL, this covers ALL Kalshi markets)

Market data returns: ticker, title, yes_bid, yes_ask, volume, 
close_time, status, category, etc.

Prices are in CENTS (1-99), not decimals like Polymarket.
"""

import json
import time
import structlog
import httpx
from datetime import datetime, timezone
from typing import Optional

from models import Config, Market

log = structlog.get_logger()

KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiScanner:
    """Fetches and parses markets from Kalshi's public API."""

    def __init__(self, config: Config):
        self.config = config
        self.client = httpx.Client(
            timeout=30.0,
            headers={"Accept": "application/json"},
        )

    def fetch_active_markets(self, max_markets: int = 500) -> list[Market]:
        """
        Fetch open markets from Kalshi.
        Returns Market objects compatible with the same pipeline as Polymarket.
        """
        log.info("kalshi_scanning", max_markets=max_markets)

        raw_markets = self._fetch_markets(max_markets)
        log.info("kalshi_raw_fetched", count=len(raw_markets))

        markets = []
        reject_reasons = {"no_price": 0, "low_volume": 0, "extreme_price": 0, "parse_error": 0}
        for raw in raw_markets:
            try:
                market, reason = self._parse_market(raw)
                if market:
                    markets.append(market)
                elif reason:
                    reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            except Exception as e:
                reject_reasons["parse_error"] += 1
                log.debug("kalshi_parse_failed", ticker=raw.get("ticker", "?"), error=str(e))
                continue

        log.info("kalshi_markets_parsed", count=len(markets), rejected=reject_reasons)
        return markets

    def _fetch_markets(self, max_markets: int) -> list[dict]:
        """Fetch markets from Kalshi API across multiple categories."""
        all_markets = []
        seen_tickers = set()

        # Known high-value series tickers for crypto and other categories
        # From Kalshi's market pages: kxbtcmaxy, kxbtcmaxm, kxbtcminy, etc.
        # The API param is series_ticker (not series)
        crypto_series = [
            "KXBTCMAXY", "KXBTCMAXM", "KXBTCMINY",   # BTC yearly/monthly highs/lows
            "KXBTCMAXM", "KXBTCMINM",                  # BTC monthly
            "KXBTC15M",                                  # BTC 15-min
            "KXBTCMAX150",                               # BTC milestones
            "KXETHMAXM", "KXETHMINM",                   # ETH monthly
            "KXETHMAXY", "KXETHMINY",                   # ETH yearly
            "KXSOLMAXM",                                 # SOL monthly
        ]

        strategies = [
            # General open markets (gets a mix of everything)
            {"status": "open", "limit": 200},
        ]

        # Add crypto series queries
        for series in crypto_series:
            strategies.append({"status": "open", "limit": 100, "series_ticker": series})

        for params in strategies:
            if len(all_markets) >= max_markets:
                break

            cursor = None
            pages = 0
            max_pages = 3

            while pages < max_pages and len(all_markets) < max_markets:
                req_params = {**params}
                if cursor:
                    req_params["cursor"] = cursor

                try:
                    resp = self.client.get(
                        f"{KALSHI_API_BASE}/markets",
                        params=req_params,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    log.debug("kalshi_fetch_failed", error=str(e), series=params.get("series_ticker", "general"))
                    break

                batch = data.get("markets", [])
                if not batch:
                    break

                added = 0
                for m in batch:
                    ticker = m.get("ticker", "")
                    if ticker and ticker not in seen_tickers:
                        seen_tickers.add(ticker)
                        all_markets.append(m)
                        added += 1

                if added > 0 and params.get("series_ticker"):
                    log.info("kalshi_series_fetched", series=params["series_ticker"], markets=added)

                cursor = data.get("cursor", "")
                if not cursor:
                    break
                pages += 1

        return all_markets

    def _parse_market(self, raw: dict) -> tuple[Optional[Market], Optional[str]]:
        """Convert Kalshi market data into our Market model. Returns (market, reject_reason)."""
        ticker = raw.get("ticker", "")
        title = raw.get("title", "")
        status = raw.get("status", "")

        if status not in ("open", "active"):
            return None, None  # Silently skip closed markets

        # Kalshi prices: try multiple fields with proper fallback
        # Fields available: yes_bid, yes_ask, yes_bid_dollars, yes_ask_dollars,
        # last_price, last_price_dollars, no_bid, no_ask
        yes_bid = None
        yes_ask = None

        # Try dollar string fields first (e.g., "0.5600")
        ybd = raw.get("yes_bid_dollars")
        if ybd and ybd != "0.0000":
            try:
                yes_bid = float(ybd)
            except (ValueError, TypeError):
                pass

        yad = raw.get("yes_ask_dollars")
        if yad and yad != "0.0000":
            try:
                yes_ask = float(yad)
            except (ValueError, TypeError):
                pass

        # Try cent integer fields (1-99)
        if yes_bid is None or yes_bid <= 0:
            yb_cents = raw.get("yes_bid", 0)
            if yb_cents and int(yb_cents) > 0:
                yes_bid = int(yb_cents) / 100.0

        if yes_ask is None or yes_ask <= 0:
            ya_cents = raw.get("yes_ask", 0)
            if ya_cents and int(ya_cents) > 0:
                yes_ask = int(ya_cents) / 100.0

        # Fallback: derive from NO side (no_bid at X means yes_ask at 1-X)
        if yes_bid is None or yes_bid <= 0:
            no_ask_d = raw.get("no_ask_dollars") or raw.get("no_ask", 0)
            try:
                no_ask_val = float(no_ask_d) if isinstance(no_ask_d, str) else (int(no_ask_d) / 100.0 if no_ask_d else 0)
                if no_ask_val > 0:
                    yes_bid = 1.0 - no_ask_val
            except (ValueError, TypeError):
                pass

        if yes_ask is None or yes_ask <= 0:
            no_bid_d = raw.get("no_bid_dollars") or raw.get("no_bid", 0)
            try:
                no_bid_val = float(no_bid_d) if isinstance(no_bid_d, str) else (int(no_bid_d) / 100.0 if no_bid_d else 0)
                if no_bid_val > 0:
                    yes_ask = 1.0 - no_bid_val
            except (ValueError, TypeError):
                pass

        # Last resort: use last_price to estimate bid/ask
        if yes_bid is None or yes_bid <= 0 or yes_ask is None or yes_ask <= 0:
            last_d = raw.get("last_price_dollars") or raw.get("last_price", 0)
            try:
                last_val = float(last_d) if isinstance(last_d, str) else (int(last_d) / 100.0 if last_d else 0)
                if last_val > 0:
                    yes_bid = yes_bid if (yes_bid and yes_bid > 0) else max(last_val - 0.03, 0.01)
                    yes_ask = yes_ask if (yes_ask and yes_ask > 0) else min(last_val + 0.03, 0.99)
            except (ValueError, TypeError):
                pass

        if not yes_bid or not yes_ask or yes_bid <= 0 or yes_ask <= 0:
            return None, "no_price"

        # Skip markets with no real spread (probably no liquidity)
        if yes_bid == yes_ask:
            yes_bid = max(yes_bid - 0.01, 0.01)
            yes_ask = min(yes_ask + 0.01, 0.99)

        mid = (yes_bid + yes_ask) / 2
        spread = yes_ask - yes_bid

        # Volume
        volume_str = raw.get("volume_24h_fp", raw.get("volume_24h", 0))
        try:
            volume = float(volume_str) if volume_str else 0.0
        except (ValueError, TypeError):
            volume = 0.0

        # Skip truly dead markets (no volume at all)
        total_volume = float(raw.get("volume_fp", raw.get("volume", 0)) or 0)
        if total_volume < 10 and volume < 5:
            return None, "low_volume"

        # Skip extreme prices
        if mid < 0.03 or mid > 0.97:
            return None, "extreme_price"

        # Category
        category = raw.get("category", "")
        subtitle = raw.get("subtitle", "")

        # End date
        close_time = raw.get("close_time", "") or raw.get("expiration_time", "")

        return Market(
            condition_id=f"kalshi_{ticker}",  # Prefix to distinguish from Polymarket
            question=title,
            token_id=ticker,  # Kalshi uses ticker for trading
            best_bid=yes_bid,
            best_ask=yes_ask,
            mid_price=mid,
            spread=spread,
            volume_24h=volume,
            category=category,
            end_date=close_time,
            source="kalshi",  # Tag the source
        ), None

    def _parse_price(self, raw: dict, dollar_key: str, cent_key: str) -> Optional[float]:
        """Parse price from dollar string or cent integer."""
        # Try dollar string first (e.g., "0.5600")
        dollar_val = raw.get(dollar_key)
        if dollar_val:
            try:
                return float(dollar_val)
            except (ValueError, TypeError):
                pass

        # Fall back to cents (e.g., 56 = $0.56)
        cent_val = raw.get(cent_key)
        if cent_val is not None:
            try:
                return int(cent_val) / 100.0
            except (ValueError, TypeError):
                pass

        return None

    def close(self):
        self.client.close()

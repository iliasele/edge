"""Scanner: fetches active markets from Polymarket's CLOB API."""

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from models import Config, Market
from typing import Optional

log = structlog.get_logger()

CLOB_API_BASE = "https://clob.polymarket.com"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"


class MarketScanner:
    """Scans Polymarket for active, liquid markets."""

    def __init__(self, config: Config):
        self.config = config
        self.client = httpx.Client(timeout=30.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def fetch_active_markets(self) -> list[Market]:
        """
        Fetch active markets from Polymarket.
        
        Uses the Gamma API for market metadata and CLOB API for order book data.
        Returns up to config.max_markets markets sorted by volume.
        """
        log.info("scanning_markets", max_markets=self.config.max_markets)

        # Step 1: Get active markets from Gamma API (metadata + discovery)
        markets_raw = self._fetch_gamma_markets()
        log.info("fetched_raw_markets", count=len(markets_raw))

        # Step 2: Filter and enrich with CLOB order book data
        markets = []
        for raw in markets_raw[:self.config.max_markets]:
            try:
                market = self._enrich_with_orderbook(raw)
                if market and self._passes_filters(market):
                    markets.append(market)
            except Exception as e:
                log.warning("market_enrich_failed", market=raw.get("question", "?"), error=str(e))
                continue

        log.info("markets_after_filter", count=len(markets))
        return markets

    def _fetch_gamma_markets(self) -> list[dict]:
        """Fetch market list from Gamma API sorted by volume."""
        all_markets = []
        offset = 0
        limit = 100

        while len(all_markets) < self.config.max_markets:
            resp = self.client.get(
                f"{GAMMA_API_BASE}/markets",
                params={
                    "active": True,
                    "closed": False,
                    "limit": limit,
                    "offset": offset,
                    "order": "volume24hr",
                    "ascending": False,
                },
            )
            resp.raise_for_status()
            batch = resp.json()

            if not batch:
                break

            all_markets.extend(batch)
            offset += limit

        log.info("gamma_fetch_complete", total=len(all_markets))
        return all_markets

    def _enrich_with_orderbook(self, raw: dict) -> Optional[Market]:
        """Build Market from Gamma API data. Uses Gamma prices directly (fast)."""
        import json as _json

        # Extract token IDs - Gamma API returns clobTokenIds as a JSON STRING
        tokens = raw.get("clobTokenIds")
        if tokens and isinstance(tokens, str):
            try:
                tokens = _json.loads(tokens)
            except (ValueError, TypeError):
                tokens = None

        if not tokens:
            tokens = raw.get("tokens", [])
            if tokens and isinstance(tokens[0], dict):
                tokens = [t.get("token_id") for t in tokens]

        if not tokens or len(tokens) == 0:
            return None

        token_id = tokens[0] if isinstance(tokens, list) else str(tokens)
        # NO token is the second token (index 1)
        no_token_id = tokens[1] if isinstance(tokens, list) and len(tokens) > 1 else None

        # Get prices from Gamma data directly (NO slow CLOB API call)
        best_bid = None
        best_ask = None

        # Try outcomePrices first (most reliable)
        outcome_prices = raw.get("outcomePrices")
        if outcome_prices:
            try:
                if isinstance(outcome_prices, str):
                    prices = _json.loads(outcome_prices)
                else:
                    prices = outcome_prices
                yes_price = float(prices[0])
                best_bid = max(yes_price - 0.02, 0.01)
                best_ask = min(yes_price + 0.02, 0.99)
            except (ValueError, IndexError, TypeError):
                pass

        # Fallback: bestBid/bestAsk fields
        if best_bid is None:
            best_bid = float(raw.get("bestBid", 0) or 0)
        if best_ask is None:
            best_ask = float(raw.get("bestAsk", 0) or 0)

        # Last resort: lastTradePrice
        if best_bid <= 0 or best_ask <= 0:
            last_price = float(raw.get("lastTradePrice", 0) or raw.get("price", 0) or 0)
            if last_price > 0:
                best_bid = best_bid if best_bid > 0 else max(last_price - 0.02, 0.01)
                best_ask = best_ask if best_ask > 0 else min(last_price + 0.02, 0.99)

        if not best_bid or not best_ask or best_bid <= 0 or best_ask <= 0:
            return None

        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        return Market(
            condition_id=raw.get("conditionId", raw.get("condition_id", "")),
            token_id=token_id,
            question=raw.get("question", ""),
            description=raw.get("description", "")[:500],  # Truncate for API costs
            outcome="Yes",
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid,
            spread=spread,
            volume_24h=float(raw.get("volume24hr", raw.get("volume_24h", 0)) or 0),
            liquidity=float(raw.get("liquidity", 0) or 0),
            end_date=raw.get("endDate", raw.get("end_date_iso")),
            category=raw.get("category"),
            no_token_id=no_token_id,
        )

    def _passes_filters(self, market: Market) -> bool:
        """Filter out markets we don't want to trade."""
        # Skip markets with no liquidity AND no volume
        # (use volume as proxy when liquidity field is missing/zero)
        if market.liquidity < self.config.min_liquidity_usd and market.volume_24h < 100:
            return False

        # Skip markets with huge spreads (> 20%)
        if market.spread > 0.20:
            return False

        # Skip markets at extreme prices (< 3% or > 97%) - hard to find edge
        if market.mid_price < 0.03 or market.mid_price > 0.97:
            return False

        # Must have a question
        if not market.question:
            return False

        return True

    def close(self):
        self.client.close()

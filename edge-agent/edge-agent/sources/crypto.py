"""
Crypto Data Source: On-chain metrics + sentiment for crypto markets.

Free APIs used:
- CoinCap (no key needed, 200 req/min)
- CoinPaprika (no key needed, 20k calls/month free)
- Blockchain.com (Bitcoin on-chain)
- Alternative.me Fear & Greed Index
- DeFiLlama (TVL data)

The edge: on-chain metrics (whale movements, exchange flows, funding rates)
often lead price action by hours. If a Polymarket crypto market hasn't
adjusted to whale sell signals, that's edge.
"""

import json
import re
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional

import httpx
import structlog

log = structlog.get_logger()

COINCAP_BASE = "https://api.coincap.io/v2"
COINPAPRIKA_BASE = "https://api.coinpaprika.com/v1"
DEFILLAMA_BASE = "https://api.llama.fi"
FEAR_GREED_URL = "https://api.alternative.me/fng/"
BLOCKCHAIN_INFO_BASE = "https://api.blockchain.info"

# Crypto keywords for market detection
CRYPTO_KEYWORDS = [
    "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
    "solana", "sol", "dogecoin", "doge", "xrp", "ripple",
    "cardano", "ada", "polygon", "matic", "avalanche", "avax",
    "chainlink", "link", "polkadot", "dot", "uniswap", "uni",
    "defi", "nft", "blockchain", "web3", "stablecoin",
    "binance", "coinbase", "kraken", "ftx",
    "halving", "etf", "spot etf", "bitcoin etf",
    "market cap", "all-time high", "ath",
    "$100k", "$150k", "$200k", "$50k", "$1000", "$10000",
]

# Map common names to CoinCap IDs (CoinCap uses lowercase names)
COIN_MAP = {
    "bitcoin": "bitcoin", "btc": "bitcoin",
    "ethereum": "ethereum", "eth": "ethereum",
    "solana": "solana", "sol": "solana",
    "dogecoin": "dogecoin", "doge": "dogecoin",
    "xrp": "xrp", "ripple": "xrp",
    "cardano": "cardano", "ada": "cardano",
    "polygon": "polygon", "matic": "polygon",
    "avalanche": "avalanche", "avax": "avalanche",
    "chainlink": "chainlink", "link": "chainlink",
    "polkadot": "polkadot", "dot": "polkadot",
    "toncoin": "toncoin", "ton": "toncoin",
    "shiba": "shiba-inu", "shib": "shiba-inu",
}

# CoinPaprika IDs (different format)
PAPRIKA_MAP = {
    "bitcoin": "btc-bitcoin",
    "ethereum": "eth-ethereum",
    "solana": "sol-solana",
    "dogecoin": "doge-dogecoin",
    "xrp": "xrp-xrp",
    "cardano": "ada-cardano",
    "polygon": "matic-polygon",
    "avalanche": "avax-avalanche",
    "chainlink": "link-chainlink",
    "polkadot": "dot-polkadot",
    "toncoin": "ton-toncoin",
    "shiba-inu": "shib-shiba-inu",
}


@dataclass
class CoinMetrics:
    """On-chain and market data for a cryptocurrency."""
    coin_id: str
    name: str
    current_price: float
    price_change_24h_pct: float
    price_change_7d_pct: float
    price_change_30d_pct: float
    market_cap: float
    volume_24h: float
    volume_to_mcap_ratio: float  # High ratio = unusual activity
    ath: float
    ath_pct_change: float  # Distance from ATH
    circulating_supply: float
    total_supply: Optional[float]


@dataclass
class OnChainData:
    """Bitcoin-specific on-chain metrics."""
    mempool_size: Optional[int] = None
    hash_rate: Optional[str] = None
    difficulty: Optional[str] = None
    avg_block_time: Optional[float] = None
    unconfirmed_txs: Optional[int] = None


@dataclass
class CryptoData:
    """Enrichment data for a crypto market."""
    market_id: str
    coins_detected: list[str] = field(default_factory=list)
    coin_metrics: list[CoinMetrics] = field(default_factory=list)
    fear_greed_index: Optional[dict] = None  # {value, classification}
    on_chain: Optional[OnChainData] = None
    defi_tvl: Optional[dict] = None  # Total DeFi TVL data
    price_targets_mentioned: list[float] = field(default_factory=list)
    market_summary: str = ""
    data_timestamp: str = ""
    source: str = "CoinGecko + On-chain"


class CryptoSource:
    """Fetches crypto on-chain and market data."""

    def __init__(self):
        self.client = httpx.Client(timeout=15.0)
        self._cache: dict[str, tuple[float, any]] = {}
        self._cache_ttl = 300  # 5 min cache

    def is_crypto_market(self, question: str) -> bool:
        """Check if a market is crypto-related."""
        q_lower = question.lower()
        return any(kw in q_lower for kw in CRYPTO_KEYWORDS)

    def enrich(self, market_id: str, question: str, description: str = "") -> Optional[CryptoData]:
        """Fetch crypto data relevant to a market question."""
        try:
            q_lower = question.lower()

            data = CryptoData(
                market_id=market_id,
                data_timestamp=datetime.now(timezone.utc).isoformat(),
            )

            # Detect which coins are mentioned
            coins = self._detect_coins(q_lower)
            data.coins_detected = coins

            # Extract any price targets from the question
            data.price_targets_mentioned = self._extract_price_targets(question)

            # Fetch metrics for detected coins
            if coins:
                coin_ids = list(set(COIN_MAP.get(c, c) for c in coins))
                metrics = self._get_coin_metrics(coin_ids)
                data.coin_metrics = metrics

            # Always fetch Fear & Greed (useful context for any crypto market)
            data.fear_greed_index = self._get_fear_greed()

            # Bitcoin on-chain data if BTC is involved
            if "bitcoin" in [COIN_MAP.get(c, c) for c in coins]:
                data.on_chain = self._get_btc_on_chain()

            # DeFi TVL if DeFi-related
            if any(kw in q_lower for kw in ["defi", "tvl", "total value locked"]):
                data.defi_tvl = self._get_defi_tvl()

            # Build summary
            data.market_summary = self._build_summary(data)

            if data.coin_metrics or data.fear_greed_index:
                log.debug("crypto_enriched", market_id=market_id, coins=coins)
                return data

            return None

        except Exception as e:
            log.warning("crypto_enrich_failed", market_id=market_id, error=str(e))
            return None

    def _detect_coins(self, question: str) -> list[str]:
        """Detect cryptocurrency mentions in question."""
        found = []
        for keyword in COIN_MAP:
            # Use word boundary matching to avoid false positives
            if re.search(rf'\b{re.escape(keyword)}\b', question):
                found.append(keyword)
        return list(set(found))[:5]  # Max 5 coins

    def _extract_price_targets(self, question: str) -> list[float]:
        """Extract price targets mentioned in the question."""
        targets = []
        # Match patterns like $100k, $50,000, $100000
        patterns = [
            r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*k\b',  # $100k
            r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\b',  # $50,000 or $100000
        ]
        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for m in matches:
                val = float(m.replace(",", ""))
                if pattern.endswith(r'k\b'):
                    val *= 1000
                targets.append(val)
        return targets

    def _get_coin_metrics(self, coin_ids: list[str]) -> list[CoinMetrics]:
        """Fetch market data. Try multiple sources for resilience."""
        ids_str = ",".join(coin_ids)
        cache_key = f"metrics_{ids_str}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Try Binance first (most reliable, no auth needed)
        metrics = self._fetch_binance(coin_ids)

        # Fallback to CoinPaprika
        if not metrics:
            metrics = self._fetch_coinpaprika(coin_ids)

        # Last resort: CoinCap
        if not metrics:
            metrics = self._fetch_coincap(coin_ids)

        if metrics:
            self._set_cached(cache_key, metrics)
        return metrics

    def _fetch_binance(self, coin_ids: list[str]) -> list[CoinMetrics]:
        """Fetch from Binance public API — no auth, very generous limits."""
        # Map our IDs to Binance symbols (they trade against USDT)
        binance_map = {
            "bitcoin": "BTCUSDT", "ethereum": "ETHUSDT", "solana": "SOLUSDT",
            "dogecoin": "DOGEUSDT", "xrp": "XRPUSDT", "cardano": "ADAUSDT",
            "polygon": "MATICUSDT", "avalanche": "AVAXUSDT",
            "chainlink": "LINKUSDT", "polkadot": "DOTUSDT",
            "toncoin": "TONUSDT", "shiba-inu": "SHIBUSDT",
        }

        symbols = [binance_map[cid] for cid in coin_ids if cid in binance_map]
        if not symbols:
            return []

        try:
            # Batch call — get 24h stats for all symbols at once
            resp = self.client.get(
                "https://api.binance.com/api/v3/ticker/24hr",
                params={"symbols": json.dumps(symbols)},
            )
            resp.raise_for_status()
            data = resp.json()

            # Reverse map: symbol -> coin_id
            sym_to_id = {v: k for k, v in binance_map.items()}

            metrics = []
            for ticker in data:
                symbol = ticker.get("symbol", "")
                coin_id = sym_to_id.get(symbol, "")
                if not coin_id:
                    continue

                price = float(ticker.get("lastPrice", 0))
                vol = float(ticker.get("quoteVolume", 0))  # Volume in USDT
                change_24h = float(ticker.get("priceChangePercent", 0))

                metrics.append(CoinMetrics(
                    coin_id=coin_id,
                    name=coin_id.capitalize(),
                    current_price=price,
                    price_change_24h_pct=change_24h,
                    price_change_7d_pct=0,
                    price_change_30d_pct=0,
                    market_cap=0,
                    volume_24h=vol,
                    volume_to_mcap_ratio=0,
                    ath=0,
                    ath_pct_change=0,
                    circulating_supply=0,
                    total_supply=None,
                ))

            return metrics

        except Exception as e:
            log.debug("binance_failed", error=str(e))
            return []

    def _fetch_coincap(self, coin_ids: list[str]) -> list[CoinMetrics]:
        """Fetch from CoinCap API — 200 req/min, no auth."""
        try:
            ids_str = ",".join(coin_ids)
            resp = self.client.get(
                f"{COINCAP_BASE}/assets",
                params={"ids": ids_str},
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])

            metrics = []
            for coin in data:
                price = float(coin.get("priceUsd", 0) or 0)
                mcap = float(coin.get("marketCapUsd", 0) or 1)
                vol = float(coin.get("volumeUsd24Hr", 0) or 0)
                change_24h = float(coin.get("changePercent24Hr", 0) or 0)
                supply = float(coin.get("supply", 0) or 0)
                max_supply = coin.get("maxSupply")

                metrics.append(CoinMetrics(
                    coin_id=coin.get("id", ""),
                    name=coin.get("name", ""),
                    current_price=price,
                    price_change_24h_pct=change_24h,
                    price_change_7d_pct=0,  # CoinCap doesn't have 7d in this endpoint
                    price_change_30d_pct=0,
                    market_cap=mcap,
                    volume_24h=vol,
                    volume_to_mcap_ratio=round(vol / mcap, 4) if mcap > 0 else 0,
                    ath=0,  # CoinCap doesn't track ATH
                    ath_pct_change=0,
                    circulating_supply=supply,
                    total_supply=float(max_supply) if max_supply else None,
                ))

            return metrics

        except Exception as e:
            log.debug("coincap_failed", error=str(e))
            return []

    def _fetch_coinpaprika(self, coin_ids: list[str]) -> list[CoinMetrics]:
        """Fallback: CoinPaprika API — no auth, 20k calls/month."""
        metrics = []
        for coin_id in coin_ids[:3]:  # Limit to avoid burning quota
            paprika_id = PAPRIKA_MAP.get(coin_id)
            if not paprika_id:
                continue
            try:
                resp = self.client.get(f"{COINPAPRIKA_BASE}/tickers/{paprika_id}")
                resp.raise_for_status()
                coin = resp.json()

                quotes = coin.get("quotes", {}).get("USD", {})
                price = quotes.get("price", 0)
                mcap = quotes.get("market_cap", 1)
                vol = quotes.get("volume_24h", 0)

                metrics.append(CoinMetrics(
                    coin_id=coin_id,
                    name=coin.get("name", ""),
                    current_price=price,
                    price_change_24h_pct=quotes.get("percent_change_24h", 0) or 0,
                    price_change_7d_pct=quotes.get("percent_change_7d", 0) or 0,
                    price_change_30d_pct=quotes.get("percent_change_30d", 0) or 0,
                    market_cap=mcap,
                    volume_24h=vol,
                    volume_to_mcap_ratio=round(vol / mcap, 4) if mcap > 0 else 0,
                    ath=quotes.get("ath_price", 0) or 0,
                    ath_pct_change=quotes.get("percent_from_price_ath", 0) or 0,
                    circulating_supply=coin.get("circulating_supply", 0) or 0,
                    total_supply=coin.get("total_supply"),
                ))
            except Exception as e:
                log.debug("coinpaprika_failed", coin=coin_id, error=str(e))
                continue

        return metrics

    def _get_fear_greed(self) -> Optional[dict]:
        """Get the Crypto Fear & Greed Index."""
        cache_key = "fear_greed"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            resp = self.client.get(FEAR_GREED_URL, params={"limit": 7})
            resp.raise_for_status()
            data = resp.json()

            entries = data.get("data", [])
            if not entries:
                return None

            current = entries[0]
            result = {
                "value": int(current["value"]),
                "classification": current["value_classification"],
                "trend_7d": [int(e["value"]) for e in entries[:7]],
            }

            self._set_cached(cache_key, result, ttl=1800)  # 30 min
            return result

        except Exception as e:
            log.debug("fear_greed_failed", error=str(e))
            return None

    def _get_btc_on_chain(self) -> Optional[OnChainData]:
        """Get Bitcoin on-chain metrics."""
        cache_key = "btc_onchain"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            on_chain = OnChainData()

            # Unconfirmed transactions (mempool pressure)
            resp = self.client.get(f"{BLOCKCHAIN_INFO_BASE}/q/unconfirmedcount")
            if resp.status_code == 200:
                on_chain.unconfirmed_txs = int(resp.text.strip())

            # Hash rate
            resp = self.client.get(f"{BLOCKCHAIN_INFO_BASE}/q/hashrate")
            if resp.status_code == 200:
                on_chain.hash_rate = resp.text.strip()

            # Difficulty
            resp = self.client.get(f"{BLOCKCHAIN_INFO_BASE}/q/getdifficulty")
            if resp.status_code == 200:
                on_chain.difficulty = resp.text.strip()

            # Block interval
            resp = self.client.get(f"{BLOCKCHAIN_INFO_BASE}/q/interval")
            if resp.status_code == 200:
                on_chain.avg_block_time = float(resp.text.strip())

            self._set_cached(cache_key, on_chain, ttl=600)  # 10 min
            return on_chain

        except Exception as e:
            log.debug("btc_onchain_failed", error=str(e))
            return None

    def _get_defi_tvl(self) -> Optional[dict]:
        """Get total DeFi TVL from DeFiLlama."""
        cache_key = "defi_tvl"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            resp = self.client.get(f"{DEFILLAMA_BASE}/v2/historicalChainTvl")
            resp.raise_for_status()
            data = resp.json()

            if not data:
                return None

            # Get latest and 7d ago TVL
            latest = data[-1] if data else {}
            week_ago = data[-8] if len(data) > 7 else data[0]

            result = {
                "total_tvl_usd": latest.get("tvl", 0),
                "tvl_7d_ago": week_ago.get("tvl", 0),
                "tvl_change_7d_pct": round(
                    ((latest.get("tvl", 0) - week_ago.get("tvl", 0)) / max(week_ago.get("tvl", 1), 1)) * 100,
                    2,
                ),
            }

            self._set_cached(cache_key, result, ttl=1800)
            return result

        except Exception as e:
            log.debug("defillama_failed", error=str(e))
            return None

    def _build_summary(self, data: CryptoData) -> str:
        """Build a concise text summary for Claude."""
        parts = []

        for m in data.coin_metrics:
            parts.append(
                f"{m.name}: ${m.current_price:,.2f} "
                f"(24h: {m.price_change_24h_pct:+.1f}%, 7d: {m.price_change_7d_pct:+.1f}%, "
                f"30d: {m.price_change_30d_pct:+.1f}%) "
                f"Vol/MCap: {m.volume_to_mcap_ratio:.3f} "
                f"ATH: ${m.ath:,.2f} ({m.ath_pct_change:+.1f}% from ATH)"
            )

        if data.fear_greed_index:
            fg = data.fear_greed_index
            parts.append(f"Fear/Greed: {fg['value']}/100 ({fg['classification']}), 7d trend: {fg['trend_7d']}")

        if data.on_chain:
            oc = data.on_chain
            if oc.unconfirmed_txs:
                parts.append(f"BTC mempool: {oc.unconfirmed_txs:,} unconfirmed txs")
            if oc.hash_rate:
                parts.append(f"BTC hash rate: {oc.hash_rate}")

        if data.defi_tvl:
            tvl = data.defi_tvl
            parts.append(
                f"DeFi TVL: ${tvl['total_tvl_usd']/1e9:.1f}B "
                f"(7d change: {tvl['tvl_change_7d_pct']:+.1f}%)"
            )

        if data.price_targets_mentioned:
            parts.append(f"Price targets in question: {data.price_targets_mentioned}")

        return " | ".join(parts)

    def _get_cached(self, key: str):
        if key in self._cache:
            ts, data, ttl = self._cache[key]
            if time.time() - ts < ttl:
                return data
        return None

    def _set_cached(self, key: str, data, ttl: int = None):
        self._cache[key] = (time.time(), data, ttl or self._cache_ttl)

    def close(self):
        self.client.close()

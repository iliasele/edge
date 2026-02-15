"""Configuration and data models for the trading agent."""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Config:
    # Polymarket CLOB
    poly_api_key: str = os.getenv("POLYMARKET_API_KEY", "")
    poly_secret: str = os.getenv("POLYMARKET_SECRET", "")
    poly_passphrase: str = os.getenv("POLYMARKET_PASSPHRASE", "")
    private_key: str = os.getenv("PRIVATE_KEY", "")
    wallet_address: str = os.getenv("WALLET_ADDRESS", "")

    # Anthropic
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Agent params
    dry_run: bool = os.getenv("DRY_RUN", "true").lower() == "true"
    max_bankroll_pct: float = float(os.getenv("MAX_BANKROLL_PCT", "0.10"))  # 10% max per trade
    min_edge_pct: float = float(os.getenv("MIN_EDGE_PCT", "0.01"))  # 1% during learning phase
    scan_interval: int = int(os.getenv("SCAN_INTERVAL_SECONDS", "600"))
    max_markets: int = int(os.getenv("MAX_MARKETS_PER_SCAN", "750"))

    # Safety
    max_daily_loss_pct: float = 0.15  # Stop trading if down 15% in a day
    max_open_positions: int = 20
    min_liquidity_usd: float = 500.0  # Skip illiquid markets
    trades_per_cycle: int = int(os.getenv("TRADES_PER_CYCLE", "5"))  # Max new trades per cycle
    market_cooldown_cycles: int = int(os.getenv("MARKET_COOLDOWN_CYCLES", "6"))  # Skip market for N cycles after trading


@dataclass
class Market:
    """Represents a prediction market (Polymarket or Kalshi)."""
    condition_id: str
    token_id: str
    question: str
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float
    volume_24h: float
    description: str = ""
    outcome: str = "Yes"
    liquidity: float = 0.0
    end_date: Optional[str] = None
    category: Optional[str] = None
    source: str = "polymarket"  # "polymarket" or "kalshi"
    no_token_id: Optional[str] = None  # Polymarket NO token (for buying NO)


@dataclass
class FairValueEstimate:
    """Claude's estimate of fair probability."""
    market_id: str
    question: str
    fair_probability: float  # 0-1
    confidence: float  # 0-1, how confident Claude is
    reasoning: str
    key_factors: list[str] = field(default_factory=list)


@dataclass
class TradeSignal:
    """A trade the agent wants to execute."""
    market: Market
    estimate: FairValueEstimate
    side: Side
    edge: float  # fair_value - market_price
    kelly_fraction: float
    position_size_usd: float
    expected_value: float


@dataclass
class Position:
    """An open position."""
    market_id: str
    token_id: str
    question: str
    side: Side
    entry_price: float
    size_usd: float
    current_price: float
    pnl: float
    timestamp: str


@dataclass
class AgentState:
    """Full agent state - persisted to disk."""
    balance_usd: float = 0.0
    total_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    api_costs_usd: float = 0.0
    open_positions: list[Position] = field(default_factory=list)
    daily_pnl: float = 0.0
    is_alive: bool = True

"""
Accountant: Tracks bankroll, records trades, persists state to JSON.

This is the financial ledger of the agent. It:
  - Tracks balance, P&L, API costs
  - Records every trade placed (with order IDs for settlement tracking)
  - Persists full state to disk so the agent survives restarts
  - Provides reporting

The Accountant does NOT check settlements — that's the SettlementMonitor's job.
The Accountant just records what it's told.
"""

import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional

import structlog

log = structlog.get_logger()

STATE_FILE = "agent_state.json"
TRADE_LOG_FILE = "trade_log.json"


@dataclass
class TradeRecord:
    """A single trade placed by the agent."""
    trade_id: str              # Unique ID (order_id from exchange)
    timestamp: str             # ISO timestamp when placed
    source: str                # "polymarket" or "kalshi"
    market_id: str             # condition_id or ticker
    question: str              # Human-readable question
    side: str                  # "BUY" or "SELL"
    price: float               # Entry price
    size_usd: float            # Dollar amount
    edge: float                # Expected edge at time of trade
    kelly: float               # Kelly fraction used
    status: str                # "placed", "filled", "settled", "cancelled", "expired"
    # Settlement fields (filled in later by SettlementMonitor)
    settled: bool = False
    settlement_time: Optional[str] = None
    market_result: Optional[str] = None   # "yes", "no", or None
    pnl: float = 0.0
    won: Optional[bool] = None
    # Kalshi-specific
    ticker: Optional[str] = None
    count: int = 0             # Number of contracts (Kalshi)


class Accountant:
    """
    Financial ledger for the trading agent.
    
    Persists to disk so state survives restarts.
    """

    def __init__(self, initial_balance: float = 100.0):
        self.trades: list[TradeRecord] = []
        self.balance_usd: float = initial_balance
        self.total_api_costs: float = 0.0
        self.start_time: str = datetime.now(timezone.utc).isoformat()
        
        # Try to load existing state
        loaded = self._load_state()
        if not loaded:
            self.balance_usd = initial_balance
            log.info("accountant_initialized", balance=f"${initial_balance:.2f}")

    def record_trade(self, result: dict) -> Optional[TradeRecord]:
        """
        Record a trade execution result.
        
        Args:
            result: Dict from executor with keys like:
                status, order_id, market, side, price, size_usd, edge, kelly, source, count, etc.
        
        Returns:
            TradeRecord if recorded, None if skipped (error/dry_run)
        """
        status = result.get("status", "unknown")
        
        # Skip errors
        if status == "error":
            log.warning("trade_not_recorded_error", error=result.get("error", ""))
            return None
        
        trade = TradeRecord(
            trade_id=result.get("order_id", f"dry_{len(self.trades)}"),
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=result.get("source", "polymarket"),
            market_id=result.get("market_id", ""),
            question=result.get("market", "")[:200],
            side=result.get("side", "BUY"),
            price=result.get("price", 0.0),
            size_usd=result.get("size_usd", 0.0),
            edge=result.get("edge", 0.0),
            kelly=result.get("kelly", 0.0),
            status=status,
            ticker=result.get("ticker", None),
            count=result.get("count", 0),
        )
        
        self.trades.append(trade)
        self._save_state()
        
        log.info(
            "trade_recorded",
            trade_id=trade.trade_id[:20],
            source=trade.source,
            side=trade.side,
            size=f"${trade.size_usd:.2f}",
            total_trades=len(self.trades),
        )
        
        return trade

    def record_settlement(
        self,
        trade_id: str,
        won: bool,
        pnl: float,
        market_result: str = "",
    ):
        """
        Record that a trade has settled.
        Called by SettlementMonitor when it detects a resolved market.
        """
        for trade in self.trades:
            if trade.trade_id == trade_id and not trade.settled:
                trade.settled = True
                trade.won = won
                trade.pnl = pnl
                trade.market_result = market_result
                trade.settlement_time = datetime.now(timezone.utc).isoformat()
                trade.status = "settled"
                
                log.info(
                    "trade_settled",
                    trade_id=trade_id[:20],
                    won=won,
                    pnl=f"${pnl:.2f}",
                    result=market_result,
                    question=trade.question[:60],
                )
                
                self._save_state()
                return True
        
        log.warning("settlement_trade_not_found", trade_id=trade_id)
        return False

    def mark_cancelled(self, trade_id: str):
        """Mark a resting order as cancelled/expired."""
        for trade in self.trades:
            if trade.trade_id == trade_id:
                trade.status = "cancelled"
                trade.settled = True
                trade.pnl = 0.0
                self._save_state()
                return True
        return False

    def record_exit(self, trade_id: str, exit_pnl: float, exit_price: float):
        """
        Record that we exited a position early (before settlement).
        
        This is different from settlement — we actively sold.
        """
        for trade in self.trades:
            if trade.trade_id == trade_id and not trade.settled:
                trade.settled = True
                trade.status = "exited"
                trade.pnl = exit_pnl
                trade.won = exit_pnl > 0
                trade.market_result = f"exited@{exit_price:.2f}"
                trade.settlement_time = datetime.now(timezone.utc).isoformat()

                # Add P&L back to balance
                self.balance_usd += exit_pnl

                log.info(
                    "trade_exited",
                    trade_id=trade_id[:20],
                    pnl=f"${exit_pnl:.2f}",
                    exit_price=exit_price,
                    question=trade.question[:60],
                )

                self._save_state()
                return True

        log.warning("exit_trade_not_found", trade_id=trade_id)
        return False

    def deduct_api_cost(self, cost: float):
        """Deduct API cost from balance."""
        self.total_api_costs += cost
        self.balance_usd -= cost
        if cost > 0.01:
            log.info("api_cost", cost=f"${cost:.4f}", total=f"${self.total_api_costs:.2f}")

    def get_bankroll(self) -> float:
        return max(self.balance_usd, 0.0)

    def is_alive(self) -> bool:
        return self.balance_usd > 0.5

    def can_afford_cycle(self, estimated_cost: float) -> bool:
        return self.balance_usd > estimated_cost * 2

    def get_open_trades(self) -> list[TradeRecord]:
        """Get all trades that haven't been settled yet."""
        return [t for t in self.trades if not t.settled and t.status in ("placed", "filled")]

    def get_settled_trades(self) -> list[TradeRecord]:
        """Get all settled trades."""
        return [t for t in self.trades if t.settled and t.status == "settled"]

    def get_stats(self) -> dict:
        """Get summary statistics."""
        settled = self.get_settled_trades()
        wins = [t for t in settled if t.won]
        losses = [t for t in settled if t.won is False]
        
        total_pnl = sum(t.pnl for t in settled)
        
        return {
            "total_trades": len(self.trades),
            "open_trades": len(self.get_open_trades()),
            "settled_trades": len(settled),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / max(len(settled), 1),
            "total_pnl": total_pnl,
            "api_costs": self.total_api_costs,
            "net_pnl": total_pnl - self.total_api_costs,
            "balance": self.balance_usd,
        }

    def get_report(self) -> str:
        """Generate a human-readable report."""
        stats = self.get_stats()
        
        lines = [
            "╔══════════════════════════════════════╗",
            "║         AGENT STATUS REPORT          ║",
            "╠══════════════════════════════════════╣",
            f"║  Balance:     ${stats['balance']:>10.2f}          ║",
            f"║  Total Trades: {stats['total_trades']:>5}               ║",
            f"║  Open:         {stats['open_trades']:>5}               ║",
            f"║  Settled:      {stats['settled_trades']:>5}               ║",
            f"║  Wins:         {stats['wins']:>5}               ║",
            f"║  Losses:       {stats['losses']:>5}               ║",
            f"║  Win Rate:    {stats['win_rate']:>8.1%}             ║",
            f"║  Trade P&L:   ${stats['total_pnl']:>10.2f}          ║",
            f"║  API Costs:   ${stats['api_costs']:>10.2f}          ║",
            f"║  Net P&L:     ${stats['net_pnl']:>10.2f}          ║",
            "╚══════════════════════════════════════╝",
        ]
        return "\n".join(lines)

    def _save_state(self):
        """Persist state to disk."""
        try:
            state = {
                "last_saved": datetime.now(timezone.utc).isoformat(),
                "start_time": self.start_time,
                "balance_usd": self.balance_usd,
                "total_api_costs": self.total_api_costs,
                "trades": [asdict(t) for t in self.trades],
            }
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log.error("state_save_failed", error=str(e))

    def _load_state(self) -> bool:
        """Load state from disk. Returns True if loaded."""
        if not os.path.exists(STATE_FILE):
            return False
        
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            
            self.start_time = state.get("start_time", self.start_time)
            self.balance_usd = state.get("balance_usd", self.balance_usd)
            self.total_api_costs = state.get("total_api_costs", 0.0)
            
            self.trades = []
            for td in state.get("trades", []):
                self.trades.append(TradeRecord(**td))
            
            log.info(
                "state_loaded",
                balance=f"${self.balance_usd:.2f}",
                trades=len(self.trades),
                api_costs=f"${self.total_api_costs:.2f}",
            )
            return True
        except Exception as e:
            log.error("state_load_failed", error=str(e))
            return False

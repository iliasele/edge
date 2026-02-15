"""
Trade Logger: CSV logging for analysis.

Logs every trade and market analysis to CSV files for later review.
This is separate from the Accountant (which does persistence for the agent).
"""

import csv
import os
from datetime import datetime, timezone
from typing import Optional

import structlog

log = structlog.get_logger()

TRADE_CSV = "trades.csv"
ANALYSIS_CSV = "analyses.csv"


class TradeLogger:
    """Logs trades and analyses to CSV for review."""

    def __init__(self):
        self._ensure_trade_csv()
        self._ensure_analysis_csv()

    def _ensure_trade_csv(self):
        if not os.path.exists(TRADE_CSV):
            with open(TRADE_CSV, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp", "cycle", "source", "market", "side",
                    "price", "size_usd", "edge", "kelly", "domain",
                    "bankroll", "status",
                    # Settlement columns (filled in later)
                    "settled_at", "market_result", "won", "pnl",
                ])

    def _ensure_analysis_csv(self):
        if not os.path.exists(ANALYSIS_CSV):
            with open(ANALYSIS_CSV, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp", "cycle", "source", "market_id", "question",
                    "mid_price", "fair_value", "edge", "confidence",
                    "domain", "traded",
                ])

    def log_trade(self, cycle, signal, execution_result, bankroll, domain="general"):
        try:
            with open(TRADE_CSV, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    datetime.now(timezone.utc).isoformat(),
                    cycle,
                    signal.market.source,
                    signal.market.question[:100],
                    signal.side.value,
                    execution_result.get("price", 0),
                    execution_result.get("size_usd", signal.position_size_usd),
                    f"{signal.edge:.4f}",
                    f"{signal.kelly_fraction:.4f}",
                    domain,
                    f"{bankroll:.2f}",
                    execution_result.get("status", "unknown"),
                    "", "", "", "",  # settlement columns filled later
                ])
        except Exception as e:
            log.debug("trade_log_failed", error=str(e))

    def log_settlement(self, settlement: dict):
        """
        Append a settlement row to the CSV.
        
        Args:
            settlement: dict with keys: trade_id, won, pnl, market_result, 
                       source, question, edge
        """
        try:
            with open(TRADE_CSV, "a", newline="") as f:
                w = csv.writer(f)
                won = settlement.get("won")
                w.writerow([
                    datetime.now(timezone.utc).isoformat(),
                    "",  # cycle not applicable for settlement
                    settlement.get("source", ""),
                    settlement.get("question", "")[:100],
                    "",  # side already logged at placement
                    "", "",  # price, size already logged
                    settlement.get("edge", ""),
                    "", "",  # kelly, domain
                    "", "settled",
                    # Settlement columns
                    datetime.now(timezone.utc).isoformat(),
                    settlement.get("market_result", ""),
                    "WIN" if won else ("LOSS" if won is False else "CANCEL"),
                    f"{settlement.get('pnl', 0):.2f}",
                ])
        except Exception as e:
            log.debug("settlement_log_failed", error=str(e))

    def update_trade_row(self, question_prefix: str, settlement: dict):
        """
        Try to update the original trade row in-place with settlement data.
        Falls back gracefully if the CSV is too large or row not found.
        """
        try:
            rows = []
            updated = False
            with open(TRADE_CSV, "r", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    # Match by question prefix (col 3) and source (col 2)
                    if (not updated 
                        and len(row) >= 16
                        and row[3][:40] == settlement.get("question", "")[:40]
                        and row[2] == settlement.get("source", "")
                        and row[12] == ""):  # settled_at column empty = not yet settled
                        # Fill in settlement columns
                        row[11] = "settled"
                        row[12] = datetime.now(timezone.utc).isoformat()
                        row[13] = settlement.get("market_result", "")
                        won = settlement.get("won")
                        row[14] = "WIN" if won else ("LOSS" if won is False else "CANCEL")
                        row[15] = f"{settlement.get('pnl', 0):.2f}"
                        updated = True
                    rows.append(row)
            
            if updated:
                with open(TRADE_CSV, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
                return True
        except Exception as e:
            log.debug("trade_row_update_failed", error=str(e))
        return False

    def log_batch(self, cycle, markets, estimates, enriched_list, signals):
        """Log all analyzed markets in a batch."""
        try:
            signal_ids = {s.market.condition_id for s in signals}
            est_map = {e.market_id: e for e in estimates}
            enr_map = {}
            for e in enriched_list:
                if hasattr(e, "market"):
                    enr_map[e.market.condition_id] = e

            with open(ANALYSIS_CSV, "a", newline="") as f:
                w = csv.writer(f)
                for m in markets:
                    est = est_map.get(m.condition_id)
                    enr = enr_map.get(m.condition_id)
                    w.writerow([
                        datetime.now(timezone.utc).isoformat(),
                        cycle,
                        m.source,
                        m.condition_id[:30],
                        m.question[:100],
                        f"{m.mid_price:.3f}",
                        f"{est.fair_probability:.3f}" if est else "",
                        f"{est.fair_probability - m.mid_price:.4f}" if est else "",
                        f"{est.confidence:.2f}" if est else "",
                        enr.domain if enr else "general",
                        m.condition_id in signal_ids,
                    ])
        except Exception as e:
            log.debug("batch_log_failed", error=str(e))

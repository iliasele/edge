"""
Position Monitor: Watches open positions and decides when to exit.

Two-phase approach:
  Phase 1 (now): Rule-based exits with learning data collection
    - Exit when edge has decayed below threshold AND position is profitable
    - Stop loss when unrealized loss exceeds threshold
    - Time-based: exit when near expiry and edge is gone
    - ALL decisions are logged for Phase 2 learning

  Phase 2 (later): Claude-powered exit decisions
    - Feed historical exit decisions + outcomes to Claude
    - Let it learn patterns: "positions in crypto markets with >30% profit
      and <3% remaining edge should exit 80% of the time"

Exit execution:
  - Polymarket: Sell YES tokens (or sell NO tokens) via CLOB
  - Kalshi: Sell contracts via authenticated API

Data collected per decision:
  - entry_price, current_price, unrealized_pnl_pct
  - original_edge, current_edge (re-estimated)
  - time_held_hours, time_to_expiry_hours
  - decision: HOLD or EXIT
  - outcome: what actually happened (filled in at settlement)
"""

import json
import os
import time
import structlog
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from typing import Optional

log = structlog.get_logger()

EXIT_LOG_FILE = "exit_decisions.json"


@dataclass
class ExitDecision:
    """Records an exit evaluation for learning."""
    trade_id: str
    timestamp: str
    source: str                 # "polymarket" or "kalshi"
    question: str
    side: str                   # "BUY" or "SELL"
    entry_price: float
    current_price: float
    unrealized_pnl_pct: float   # e.g. +0.30 = +30%
    unrealized_pnl_usd: float
    original_edge: float
    current_edge: float         # Re-estimated edge at current price
    time_held_hours: float
    time_to_expiry_hours: float
    decision: str               # "HOLD" or "EXIT"
    reason: str                 # Why this decision was made
    # Filled in at settlement:
    final_pnl: Optional[float] = None
    was_correct: Optional[bool] = None  # Was the decision good in hindsight?


class PositionMonitor:
    """
    Monitors open positions and generates exit signals.
    
    Phase 1: Rule-based with data collection
    Phase 2: Claude-assisted (uses collected data)
    """

    # === TUNABLE PARAMETERS ===
    # These start conservative and can be adjusted as we collect data

    # Take profit: exit if profit > X% AND remaining edge < Y%
    TAKE_PROFIT_THRESHOLD = 0.30      # 30% unrealized profit
    EDGE_DECAY_EXIT = 0.03            # Exit if edge dropped below 3%

    # Stop loss: exit if unrealized loss > X%
    STOP_LOSS_THRESHOLD = -0.50       # -50% unrealized loss

    # Time-based: exit if < X hours to expiry AND edge < Y%
    EXPIRY_HOURS_THRESHOLD = 2.0      # Within 2 hours of expiry
    EXPIRY_EDGE_THRESHOLD = 0.05      # And edge below 5%

    # Minimum position age before considering exit (avoid churning)
    MIN_HOLD_HOURS = 1.0

    # How often to check positions (seconds)
    CHECK_INTERVAL = 300  # 5 minutes

    def __init__(self, config):
        self.config = config
        self.decisions: list[ExitDecision] = []
        self._load_decisions()
        self._last_check = 0

    def should_check(self) -> bool:
        """Rate limit position checks."""
        now = time.time()
        if now - self._last_check >= self.CHECK_INTERVAL:
            self._last_check = now
            return True
        return False

    def evaluate_positions(
        self,
        open_trades: list,
        price_fetcher: callable,
        fair_value_fetcher: callable = None,
    ) -> list[dict]:
        """
        Evaluate all open positions and return exit signals.

        Args:
            open_trades: list of TradeRecord from accountant
            price_fetcher: func(trade) -> current_price or None
            fair_value_fetcher: func(trade) -> fair_value or None (Phase 2)

        Returns:
            list of exit signals: [{"trade": TradeRecord, "decision": ExitDecision}, ...]
        """
        exit_signals = []

        for trade in open_trades:
            try:
                decision = self._evaluate_single(trade, price_fetcher, fair_value_fetcher)
                if decision is None:
                    continue

                self.decisions.append(decision)

                if decision.decision == "EXIT":
                    exit_signals.append({
                        "trade": trade,
                        "decision": decision,
                    })
                    log.info(
                        "exit_signal",
                        question=trade.question[:60],
                        pnl_pct=f"{decision.unrealized_pnl_pct:.1%}",
                        edge=f"{decision.current_edge:.1%}",
                        reason=decision.reason,
                    )
                else:
                    log.debug(
                        "hold_signal",
                        question=trade.question[:50],
                        pnl_pct=f"{decision.unrealized_pnl_pct:.1%}",
                        edge=f"{decision.current_edge:.1%}",
                    )

            except Exception as e:
                log.debug("position_eval_failed", trade_id=trade.trade_id[:20], error=str(e))
                continue

        if exit_signals:
            log.info("exit_signals_generated", count=len(exit_signals), total_positions=len(open_trades))

        self._save_decisions()
        return exit_signals

    def _evaluate_single(
        self,
        trade,
        price_fetcher,
        fair_value_fetcher=None,
    ) -> Optional[ExitDecision]:
        """Evaluate a single position. Returns ExitDecision or None if can't evaluate."""

        # Get current price
        current_price = price_fetcher(trade)
        if current_price is None or current_price <= 0:
            return None

        # Calculate unrealized P&L
        entry_price = trade.price
        if entry_price <= 0:
            return None

        # P&L depends on which side we're on
        if trade.side == "BUY":
            # Bought YES at entry_price, currently worth current_price
            unrealized_pnl_pct = (current_price - entry_price) / entry_price
            unrealized_pnl_usd = (current_price - entry_price) * trade.size_usd / entry_price
        else:
            # Bought NO (or sold YES): profit when price drops
            unrealized_pnl_pct = (entry_price - current_price) / entry_price
            unrealized_pnl_usd = (entry_price - current_price) * trade.size_usd / entry_price

        # Time calculations
        try:
            entry_time = datetime.fromisoformat(trade.timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            entry_time = datetime.now(timezone.utc) - timedelta(hours=1)

        now = datetime.now(timezone.utc)
        time_held_hours = (now - entry_time).total_seconds() / 3600

        # Time to expiry (parse from trade's end_date if available)
        time_to_expiry_hours = self._estimate_expiry_hours(trade)

        # Re-estimate current edge
        if fair_value_fetcher:
            fair_value = fair_value_fetcher(trade)
            if fair_value is not None:
                if trade.side == "BUY":
                    current_edge = fair_value - current_price
                else:
                    current_edge = (1.0 - fair_value) - (1.0 - current_price)
            else:
                current_edge = trade.edge * 0.5  # Assume some edge decay
        else:
            # Simple estimate: original edge decays over time
            # More time passed → less edge remaining (market converges)
            if time_to_expiry_hours > 0 and time_held_hours > 0:
                decay_factor = max(0, 1.0 - (time_held_hours / (time_held_hours + time_to_expiry_hours)))
                current_edge = trade.edge * decay_factor
            else:
                current_edge = trade.edge * 0.5

        # === RULE-BASED DECISIONS ===

        # Don't exit too early (avoid churning from spread costs)
        if time_held_hours < self.MIN_HOLD_HOURS:
            return ExitDecision(
                trade_id=trade.trade_id,
                timestamp=now.isoformat(),
                source=trade.source,
                question=trade.question,
                side=trade.side,
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl_pct=unrealized_pnl_pct,
                unrealized_pnl_usd=unrealized_pnl_usd,
                original_edge=trade.edge,
                current_edge=current_edge,
                time_held_hours=time_held_hours,
                time_to_expiry_hours=time_to_expiry_hours,
                decision="HOLD",
                reason=f"too_early (held {time_held_hours:.1f}h, min {self.MIN_HOLD_HOURS}h)",
            )

        decision = "HOLD"
        reason = "no_exit_trigger"

        # Rule 1: Take profit — good profit AND edge is gone
        if unrealized_pnl_pct >= self.TAKE_PROFIT_THRESHOLD and current_edge < self.EDGE_DECAY_EXIT:
            decision = "EXIT"
            reason = f"take_profit (pnl={unrealized_pnl_pct:.0%}, edge_left={current_edge:.1%})"

        # Rule 2: Stop loss — cut deep losses
        elif unrealized_pnl_pct <= self.STOP_LOSS_THRESHOLD:
            decision = "EXIT"
            reason = f"stop_loss (pnl={unrealized_pnl_pct:.0%})"

        # Rule 3: Expiry approaching with no edge
        elif time_to_expiry_hours <= self.EXPIRY_HOURS_THRESHOLD and current_edge < self.EXPIRY_EDGE_THRESHOLD:
            # Only exit near expiry if we have some profit or minimal loss
            if unrealized_pnl_pct > -0.10:  # Don't panic-sell huge losses at expiry
                decision = "EXIT"
                reason = f"near_expiry ({time_to_expiry_hours:.1f}h left, edge={current_edge:.1%})"

        # Rule 4: Edge completely gone (even without big profit)
        elif current_edge <= 0 and unrealized_pnl_pct > 0.10:
            decision = "EXIT"
            reason = f"edge_gone (pnl={unrealized_pnl_pct:.0%}, edge={current_edge:.1%})"

        return ExitDecision(
            trade_id=trade.trade_id,
            timestamp=now.isoformat(),
            source=trade.source,
            question=trade.question,
            side=trade.side,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl_pct=unrealized_pnl_pct,
            unrealized_pnl_usd=unrealized_pnl_usd,
            original_edge=trade.edge,
            current_edge=current_edge,
            time_held_hours=time_held_hours,
            time_to_expiry_hours=time_to_expiry_hours,
            decision=decision,
            reason=reason,
        )

    def _estimate_expiry_hours(self, trade) -> float:
        """Estimate hours until market closes/settles."""
        # Try to parse end_date from various sources
        # For now, use a default since we don't store end_date on TradeRecord
        # TODO: Store end_date on TradeRecord or look up from market data

        # Heuristic: Kalshi daily markets expire same day,
        # Polymarket varies. Default to 24h if unknown.
        try:
            # Check if trade has an end_date attribute (it won't yet, but future-proof)
            end_date = getattr(trade, "end_date", None)
            if end_date:
                expiry = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                remaining = (expiry - now).total_seconds() / 3600
                return max(0, remaining)
        except (ValueError, AttributeError):
            pass

        # Default: assume 24 hours for unknown markets
        return 24.0

    def record_outcome(self, trade_id: str, final_pnl: float):
        """
        After a trade settles, update the exit decisions to record
        whether HOLD or EXIT was the right call.

        Called by the settlement monitor.
        """
        for decision in self.decisions:
            if decision.trade_id == trade_id and decision.final_pnl is None:
                decision.final_pnl = final_pnl

                if decision.decision == "EXIT":
                    # We exited. Was that good?
                    # Good exit = we would have lost more by holding
                    # (approximate: if final settlement would have been worse)
                    # Since we actually exited, we locked in unrealized_pnl
                    # Compare: what we got (unrealized_pnl) vs what we would have gotten (final_pnl)
                    exit_pnl_pct = decision.unrealized_pnl_pct
                    hold_pnl_pct = final_pnl / max(decision.entry_price * (trade_id and 1 or 1), 0.01)  # approximate
                    decision.was_correct = exit_pnl_pct >= hold_pnl_pct
                elif decision.decision == "HOLD":
                    # We held. Was that good?
                    # Good hold = final pnl is better than what we had at decision time
                    hold_pnl_pct = final_pnl / max(decision.entry_price, 0.01)
                    decision.was_correct = hold_pnl_pct >= decision.unrealized_pnl_pct

        self._save_decisions()

    def get_learning_summary(self) -> dict:
        """
        Analyze past exit decisions to learn patterns.

        Returns summary stats useful for tuning thresholds
        or feeding to Claude in Phase 2.
        """
        evaluated = [d for d in self.decisions if d.was_correct is not None]
        if not evaluated:
            return {"total_decisions": len(self.decisions), "evaluated": 0}

        holds = [d for d in evaluated if d.decision == "HOLD"]
        exits = [d for d in evaluated if d.decision == "EXIT"]

        good_holds = [d for d in holds if d.was_correct]
        bad_holds = [d for d in holds if not d.was_correct]
        good_exits = [d for d in exits if d.was_correct]
        bad_exits = [d for d in exits if not d.was_correct]

        summary = {
            "total_decisions": len(self.decisions),
            "evaluated": len(evaluated),
            "holds": {
                "total": len(holds),
                "correct": len(good_holds),
                "wrong": len(bad_holds),
                "accuracy": len(good_holds) / max(len(holds), 1),
            },
            "exits": {
                "total": len(exits),
                "correct": len(good_exits),
                "wrong": len(bad_exits),
                "accuracy": len(good_exits) / max(len(exits), 1),
            },
        }

        # Patterns: which exit reasons work best?
        if exits:
            by_reason = {}
            for d in exits:
                reason_type = d.reason.split("(")[0].strip()
                if reason_type not in by_reason:
                    by_reason[reason_type] = {"total": 0, "correct": 0}
                by_reason[reason_type]["total"] += 1
                if d.was_correct:
                    by_reason[reason_type]["correct"] += 1
            summary["exit_reasons"] = by_reason

        # Patterns: at what profit levels are holds good/bad?
        if bad_holds:
            avg_bad_hold_pnl = sum(d.unrealized_pnl_pct for d in bad_holds) / len(bad_holds)
            summary["avg_pnl_at_bad_hold"] = avg_bad_hold_pnl

        return summary

    def get_learning_prompt(self) -> str:
        """
        Generate a prompt section for Claude with learning data.
        Used in Phase 2 for AI-assisted exit decisions.
        """
        summary = self.get_learning_summary()
        if summary.get("evaluated", 0) < 10:
            return ""  # Not enough data yet

        prompt = f"""
EXIT DECISION HISTORY (from {summary['evaluated']} evaluated positions):

Hold accuracy: {summary['holds']['accuracy']:.0%} ({summary['holds']['correct']}/{summary['holds']['total']})
Exit accuracy: {summary['exits']['accuracy']:.0%} ({summary['exits']['correct']}/{summary['exits']['total']})
"""
        if "exit_reasons" in summary:
            prompt += "\nExit reason effectiveness:\n"
            for reason, stats in summary["exit_reasons"].items():
                acc = stats["correct"] / max(stats["total"], 1)
                prompt += f"  {reason}: {acc:.0%} correct ({stats['correct']}/{stats['total']})\n"

        if "avg_pnl_at_bad_hold" in summary:
            prompt += f"\nAvg unrealized P&L when HOLD was wrong: {summary['avg_pnl_at_bad_hold']:.0%}\n"
            prompt += "(Consider lowering take-profit threshold)\n"

        return prompt

    def _load_decisions(self):
        """Load exit decision history from disk."""
        if os.path.exists(EXIT_LOG_FILE):
            try:
                with open(EXIT_LOG_FILE, "r") as f:
                    raw = json.load(f)
                self.decisions = [ExitDecision(**d) for d in raw]
                log.info("exit_decisions_loaded", count=len(self.decisions))
            except Exception:
                self.decisions = []

    def _save_decisions(self):
        """Save exit decision history to disk."""
        try:
            with open(EXIT_LOG_FILE, "w") as f:
                json.dump([asdict(d) for d in self.decisions[-500:]], f, indent=2)  # Keep last 500
        except Exception:
            pass

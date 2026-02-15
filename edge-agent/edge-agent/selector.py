"""
Hybrid Market Selector: Claude strategy + adaptive learning.

Phase 1 (Claude): Ask Claude which markets look promising given current
context, categories, and past performance data.

Phase 2 (Learning): Track which market patterns produce winning trades.
Feed this data BACK to Claude so it gets smarter over time.

This combines the best of both:
- Claude's reasoning about current events and market context
- Data-driven learning from actual trade outcomes
"""

import json
import math
import os
import random
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

import anthropic
import structlog

from models import Config, Market

log = structlog.get_logger()

MEMORY_FILE = "agent_memory.json"

COST_PER_INPUT_TOKEN = 3.0 / 1_000_000
COST_PER_OUTPUT_TOKEN = 15.0 / 1_000_000

STRATEGY_PROMPT = """You are a prediction market strategist selecting markets for an AI trading agent.

Current context:
- Date/time: {current_time}
- Agent bankroll: ${bankroll:.2f}
- Total markets available: {total_markets}
- Markets from: {sources}

LEARNED PERFORMANCE DATA (from past trades):
{performance_data}

Here are the available markets (sampled):
{market_sample}

RULES:
1. Pick 40-60 specific market IDs that are most likely to be MISPRICED right now. MORE IS BETTER.
2. Prioritize markets ENDING WITHIN 24-72 HOURS (faster resolution = faster learning).
3. Prioritize markets where we have DATA EDGE (our sources: NOAA weather, ESPN sports, Binance/CoinPaprika crypto).
4. Favor categories that have historically produced wins (see performance data above).
5. Avoid markets with spreads > 0.10 (too illiquid).
6. Include ALL crypto and sports markets — these are where our data sources give us an edge.
6. Mix: 70% high-confidence picks, 30% exploratory picks from underexplored categories.

Respond with ONLY valid JSON:
{{
  "picks": ["market_id_1", "market_id_2", "..."],
  "reasoning": "1-2 sentence strategy explanation"
}}"""


@dataclass
class PatternStats:
    """Tracks performance for a market pattern."""
    trades: int = 0
    wins: int = 0
    total_edge: float = 0.0
    total_pnl: float = 0.0
    times_analyzed: int = 0
    times_had_signal: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.trades, 1)

    @property
    def signal_rate(self) -> float:
        return self.times_had_signal / max(self.times_analyzed, 1)


class HybridSelector:
    """
    Claude picks markets, learning layer tracks outcomes.
    """

    def __init__(self, config: Config):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self.patterns: dict[str, PatternStats] = {}
        self.total_cost = 0.0
        self._load_memory()

    def select_markets(
        self,
        all_markets: list[Market],
        bankroll: float,
    ) -> list[Market]:
        """
        Select markets using Claude + learning data.
        Falls back to volume+expiry sorting if Claude fails.
        """
        if bankroll < 5:
            log.info("low_bankroll_skip_claude", bankroll=bankroll)
            return self._fallback_selection(all_markets)

        try:
            picks = self._claude_selection(all_markets, bankroll)
            if picks and len(picks) >= 5:
                log.info("hybrid_selection", claude_picks=len(picks))
                return picks
        except Exception as e:
            log.warning("claude_selection_failed", error=str(e))

        return self._fallback_selection(all_markets)

    def _claude_selection(self, markets: list[Market], bankroll: float) -> list[Market]:
        """Ask Claude to pick specific markets, informed by learning data."""
        # Build market sample — use short index IDs to save tokens
        market_entries = []
        id_map = {}  # index -> real condition_id
        for i, m in enumerate(markets[:300]):
            short_id = str(i)
            id_map[short_id] = m.condition_id
            hours_left = self._hours_until_end(m)
            hours_str = f"{hours_left:.0f}h" if hours_left is not None else "?"
            market_entries.append({
                "id": short_id,
                "q": m.question[:100],
                "price": round(m.mid_price, 2),
                "spread": round(m.spread, 3),
                "vol": round(m.volume_24h, 0),
                "ends": hours_str,
                "cat": m.category or "",
                "src": m.source,
            })

        market_sample = json.dumps(market_entries, separators=(",", ":"))[:4000]

        # Build performance summary from learning data
        performance = self._get_performance_summary()

        # Detect sources
        sources = set(m.source for m in markets)
        source_str = ", ".join(sources) if sources else "polymarket"

        prompt = STRATEGY_PROMPT.format(
            current_time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            bankroll=bankroll,
            total_markets=len(markets),
            sources=source_str,
            performance_data=performance,
            market_sample=market_sample,
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )

        cost = (
            response.usage.input_tokens * COST_PER_INPUT_TOKEN
            + response.usage.output_tokens * COST_PER_OUTPUT_TOKEN
        )
        self.total_cost += cost

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]

        data = json.loads(text)
        picked_ids_raw = set(str(x) for x in data.get("picks", []))
        reasoning = data.get("reasoning", "")

        log.info("claude_strategy", reasoning=reasoning[:120], picks=len(picked_ids_raw))

        # Map short IDs back to real condition_ids
        picked_real_ids = set()
        for short_id in picked_ids_raw:
            real_id = id_map.get(short_id)
            if real_id:
                picked_real_ids.add(real_id)

        # Match picked IDs to market objects
        market_map = {m.condition_id: m for m in markets}
        selected = [market_map[mid] for mid in picked_real_ids if mid in market_map]

        # If Claude picked too few, pad with high-volume markets ending soon
        if len(selected) < 40:
            remaining = [m for m in markets if m.condition_id not in picked_real_ids]
            remaining.sort(key=lambda m: self._urgency_volume_score(m), reverse=True)
            selected.extend(remaining[:60 - len(selected)])

        return selected

    def _fallback_selection(self, markets: list[Market]) -> list[Market]:
        """Volume + expiry based selection when Claude is unavailable."""
        scored = [(self._urgency_volume_score(m), m) for m in markets]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:80]]

    def _urgency_volume_score(self, market: Market) -> float:
        """Score by volume × urgency. Used for fallback and padding."""
        score = max(market.volume_24h, 1.0)

        hours = self._hours_until_end(market)
        if hours is not None:
            if hours <= 24:
                score *= 5.0
            elif hours <= 72:
                score *= 3.0
            elif hours <= 168:
                score *= 1.5

        # Penalize wide spreads
        if market.spread > 0.08:
            score *= 0.3

        return score

    def _get_performance_summary(self) -> str:
        """Summarize learned performance for Claude's context."""
        if not self.patterns:
            return "No trade history yet — exploring all categories."

        # Group by simplified category
        cat_stats = {}
        for key, stats in self.patterns.items():
            cat = key.split("|")[0] if "|" in key else key
            if cat not in cat_stats:
                cat_stats[cat] = PatternStats()
            cs = cat_stats[cat]
            cs.trades += stats.trades
            cs.wins += stats.wins
            cs.total_pnl += stats.total_pnl
            cs.times_analyzed += stats.times_analyzed
            cs.times_had_signal += stats.times_had_signal

        lines = []
        for cat, s in sorted(cat_stats.items(), key=lambda x: x[1].trades, reverse=True)[:8]:
            if s.trades > 0:
                lines.append(
                    f"  {cat}: {s.win_rate:.0%} win rate, {s.trades} trades, "
                    f"signal rate {s.signal_rate:.0%}, PnL ${s.total_pnl:.2f}"
                )
            elif s.times_analyzed > 0:
                lines.append(f"  {cat}: analyzed {s.times_analyzed}x, signal rate {s.signal_rate:.0%}, 0 trades yet")

        return "\n".join(lines) if lines else "No trade history yet."

    def record_analysis(self, market: Market, had_signal: bool, edge: float = 0.0):
        """Record that we analyzed a market."""
        key = self._pattern_key(market)
        if key not in self.patterns:
            self.patterns[key] = PatternStats()
        p = self.patterns[key]
        p.times_analyzed += 1
        if had_signal:
            p.times_had_signal += 1
            p.total_edge += edge
        self._save_memory()

    def record_trade(self, market: Market, won: bool, edge: float, pnl: float):
        """Record trade outcome."""
        key = self._pattern_key(market)
        if key not in self.patterns:
            self.patterns[key] = PatternStats()
        p = self.patterns[key]
        p.trades += 1
        if won:
            p.wins += 1
        p.total_pnl += pnl
        self._save_memory()

    def get_learned_insights(self) -> str:
        """Print what the agent has learned."""
        if not self.patterns:
            return "No data yet."

        ranked = sorted(
            self.patterns.items(),
            key=lambda x: x[1].trades,
            reverse=True,
        )

        lines = ["LEARNED PATTERNS:"]
        for key, p in ranked[:8]:
            if p.trades > 0:
                lines.append(
                    f"  {key}: {p.win_rate:.0%} win ({p.trades}t), "
                    f"signal {p.signal_rate:.0%} ({p.times_analyzed}a)"
                )
        return "\n".join(lines)

    def _pattern_key(self, market: Market) -> str:
        """Classify market into a learnable pattern."""
        parts = []
        cat = (market.category or "unknown").lower().strip()
        parts.append(cat)

        q = market.question.lower()
        if any(w in q for w in ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana"]):
            parts.append("crypto")
        elif any(w in q for w in ["nfl", "nba", "mlb", "nhl", "ufc", "game", "match"]):
            parts.append("sports")
        elif any(w in q for w in ["temperature", "weather", "hurricane", "snow"]):
            parts.append("weather")
        elif any(w in q for w in ["president", "election", "vote", "congress"]):
            parts.append("politics")
        else:
            parts.append("other")

        hours = self._hours_until_end(market)
        if hours is not None and hours <= 72:
            parts.append("soon")
        else:
            parts.append("later")

        parts.append(market.source)
        return "|".join(parts)

    def _hours_until_end(self, market: Market) -> Optional[float]:
        if not market.end_date:
            return None
        try:
            end_str = market.end_date
            if end_str.endswith("Z"):
                end_str = end_str.replace("Z", "+00:00")
            end_dt = datetime.fromisoformat(end_str)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            delta = end_dt - datetime.now(timezone.utc)
            return max(delta.total_seconds() / 3600, 0)
        except (ValueError, TypeError):
            return None

    def _load_memory(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r") as f:
                    data = json.load(f)
                for key, vals in data.get("patterns", {}).items():
                    self.patterns[key] = PatternStats(**vals)
                log.info("memory_loaded", patterns=len(self.patterns))
            except Exception:
                pass

    def _save_memory(self):
        data = {
            "last_saved": datetime.now(timezone.utc).isoformat(),
            "patterns": {
                key: {
                    "trades": p.trades,
                    "wins": p.wins,
                    "total_edge": p.total_edge,
                    "total_pnl": p.total_pnl,
                    "times_analyzed": p.times_analyzed,
                    "times_had_signal": p.times_had_signal,
                }
                for key, p in self.patterns.items()
            },
        }
        with open(MEMORY_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def get_cost(self) -> float:
        cost = self.total_cost
        self.total_cost = 0.0
        return cost

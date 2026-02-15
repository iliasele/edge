"""Position sizing using Kelly Criterion with safety caps."""

import structlog
from datetime import datetime, timezone

from models import Config, Market, FairValueEstimate, TradeSignal, Side
from typing import Optional

log = structlog.get_logger()


class PositionSizer:
    """
    Calculates position sizes using fractional Kelly Criterion.
    
    Kelly Criterion: f* = (p * b - q) / b
    where:
        p = probability of winning (our fair value)
        q = 1 - p (probability of losing)
        b = odds received (payout ratio)
    
    We use HALF-Kelly for safety (full Kelly is too aggressive).
    
    PRIORITY: Markets ending sooner are prioritized because:
    1. Less time for the market to correct → edge persists longer
    2. Faster capital turnover → more compound cycles
    3. Resolution is imminent → our data is most relevant NOW
    """

    def __init__(self, config: Config):
        self.config = config
        self.kelly_fraction = 0.5  # Half-Kelly during learning phase

    def find_trades(
        self,
        markets: list[Market],
        estimates: list[FairValueEstimate],
        bankroll: float,
    ) -> list[TradeSignal]:
        """
        Find mispriced markets and calculate position sizes.
        
        Returns trades sorted by URGENCY-WEIGHTED expected value:
        markets ending soon with good edge come first.
        """
        estimate_map = {e.market_id: e for e in estimates}
        signals = []
        
        # Log pre-analyzed estimates specifically
        for est in estimates:
            if est.key_factors and est.key_factors[0] in ("crypto_target", "crypto_direction_skip"):
                market = next((m for m in markets if m.condition_id == est.market_id), None)
                if market:
                    log.info(
                        "pre_estimate_in_sizer",
                        q=market.question[:60],
                        fair=est.fair_probability,
                        bid=round(market.best_bid, 3),
                        ask=round(market.best_ask, 3),
                        conf=est.confidence,
                        method=est.key_factors[0],
                    )

        for market in markets:
            est = estimate_map.get(market.condition_id)
            if not est:
                continue

            signal = self._evaluate_trade(market, est, bankroll)
            if signal:
                signals.append(signal)
        
        log.info("sizer_stats", total_markets=len(markets), total_estimates=len(estimates), matched=len(markets) - sum(1 for m in markets if m.condition_id not in estimate_map))

        # Sort by urgency-weighted score (ending soon + high EV = best)
        signals.sort(key=lambda s: self._urgency_score(s), reverse=True)

        log.info("trade_signals_found", count=len(signals))
        return signals

    def _urgency_score(self, signal: TradeSignal) -> float:
        """
        Score that combines expected value with time urgency.
        
        Markets ending within 24h get 3x weight.
        Markets ending within 72h get 2x weight.
        Markets ending within 7d get 1.5x weight.
        Others get 1x weight.
        """
        hours_left = self._hours_until_end(signal.market)
        
        if hours_left is not None:
            if hours_left <= 24:
                urgency_multiplier = 3.0
            elif hours_left <= 72:
                urgency_multiplier = 2.0
            elif hours_left <= 168:  # 7 days
                urgency_multiplier = 1.5
            else:
                urgency_multiplier = 1.0
        else:
            urgency_multiplier = 1.0  # Unknown end date

        return signal.expected_value * urgency_multiplier

    def _hours_until_end(self, market: Market) -> Optional[float]:
        """Calculate hours until market ends."""
        if not market.end_date:
            return None
        try:
            end_str = market.end_date
            # Handle various date formats
            if end_str.endswith("Z"):
                end_str = end_str.replace("Z", "+00:00")
            end_dt = datetime.fromisoformat(end_str)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            delta = end_dt - datetime.now(timezone.utc)
            return max(delta.total_seconds() / 3600, 0)
        except (ValueError, TypeError):
            return None

    def _evaluate_trade(
        self, market: Market, estimate: FairValueEstimate, bankroll: float
    ) -> Optional[TradeSignal]:
        """Evaluate a single market for trading opportunity."""
        fair_p = estimate.fair_probability
        confidence = estimate.confidence

        # Determine trade direction
        # If fair value > ask price → BUY YES (market underpricing YES)
        # If fair value < bid price → SELL YES / BUY NO (market overpricing YES)

        if fair_p > market.best_ask:
            # Buy YES
            side = Side.BUY
            entry_price = market.best_ask
            edge = fair_p - entry_price
        elif fair_p < market.best_bid:
            # Sell YES (buy NO)
            side = Side.SELL
            entry_price = market.best_bid
            edge = entry_price - fair_p
        else:
            # Fair value is within the spread - no trade
            log.debug(
                "no_edge_in_spread",
                q=market.question[:60],
                fair=round(fair_p, 3),
                bid=round(market.best_bid, 3),
                ask=round(market.best_ask, 3),
                conf=round(confidence, 2),
            )
            return None

        # Minimum edge threshold (per-source)
        source = getattr(market, "source", "polymarket")
        min_edge = self.config.min_edge_pct
        min_confidence = 0.15  # Low during learning - let trades through

        if source == "kalshi":
            min_edge = min(min_edge, 0.01)
            min_confidence = 0.15

        if edge < min_edge - 0.001:  # Small epsilon for floating point
            log.debug("edge_too_small", q=market.question[:50], edge=f"{edge:.1%}", min=f"{min_edge:.1%}", source=source)
            return None

        # Low confidence → skip
        if confidence < min_confidence:
            log.debug("low_confidence", q=market.question[:50], confidence=round(confidence, 2), source=source)
            return None

        # Kelly Criterion calculation
        # For binary markets: buying at price p, winning pays 1, losing pays 0
        if side == Side.BUY:
            # Buying YES at `entry_price`, pays $1 if YES
            win_prob = fair_p
            # Net profit per dollar risked: (1 - entry_price) / entry_price
            b = (1.0 - entry_price) / entry_price
        else:
            # Selling YES at `entry_price` (equivalent to buying NO)
            win_prob = 1.0 - fair_p
            b = entry_price / (1.0 - entry_price)

        lose_prob = 1.0 - win_prob
        kelly_raw = (win_prob * b - lose_prob) / b

        if kelly_raw <= 0:
            log.debug("negative_kelly", q=market.question[:50], kelly_raw=round(kelly_raw, 4), 
                      fair=round(fair_p, 3), side=side.value, entry=round(entry_price, 3))
            return None

        # Apply half-Kelly and confidence scaling
        kelly_adjusted = kelly_raw * self.kelly_fraction * confidence

        # Cap at max bankroll percentage
        kelly_capped = min(kelly_adjusted, self.config.max_bankroll_pct)

        # Position size in USD
        position_size = bankroll * kelly_capped

        # Minimum trade size — avoid dust bets that waste API calls
        # Kalshi minimum is 1 contract, but 1 contract at 7¢ = $0.07 profit max
        # Not worth the execution overhead
        min_trade = 1.0  # Always require at least $1 position
        if position_size < min_trade:
            log.debug("position_too_small", q=market.question[:50], size=f"${position_size:.2f}",
                      kelly=round(kelly_capped, 4), bankroll=round(bankroll, 2))
            return None

        # Expected value
        ev = position_size * edge

        return TradeSignal(
            market=market,
            estimate=estimate,
            side=side,
            edge=round(edge, 4),
            kelly_fraction=round(kelly_capped, 4),
            position_size_usd=round(position_size, 2),
            expected_value=round(ev, 4),
        )

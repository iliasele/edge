"""
Pre-Analyzer: Computes fair values for markets where we can use MATH instead of Claude.

For crypto "above/below $X" markets, we don't need an LLM to tell us the probability.
We have the current price, volatility, and time to expiry — that's a Black-Scholes-like calc.

This runs BEFORE the Claude analyst and overrides its estimates for markets where
we have high-confidence mathematical estimates.
"""

import re
import math
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass

import structlog

from models import Market, FairValueEstimate
from enricher import EnrichedMarket

log = structlog.get_logger()


@dataclass
class PreAnalysis:
    """Result of pre-analysis. If computed, this overrides Claude."""
    market_id: str
    fair_probability: float
    confidence: float
    reasoning: str
    method: str  # "crypto_target", "expired", etc.


class PreAnalyzer:
    """
    Computes fair values for markets where math > LLM.
    
    Currently handles:
    1. Crypto price target markets ("Will BTC be above $X by date?")
    2. Already-expired markets
    3. Near-certain markets (price already past target with time left)
    """

    def analyze(
        self,
        markets: list[Market],
        enriched: list[Optional[EnrichedMarket]],
    ) -> dict[str, PreAnalysis]:
        """
        Returns a dict of market_id -> PreAnalysis for markets we can handle.
        Markets NOT in the dict should be sent to Claude.
        """
        results = {}
        enrichment_map = {e.market.condition_id: e for e in enriched if e}

        for market in markets:
            enrichment = enrichment_map.get(market.condition_id)
            
            # Try crypto target analysis
            if enrichment and enrichment.domain == "crypto" and enrichment.crypto_data:
                result = self._analyze_crypto_target(market, enrichment)
                if result:
                    results[market.condition_id] = result
                    continue

            # Try "up or down" analysis
            if enrichment and enrichment.domain == "crypto" and enrichment.crypto_data:
                result = self._analyze_crypto_direction(market, enrichment)
                if result:
                    results[market.condition_id] = result

        if results:
            log.info("pre_analysis_complete", computed=len(results))
        return results

    def _analyze_crypto_target(
        self, market: Market, enrichment: EnrichedMarket
    ) -> Optional[PreAnalysis]:
        """
        For markets like "Will Bitcoin be above $85,000 in February?"
        
        Uses log-normal model:
        - Current price from Binance
        - Historical daily volatility from 30d change
        - Time to expiry
        - Compute probability of crossing target
        """
        question = market.question.lower()
        
        # Must be a "reach/above/below $X" type question
        target_match = re.search(
            r'(?:above|reach|below|exceed|hit|surpass|over|under)\s*\$?([\d,]+(?:\.\d+)?)',
            question
        )
        if not target_match:
            return None

        target_price = float(target_match.group(1).replace(",", ""))
        is_above = not any(w in question for w in ["below", "under"])

        # Get current price from enrichment
        crypto_data = enrichment.crypto_data
        if not crypto_data or not crypto_data.coin_metrics:
            return None

        coin = crypto_data.coin_metrics[0]
        current_price = coin.current_price
        if current_price <= 0:
            return None

        # Calculate days to expiry
        days_left = self._days_until_end(market)
        if days_left is None or days_left < 0:
            return None

        # If already past target
        if is_above and current_price > target_price:
            # Already above — but could dip before expiry
            # The further above, the safer
            pct_above = (current_price - target_price) / target_price
            if pct_above > 0.15:  # >15% above target
                prob = 0.96
                conf = 0.85
            elif pct_above > 0.08:  # >8% above
                prob = 0.92
                conf = 0.80
            elif pct_above > 0.03:  # >3% above
                prob = 0.85
                conf = 0.75
            elif pct_above > 0.01:  # >1% above
                prob = 0.75
                conf = 0.65
            else:  # barely above (<1%)
                prob = 0.62
                conf = 0.55
            
            # More time = more risk of dipping
            if days_left > 14:
                prob -= 0.05
            
            reasoning = (
                f"{coin.name} at ${current_price:,.0f}, {pct_above:.1%} above target ${target_price:,.0f}. "
                f"{days_left:.0f} days left. Already cleared target."
            )
        elif not is_above and current_price < target_price:
            # Already below target for a "below" question
            pct_below = (target_price - current_price) / target_price
            if pct_below > 0.15:
                prob = 0.95
                conf = 0.85
            elif pct_below > 0.08:
                prob = 0.90
                conf = 0.80
            else:
                prob = 0.75
                conf = 0.65
            reasoning = f"{coin.name} at ${current_price:,.0f}, already below ${target_price:,.0f}."
        else:
            # Need to move to hit target
            pct_needed = abs(target_price - current_price) / current_price
            
            # Estimate daily volatility from 30d data
            daily_vol = self._estimate_daily_vol(coin)
            
            if days_left < 0.1:
                # Expires in minutes/hours
                prob = 0.15 if pct_needed > 0.02 else 0.45
                conf = 0.60
            elif days_left < 1:
                # Less than a day
                max_likely_move = daily_vol * 1.5
                if pct_needed > max_likely_move:
                    prob = 0.10
                else:
                    prob = max(0.15, 0.50 - (pct_needed / max_likely_move) * 0.40)
                conf = 0.55
            else:
                # Multi-day: use simplified model
                # Expected range ~ daily_vol * sqrt(days)
                expected_range = daily_vol * math.sqrt(days_left)
                z_score = pct_needed / expected_range if expected_range > 0 else 5
                
                # Approximate normal CDF
                prob = self._norm_cdf(-z_score) if is_above else self._norm_cdf(z_score)
                
                # Add momentum adjustment
                momentum = coin.price_change_7d_pct / 100
                if (is_above and momentum > 0) or (not is_above and momentum < 0):
                    prob = min(prob + 0.05, 0.95)  # Momentum helps
                elif (is_above and momentum < -0.05) or (not is_above and momentum > 0.05):
                    prob = max(prob - 0.05, 0.05)  # Momentum hurts
                
                conf = 0.65
            
            direction = "above" if is_above else "below"
            reasoning = (
                f"{coin.name} at ${current_price:,.0f}, needs {pct_needed:.1%} move to {direction} "
                f"${target_price:,.0f}. {days_left:.0f}d left. Daily vol ~{daily_vol:.1%}. "
                f"7d momentum: {coin.price_change_7d_pct:+.1f}%."
            )

        if not is_above:
            prob = 1 - prob  # Invert for "below" questions

        log.info(
            "pre_analyzed_crypto",
            q=market.question[:60],
            price=f"${current_price:,.0f}",
            target=f"${target_price:,.0f}",
            fair=round(prob, 3),
            market_mid=round(market.mid_price, 3),
            edge=f"{abs(prob - market.mid_price):.1%}",
        )

        return PreAnalysis(
            market_id=market.condition_id,
            fair_probability=round(prob, 3),
            confidence=round(conf, 2),
            reasoning=reasoning,
            method="crypto_target",
        )

    def _analyze_crypto_direction(
        self, market: Market, enrichment: EnrichedMarket
    ) -> Optional[PreAnalysis]:
        """
        For "Up or Down" markets — these are near-random, skip them.
        Return low confidence so the sizer filters them out.
        """
        question = market.question.lower()
        if "up or down" not in question:
            return None

        return PreAnalysis(
            market_id=market.condition_id,
            fair_probability=0.50,
            confidence=0.10,  # Very low — will be filtered by sizer
            reasoning="Short-term direction is near-random. No edge.",
            method="crypto_direction_skip",
        )

    def _estimate_daily_vol(self, coin) -> float:
        """Estimate daily volatility from available data."""
        # Use 30d change to estimate daily vol
        # Rough: daily_vol ≈ |30d_change| / sqrt(30)
        if abs(coin.price_change_30d_pct) > 0:
            daily_vol = abs(coin.price_change_30d_pct / 100) / math.sqrt(30)
        elif abs(coin.price_change_7d_pct) > 0:
            daily_vol = abs(coin.price_change_7d_pct / 100) / math.sqrt(7)
        else:
            daily_vol = 0.025  # Default 2.5% daily for crypto
        
        # Crypto is volatile, floor at 1.5%
        return max(daily_vol, 0.015)

    def _days_until_end(self, market: Market) -> Optional[float]:
        """Calculate days until market ends."""
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
            return max(delta.total_seconds() / 86400, 0)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Approximate standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

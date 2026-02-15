"""
Analyst v2: Uses Claude + real-world data to estimate fair probabilities.

UPGRADE: Instead of just sending Claude the question text, we now attach
domain-specific data (NOAA forecasts, injury reports, on-chain metrics).
This is the difference between "guess" and "informed estimate".
"""

import json
import time
import anthropic
import structlog

from models import Config, Market, FairValueEstimate
from enricher import EnrichedMarket
from typing import Optional

log = structlog.get_logger()

# Cost tracking (Sonnet pricing)
COST_PER_INPUT_TOKEN = 3.0 / 1_000_000   # $3 per 1M input tokens
COST_PER_OUTPUT_TOKEN = 15.0 / 1_000_000  # $15 per 1M output tokens


ENRICHED_ANALYSIS_PROMPT = """You are a prediction market trader looking for MISPRICED markets.

You have REAL-TIME DATA that most market participants may not have priced in yet.

For each market below:
1. Look at the question, the current market price, and the real-time data provided
2. Use the data to estimate the TRUE probability of YES
3. If your estimate differs from the market by 3% or more, that's a trade

KEY DATA SIGNALS:
- CRYPTO: You are given the CURRENT PRICE from Binance. For "Will BTC be above $X?" questions, if BTC is currently well above $X with 2+ weeks left, the probability should be HIGH (85-95%). If BTC needs to move 10%+ in days, probability should be LOW (10-25%). Use the 7d/30d momentum to judge trend.
- SPORTS: Injury data is the #1 edge. If a star player is OUT and the market hasn't adjusted, that's 5-15% edge. Sportsbook spreads are your baseline — adjust for injuries.
- WEATHER: NOAA forecasts for 0-3 days are very reliable. Use them directly.
- POLITICS/GENERAL: Use base rates. Rare events should be priced low (<5%) unless specific evidence.

BE PRECISE WITH CRYPTO:
- "Bitcoin above $85k" when BTC is at $97k with 14 days left → ~88% (it could dip but unlikely to drop 12%)
- "Bitcoin above $100k" when BTC is at $97k with 2 days left → ~35% (needs 3% move up)
- "Ethereum up or down today" with no clear signal → 50% (coin flip, LOW confidence)

CONFIDENCE = how much your data actually helps:
- 0.7-0.9: Your data directly answers the question (current price vs target, NOAA forecast)
- 0.4-0.6: Data is relevant but uncertain (momentum, injury impact)
- 0.1-0.3: No useful data, you're guessing

Markets:
{markets_json}

Respond with ONLY valid JSON (no markdown):
{{
  "estimates": [
    {{
      "market_id": "the condition_id",
      "fair_probability": 0.XX,
      "confidence": 0.XX,
      "reasoning": "1-2 sentences",
      "key_factors": ["factor1", "factor2"],
      "data_edge": "what data gives us edge"
    }}
  ]
}}"""


class Analyst:
    """Uses Claude + enrichment data to build fair value estimates."""

    def __init__(self, config: Config):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self.total_cost = 0.0
        self.total_calls = 0

    def estimate_batch(
        self,
        markets: list[Market],
        enriched: list[EnrichedMarket] = None,
        batch_size: int = 12,  # Slightly smaller batches since enrichment adds tokens
    ) -> list[FairValueEstimate]:
        """
        Estimate fair values for a batch of markets.
        If enriched data is provided, uses the enriched prompt for better accuracy.
        """
        all_estimates = []

        # Build enrichment lookup
        enrichment_map = {}
        if enriched:
            enrichment_map = {e.market.condition_id: e for e in enriched}

        for i in range(0, len(markets), batch_size):
            batch = markets[i : i + batch_size]
            try:
                batch_enriched = [enrichment_map.get(m.condition_id) for m in batch]
                has_enrichment = any(e is not None for e in batch_enriched)

                if has_enrichment:
                    estimates = self._analyze_enriched_batch(batch, batch_enriched)
                else:
                    estimates = self._analyze_basic_batch(batch)

                all_estimates.extend(estimates)
            except Exception as e:
                log.error("batch_analysis_failed", batch_start=i, error=str(e))
                continue

            if i + batch_size < len(markets):
                time.sleep(1)

        log.info(
            "analysis_complete",
            markets_analyzed=len(all_estimates),
            total_api_cost=f"${self.total_cost:.4f}",
        )
        return all_estimates

    def _analyze_enriched_batch(
        self, markets: list[Market], enriched: list[Optional[EnrichedMarket]]
    ) -> list[FairValueEstimate]:
        """Analyze markets WITH enrichment data — the high-accuracy path."""
        markets_json = json.dumps(
            [
                {
                    "condition_id": m.condition_id,
                    "question": m.question,
                    "description": m.description[:150],
                    "current_yes_price": round(m.mid_price, 3),
                    "best_bid": round(m.best_bid, 3),
                    "best_ask": round(m.best_ask, 3),
                    "volume_24h": round(m.volume_24h, 2),
                    "end_date": m.end_date,
                    "domain": e.domain if e else "general",
                    "real_time_data": (e.enrichment_summary[:500] if e and e.enrichment_summary else "None available"),
                }
                for m, e in zip(markets, enriched)
            ],
            indent=2,
        )

        prompt = ENRICHED_ANALYSIS_PROMPT.format(markets_json=markets_json)
        return self._call_claude(prompt, markets)

    def _analyze_basic_batch(self, markets: list[Market]) -> list[FairValueEstimate]:
        """Analyze markets WITHOUT enrichment data — the fallback path."""
        basic_prompt = """You are a prediction market analyst. Estimate the TRUE probability of "Yes" for each market.
Be calibrated. Consider base rates and common biases. Markets often have a favorite-longshot bias.

Respond with ONLY valid JSON.

Markets:
{markets_json}

JSON structure:
{{
  "estimates": [
    {{
      "market_id": "the condition_id",
      "fair_probability": 0.XX,
      "confidence": 0.XX,
      "reasoning": "1-2 sentence explanation",
      "key_factors": ["factor1", "factor2"]
    }}
  ]
}}"""
        markets_json = json.dumps(
            [
                {
                    "condition_id": m.condition_id,
                    "question": m.question,
                    "description": m.description[:200],
                    "current_yes_price": round(m.mid_price, 3),
                    "best_bid": round(m.best_bid, 3),
                    "best_ask": round(m.best_ask, 3),
                    "volume_24h": round(m.volume_24h, 2),
                    "end_date": m.end_date,
                }
                for m in markets
            ],
            indent=2,
        )

        prompt = basic_prompt.format(markets_json=markets_json)
        return self._call_claude(prompt, markets)

    def _call_claude(self, prompt: str, markets: list[Market]) -> list[FairValueEstimate]:
        """Send prompt to Claude and parse response."""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2500,
            temperature=0.5,  # Slightly higher for more independent estimates
            messages=[{"role": "user", "content": prompt}],
        )

        # Track costs
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        call_cost = (input_tokens * COST_PER_INPUT_TOKEN) + (output_tokens * COST_PER_OUTPUT_TOKEN)
        self.total_cost += call_cost
        self.total_calls += 1

        log.debug(
            "api_call",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=f"${call_cost:.4f}",
        )

        # Parse response
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]

        data = json.loads(text)
        market_map = {m.condition_id: m for m in markets}

        estimates = []
        for est in data.get("estimates", []):
            market_id = est["market_id"]
            if market_id not in market_map:
                continue

            estimates.append(
                FairValueEstimate(
                    market_id=market_id,
                    question=market_map[market_id].question,
                    fair_probability=float(est["fair_probability"]),
                    confidence=float(est["confidence"]),
                    reasoning=est.get("reasoning", ""),
                    key_factors=est.get("key_factors", []),
                )
            )

        return estimates

    def get_session_cost(self) -> float:
        return self.total_cost

"""
Data Enricher: Routes markets to the right data sources and 
combines enrichment data into a format Claude can use.

This is the KEY module that gives the agent its edge.
Instead of Claude guessing from just the question text,
it now gets real-time data: NOAA forecasts, injury reports,
on-chain metrics, etc.
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import structlog

from models import Market
from sources.weather import WeatherSource, WeatherData
from sources.sports import SportsSource, SportsData
from sources.crypto import CryptoSource, CryptoData

log = structlog.get_logger()


@dataclass
class EnrichedMarket:
    """A market with all available enrichment data attached."""
    market: Market
    domain: str  # "weather", "sports", "crypto", "politics", "general"
    weather_data: Optional[WeatherData] = None
    sports_data: Optional[SportsData] = None
    crypto_data: Optional[CryptoData] = None
    enrichment_summary: str = ""  # Condensed text for Claude's prompt


class DataEnricher:
    """
    Classifies markets by domain and fetches relevant real-world data.
    
    This is what separates a profitable agent from one that just asks
    Claude to guess: we're giving Claude DATA, not just questions.
    """

    def __init__(self):
        self.weather = WeatherSource()
        self.sports = SportsSource()
        self.crypto = CryptoSource()
        self._enrichment_stats = {"weather": 0, "sports": 0, "crypto": 0, "general": 0}

    def enrich_markets(self, markets: list[Market], max_workers: int = 4) -> list[EnrichedMarket]:
        """
        Enrich a batch of markets with domain-specific data.
        Uses threading for parallel API calls.
        """
        start = time.time()
        enriched = []

        # Process in parallel for speed
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._enrich_single, market): market
                for market in markets
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    enriched.append(result)
                except Exception as e:
                    market = futures[future]
                    log.warning("enrichment_failed", market=market.question[:50], error=str(e))
                    # Still include the market, just without enrichment
                    enriched.append(EnrichedMarket(
                        market=market,
                        domain="general",
                        enrichment_summary="No enrichment data available.",
                    ))

        elapsed = time.time() - start
        log.info(
            "enrichment_complete",
            total=len(enriched),
            elapsed=f"{elapsed:.1f}s",
            stats=self._enrichment_stats,
        )
        return enriched

    def _enrich_single(self, market: Market) -> EnrichedMarket:
        """Classify and enrich a single market."""
        question = market.question
        description = market.description

        result = EnrichedMarket(market=market, domain="general")

        # Classify and enrich by domain
        # Check in priority order (a market could match multiple)
        
        if self.weather.is_weather_market(question):
            result.domain = "weather"
            self._enrichment_stats["weather"] += 1
            data = self.weather.enrich(market.condition_id, question, description)
            if data:
                result.weather_data = data
                result.enrichment_summary = self._format_weather(data)

        elif self.sports.is_sports_market(question):
            result.domain = "sports"
            self._enrichment_stats["sports"] += 1
            data = self.sports.enrich(market.condition_id, question, description)
            if data:
                result.sports_data = data
                result.enrichment_summary = self._format_sports(data)

        elif self.crypto.is_crypto_market(question):
            result.domain = "crypto"
            self._enrichment_stats["crypto"] += 1
            data = self.crypto.enrich(market.condition_id, question, description)
            if data:
                result.crypto_data = data
                result.enrichment_summary = data.market_summary

        else:
            result.domain = "general"
            self._enrichment_stats["general"] += 1
            result.enrichment_summary = ""

        return result

    def _format_weather(self, data: WeatherData) -> str:
        """Format weather data into a concise context string for Claude."""
        parts = []

        if data.location:
            parts.append(f"Location: {data.location.title()}")

        if data.forecast_summary:
            parts.append(f"NOAA Forecast: {data.forecast_summary[:300]}")

        if data.temperature_forecast:
            tf = data.temperature_forecast
            parts.append(f"Temp range: {tf.get('low')}°F - {tf.get('high')}°F (avg {tf.get('avg')}°F)")

        if data.alerts:
            parts.append(f"ACTIVE ALERTS: {' | '.join(data.alerts[:3])}")

        if data.hurricane_outlook:
            parts.append(f"Tropical outlook: {data.hurricane_outlook[:200]}")

        return " | ".join(parts)

    def _format_sports(self, data: SportsData) -> str:
        """Format sports data into a concise context string for Claude."""
        parts = []

        parts.append(f"Sport: {data.sport.upper()} ({data.league})")

        if data.teams_detected:
            parts.append(f"Teams: {', '.join(data.teams_detected)}")

        # Injuries are the highest-value signal
        if data.injuries:
            key_injuries = [
                f"{inj.player_name} ({inj.team}): {inj.status} - {inj.injury}"
                for inj in data.injuries
                if inj.status in ("Out", "OUT", "Doubtful", "DOUBTFUL")
            ][:5]  # Top 5 key injuries
            if key_injuries:
                parts.append(f"KEY INJURIES: {' | '.join(key_injuries)}")

            all_injuries = [
                f"{inj.player_name}: {inj.status}"
                for inj in data.injuries
            ][:10]
            parts.append(f"All injuries: {', '.join(all_injuries)}")

        if data.upcoming_games:
            games_str = []
            for g in data.upcoming_games[:3]:
                odds_str = ""
                if g.odds:
                    odds_str = f" (spread: {g.odds.get('spread', 'N/A')}, O/U: {g.odds.get('over_under', 'N/A')})"
                games_str.append(f"{g.away_team} @ {g.home_team}{odds_str}")
            parts.append(f"Upcoming: {' | '.join(games_str)}")

        if data.recent_results:
            results_str = [
                f"{g.away_team} {g.away_score} - {g.home_score} {g.home_team}"
                for g in data.recent_results[:3]
            ]
            parts.append(f"Recent: {' | '.join(results_str)}")

        if data.news_headlines:
            parts.append(f"Headlines: {' | '.join(data.news_headlines[:3])}")

        return " | ".join(parts)

    def get_stats(self) -> dict:
        return dict(self._enrichment_stats)

    def close(self):
        self.weather.close()
        self.sports.close()
        self.crypto.close()

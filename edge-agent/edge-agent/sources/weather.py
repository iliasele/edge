"""
Weather Data Source: Parses NOAA forecasts for weather-related markets.

NOAA's API is FREE, no key required for basic access.
The edge: NOAA updates forecasts every 1-6 hours. If we parse them
before Polymarket traders react, we can find mispricing on weather markets.

Covers:
- Temperature records ("Will NYC hit 100°F?")
- Hurricane/storm markets ("Will a Cat 5 hit Florida?")
- Precipitation/snowfall ("Will Chicago get 12+ inches of snow?")
- Seasonal outlook markets
"""

import re
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional

import httpx
import structlog

log = structlog.get_logger()

NOAA_API_BASE = "https://api.weather.gov"
NOAA_CPC_BASE = "https://www.cpc.ncep.noaa.gov"

# Common locations in Polymarket weather markets
# Maps city names → NOAA grid coordinates
LOCATION_GRID = {
    "new york": {"office": "OKX", "gridX": 33, "gridY": 37, "lat": 40.7128, "lon": -74.0060},
    "nyc": {"office": "OKX", "gridX": 33, "gridY": 37, "lat": 40.7128, "lon": -74.0060},
    "los angeles": {"office": "LOX", "gridX": 154, "gridY": 44, "lat": 34.0522, "lon": -118.2437},
    "chicago": {"office": "LOT", "gridX": 65, "gridY": 76, "lat": 41.8781, "lon": -87.6298},
    "miami": {"office": "MFL", "gridX": 110, "gridY": 50, "lat": 25.7617, "lon": -80.1918},
    "houston": {"office": "HGX", "gridX": 65, "gridY": 97, "lat": 29.7604, "lon": -95.3698},
    "phoenix": {"office": "PSR", "gridX": 159, "gridY": 57, "lat": 33.4484, "lon": -112.0740},
    "denver": {"office": "BOU", "gridX": 62, "gridY": 60, "lat": 39.7392, "lon": -104.9903},
    "seattle": {"office": "SEW", "gridX": 124, "gridY": 67, "lat": 47.6062, "lon": -122.3321},
    "washington": {"office": "LWX", "gridX": 97, "gridY": 71, "lat": 38.9072, "lon": -77.0369},
    "dc": {"office": "LWX", "gridX": 97, "gridY": 71, "lat": 38.9072, "lon": -77.0369},
    "atlanta": {"office": "FFC", "gridX": 50, "gridY": 86, "lat": 33.749, "lon": -84.388},
    "dallas": {"office": "FWD", "gridX": 79, "gridY": 108, "lat": 32.7767, "lon": -96.7970},
    "boston": {"office": "BOX", "gridX": 71, "gridY": 90, "lat": 42.3601, "lon": -71.0589},
    "san francisco": {"office": "MTR", "gridX": 85, "gridY": 105, "lat": 37.7749, "lon": -122.4194},
}

# Keywords that indicate a weather market
WEATHER_KEYWORDS = [
    "temperature", "degrees", "°f", "°c", "fahrenheit", "celsius",
    "hurricane", "tropical storm", "cyclone", "typhoon",
    "snow", "snowfall", "blizzard", "ice storm",
    "rain", "rainfall", "precipitation", "flood",
    "heat wave", "heatwave", "cold snap", "freeze",
    "tornado", "wildfire", "drought",
    "hottest", "coldest", "warmest", "record high", "record low",
    "weather", "el nino", "la nina", "el niño", "la niña",
    "category 5", "cat 5", "cat 4", "major hurricane",
]


@dataclass
class WeatherData:
    """Enrichment data for a weather market."""
    market_id: str
    location: Optional[str] = None
    forecast_summary: str = ""
    temperature_forecast: Optional[dict] = None  # {high, low, avg} for relevant period
    precipitation_forecast: Optional[dict] = None
    alerts: list[str] = field(default_factory=list)
    hurricane_outlook: Optional[str] = None
    seasonal_outlook: Optional[str] = None
    data_timestamp: str = ""
    source: str = "NOAA"


class WeatherSource:
    """Fetches and parses NOAA weather data for market enrichment."""

    def __init__(self):
        self.client = httpx.Client(
            timeout=15.0,
            headers={"User-Agent": "(polymarket-agent, contact@example.com)"},  # NOAA requires User-Agent
        )
        self._cache: dict[str, tuple[float, any]] = {}  # Simple TTL cache
        self._cache_ttl = 1800  # 30 min cache

    def is_weather_market(self, question: str) -> bool:
        """Check if a market question is weather-related."""
        q_lower = question.lower()
        return any(kw in q_lower for kw in WEATHER_KEYWORDS)

    def enrich(self, market_id: str, question: str, description: str = "") -> Optional[WeatherData]:
        """
        Fetch relevant weather data for a market question.
        Returns WeatherData or None if no relevant data found.
        """
        try:
            q_lower = question.lower()
            location = self._extract_location(q_lower)
            data = WeatherData(
                market_id=market_id,
                location=location,
                data_timestamp=datetime.now(timezone.utc).isoformat(),
            )

            if location and location in LOCATION_GRID:
                grid = LOCATION_GRID[location]

                # Get point forecast
                forecast = self._get_forecast(grid)
                if forecast:
                    data.forecast_summary = forecast.get("summary", "")
                    data.temperature_forecast = forecast.get("temperatures")
                    data.precipitation_forecast = forecast.get("precipitation")

                # Get active alerts
                alerts = self._get_alerts(grid["lat"], grid["lon"])
                data.alerts = alerts

            # Hurricane markets
            if any(kw in q_lower for kw in ["hurricane", "tropical storm", "cyclone", "cat 5", "category"]):
                data.hurricane_outlook = self._get_hurricane_outlook()

            # Only return if we actually got useful data
            if data.forecast_summary or data.alerts or data.hurricane_outlook:
                log.debug("weather_enriched", market_id=market_id, location=location)
                return data

            return None

        except Exception as e:
            log.warning("weather_enrich_failed", market_id=market_id, error=str(e))
            return None

    def _extract_location(self, question: str) -> Optional[str]:
        """Extract city/location from question text."""
        for city in LOCATION_GRID:
            if city in question:
                return city
        return None

    def _get_forecast(self, grid: dict) -> Optional[dict]:
        """Get NOAA grid point forecast."""
        cache_key = f"forecast_{grid['office']}_{grid['gridX']}_{grid['gridY']}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            url = f"{NOAA_API_BASE}/gridpoints/{grid['office']}/{grid['gridX']},{grid['gridY']}/forecast"
            resp = self.client.get(url)
            resp.raise_for_status()
            data = resp.json()

            periods = data.get("properties", {}).get("periods", [])
            if not periods:
                return None

            # Extract next 7 days of forecasts
            temps = []
            precip_mentions = []
            summaries = []

            for period in periods[:14]:  # 14 periods = 7 days (day + night)
                temp = period.get("temperature")
                if temp is not None:
                    temps.append(temp)
                detail = period.get("detailedForecast", "")
                summaries.append(f"{period.get('name', '')}: {period.get('shortForecast', '')} ({temp}°{period.get('temperatureUnit', 'F')})")
                
                # Check for precip
                pop = period.get("probabilityOfPrecipitation", {}).get("value")
                if pop and pop > 0:
                    precip_mentions.append(f"{period.get('name')}: {pop}% chance")

            result = {
                "summary": " | ".join(summaries[:6]),  # Next 3 days
                "temperatures": {
                    "high": max(temps) if temps else None,
                    "low": min(temps) if temps else None,
                    "avg": round(sum(temps) / len(temps), 1) if temps else None,
                    "all_temps": temps[:14],
                },
                "precipitation": {
                    "mentions": precip_mentions[:7],
                },
            }

            self._set_cached(cache_key, result)
            return result

        except Exception as e:
            log.debug("noaa_forecast_failed", error=str(e))
            return None

    def _get_alerts(self, lat: float, lon: float) -> list[str]:
        """Get active weather alerts for a location."""
        cache_key = f"alerts_{lat}_{lon}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            url = f"{NOAA_API_BASE}/alerts/active"
            resp = self.client.get(url, params={"point": f"{lat},{lon}"})
            resp.raise_for_status()
            data = resp.json()

            alerts = []
            for feature in data.get("features", [])[:5]:
                props = feature.get("properties", {})
                event = props.get("event", "")
                headline = props.get("headline", "")
                severity = props.get("severity", "")
                alerts.append(f"[{severity}] {event}: {headline}")

            self._set_cached(cache_key, alerts)
            return alerts

        except Exception as e:
            log.debug("noaa_alerts_failed", error=str(e))
            return []

    def _get_hurricane_outlook(self) -> Optional[str]:
        """Get current NHC tropical outlook."""
        cache_key = "hurricane_outlook"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            # NHC Atlantic outlook
            url = f"{NOAA_API_BASE}/alerts/active"
            resp = self.client.get(url, params={"event": "Hurricane,Tropical Storm"})
            resp.raise_for_status()
            data = resp.json()

            features = data.get("features", [])
            if not features:
                result = "No active tropical systems in the Atlantic/Pacific basins."
            else:
                systems = []
                for f in features[:5]:
                    props = f.get("properties", {})
                    systems.append(f"{props.get('event', '')}: {props.get('headline', '')}")
                result = " | ".join(systems)

            self._set_cached(cache_key, result)
            return result

        except Exception as e:
            log.debug("hurricane_outlook_failed", error=str(e))
            return None

    def _get_cached(self, key: str):
        """Get from cache if not expired."""
        if key in self._cache:
            ts, data = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return data
        return None

    def _set_cached(self, key: str, data):
        """Set cache entry."""
        self._cache[key] = (time.time(), data)

    def close(self):
        self.client.close()

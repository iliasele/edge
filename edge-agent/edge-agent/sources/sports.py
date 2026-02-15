"""
Sports Data Source: Scrapes injury reports, lineups, and breaking news.

The edge: injury reports drop throughout the day. A star player ruled OUT
20 minutes ago might not be priced into the Polymarket line yet.

Free sources used:
- ESPN API (undocumented but public)
- Rotowire-style injury feeds via ESPN
- Official league transaction wires

Covers: NFL, NBA, MLB, NHL, MMA/UFC, Soccer, College sports
"""

import re
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional

import httpx
import structlog

log = structlog.get_logger()

ESPN_API_BASE = "https://site.api.espn.com/apis/site/v2/sports"

# Maps sport keywords to ESPN league codes
SPORT_LEAGUES = {
    # NFL
    "nfl": ("football", "nfl"),
    "super bowl": ("football", "nfl"),
    "touchdown": ("football", "nfl"),
    "quarterback": ("football", "nfl"),
    # NBA
    "nba": ("basketball", "nba"),
    "basketball": ("basketball", "nba"),
    # MLB
    "mlb": ("baseball", "mlb"),
    "baseball": ("baseball", "mlb"),
    "world series": ("baseball", "mlb"),
    "home run": ("baseball", "mlb"),
    # NHL
    "nhl": ("hockey", "nhl"),
    "hockey": ("hockey", "nhl"),
    "stanley cup": ("hockey", "nhl"),
    # MMA/UFC
    "ufc": ("mma", "ufc"),
    "mma": ("mma", "ufc"),
    # Soccer
    "premier league": ("soccer", "eng.1"),
    "epl": ("soccer", "eng.1"),
    "champions league": ("soccer", "uefa.champions"),
    "la liga": ("soccer", "esp.1"),
    "mls": ("soccer", "usa.1"),
    "world cup": ("soccer", "fifa.world"),
    # College
    "ncaa basketball": ("basketball", "mens-college-basketball"),
    "march madness": ("basketball", "mens-college-basketball"),
    "college football": ("football", "college-football"),
    "cfb": ("football", "college-football"),
}

# Common sports terms in market questions
SPORTS_KEYWORDS = [
    "win", "championship", "playoff", "finals", "mvp", "series",
    "game", "match", "bout", "fight", "round",
    "score", "points", "goals", "rushing", "passing",
    "season", "draft", "trade", "free agent",
    "nfl", "nba", "mlb", "nhl", "ufc", "mma",
    "super bowl", "world series", "stanley cup",
    "premier league", "champions league", "la liga",
    "march madness", "ncaa", "college football",
] + list(SPORT_LEAGUES.keys())


@dataclass
class InjuryReport:
    """A single injury entry."""
    player_name: str
    team: str
    status: str  # OUT, DOUBTFUL, QUESTIONABLE, PROBABLE, DAY-TO-DAY
    injury: str  # e.g., "knee", "concussion"
    updated: str


@dataclass
class GameInfo:
    """Upcoming or recent game info."""
    home_team: str
    away_team: str
    start_time: str
    status: str  # pre, in, post
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    odds: Optional[dict] = None  # spread, over/under if available
    venue: Optional[str] = None


@dataclass
class SportsData:
    """Enrichment data for a sports market."""
    market_id: str
    sport: str
    league: str
    teams_detected: list[str] = field(default_factory=list)
    injuries: list[InjuryReport] = field(default_factory=list)
    upcoming_games: list[GameInfo] = field(default_factory=list)
    recent_results: list[GameInfo] = field(default_factory=list)
    standings_context: str = ""
    news_headlines: list[str] = field(default_factory=list)
    data_timestamp: str = ""
    source: str = "ESPN"


class SportsSource:
    """Fetches sports data for market enrichment."""

    def __init__(self):
        self.client = httpx.Client(timeout=15.0)
        self._cache: dict[str, tuple[float, any]] = {}
        self._cache_ttl = 300  # 5 min cache (sports data changes fast)

    def is_sports_market(self, question: str) -> bool:
        """Check if a market is sports-related."""
        q_lower = question.lower()
        return any(kw in q_lower for kw in SPORTS_KEYWORDS)

    def enrich(self, market_id: str, question: str, description: str = "") -> Optional[SportsData]:
        """Fetch relevant sports data for a market."""
        try:
            q_lower = question.lower()

            # Detect sport/league
            sport, league = self._detect_league(q_lower)
            if not sport:
                return None

            data = SportsData(
                market_id=market_id,
                sport=sport,
                league=league,
                data_timestamp=datetime.now(timezone.utc).isoformat(),
            )

            # Detect team names from question
            teams = self._detect_teams(q_lower, sport, league)
            data.teams_detected = teams

            # Fetch scoreboard (upcoming + recent games)
            games = self._get_scoreboard(sport, league)
            for game in games:
                if game.status == "pre":
                    data.upcoming_games.append(game)
                elif game.status == "post":
                    data.recent_results.append(game)

            # Filter games to relevant teams if detected
            if teams:
                team_set = set(t.lower() for t in teams)
                data.upcoming_games = [
                    g for g in data.upcoming_games
                    if any(t in g.home_team.lower() or t in g.away_team.lower() for t in team_set)
                ] or data.upcoming_games[:3]  # Fallback to top 3

                data.recent_results = [
                    g for g in data.recent_results
                    if any(t in g.home_team.lower() or t in g.away_team.lower() for t in team_set)
                ] or data.recent_results[:3]

            # Fetch injuries for relevant teams
            if teams:
                for team_name in teams[:4]:  # Limit to avoid too many requests
                    injuries = self._get_team_injuries(sport, league, team_name)
                    data.injuries.extend(injuries)

            # Fetch latest news/headlines
            headlines = self._get_news(sport, league)
            data.news_headlines = headlines[:5]

            # Only return if we got useful data
            if data.injuries or data.upcoming_games or data.recent_results or data.news_headlines:
                log.debug(
                    "sports_enriched",
                    market_id=market_id,
                    sport=sport,
                    teams=teams,
                    injuries=len(data.injuries),
                )
                return data

            return None

        except Exception as e:
            log.warning("sports_enrich_failed", market_id=market_id, error=str(e))
            return None

    def _detect_league(self, question: str) -> tuple[Optional[str], Optional[str]]:
        """Detect which sport/league a question is about."""
        for keyword, (sport, league) in SPORT_LEAGUES.items():
            if keyword in question:
                return sport, league
        return None, None

    def _detect_teams(self, question: str, sport: str, league: str) -> list[str]:
        """Try to extract team names from the question."""
        # Get all teams for this league
        teams_list = self._get_teams_list(sport, league)
        found = []
        for team_name in teams_list:
            if team_name.lower() in question:
                found.append(team_name)
        return found[:4]  # Max 4 teams

    def _get_teams_list(self, sport: str, league: str) -> list[str]:
        """Get list of team names for a league."""
        cache_key = f"teams_{sport}_{league}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            url = f"{ESPN_API_BASE}/{sport}/{league}/teams"
            resp = self.client.get(url, params={"limit": 100})
            resp.raise_for_status()
            data = resp.json()

            teams = []
            for team_data in data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
                team = team_data.get("team", {})
                teams.append(team.get("displayName", ""))
                teams.append(team.get("shortDisplayName", ""))
                teams.append(team.get("nickname", ""))
                teams.append(team.get("abbreviation", ""))

            teams = [t for t in teams if t]  # Remove empty
            self._set_cached(cache_key, teams, ttl=3600)  # Cache 1 hour
            return teams

        except Exception:
            return []

    def _get_scoreboard(self, sport: str, league: str) -> list[GameInfo]:
        """Get current scoreboard (live, upcoming, recent games)."""
        cache_key = f"scoreboard_{sport}_{league}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            url = f"{ESPN_API_BASE}/{sport}/{league}/scoreboard"
            resp = self.client.get(url)
            resp.raise_for_status()
            data = resp.json()

            games = []
            for event in data.get("events", []):
                competition = event.get("competitions", [{}])[0]
                competitors = competition.get("competitors", [])
                
                if len(competitors) < 2:
                    continue

                home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
                away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

                status_type = event.get("status", {}).get("type", {}).get("state", "pre")

                game = GameInfo(
                    home_team=home.get("team", {}).get("displayName", "Unknown"),
                    away_team=away.get("team", {}).get("displayName", "Unknown"),
                    start_time=event.get("date", ""),
                    status=status_type,
                    home_score=int(home.get("score", 0)) if status_type != "pre" else None,
                    away_score=int(away.get("score", 0)) if status_type != "pre" else None,
                    venue=competition.get("venue", {}).get("fullName"),
                )

                # Try to get odds
                odds_data = competition.get("odds", [{}])
                if odds_data:
                    odd = odds_data[0] if isinstance(odds_data, list) else odds_data
                    game.odds = {
                        "spread": odd.get("details", ""),
                        "over_under": odd.get("overUnder"),
                    }

                games.append(game)

            self._set_cached(cache_key, games)
            return games

        except Exception as e:
            log.debug("espn_scoreboard_failed", error=str(e))
            return []

    def _get_team_injuries(self, sport: str, league: str, team_name: str) -> list[InjuryReport]:
        """Get injury report for a specific team."""
        cache_key = f"injuries_{sport}_{league}_{team_name}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            # First find the team ID
            team_id = self._find_team_id(sport, league, team_name)
            if not team_id:
                return []

            url = f"{ESPN_API_BASE}/{sport}/{league}/teams/{team_id}/injuries"
            resp = self.client.get(url)
            resp.raise_for_status()
            data = resp.json()

            injuries = []
            for item in data.get("injuries", []):
                for injury_item in item.get("injuries", []):
                    athlete = injury_item.get("athlete", {})
                    injuries.append(
                        InjuryReport(
                            player_name=athlete.get("displayName", "Unknown"),
                            team=team_name,
                            status=injury_item.get("status", "Unknown"),
                            injury=injury_item.get("type", {}).get("description", "Undisclosed"),
                            updated=injury_item.get("date", ""),
                        )
                    )

            self._set_cached(cache_key, injuries)
            return injuries

        except Exception as e:
            log.debug("injuries_fetch_failed", team=team_name, error=str(e))
            return []

    def _find_team_id(self, sport: str, league: str, team_name: str) -> Optional[str]:
        """Find ESPN team ID from name."""
        cache_key = f"team_id_{sport}_{league}_{team_name}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            url = f"{ESPN_API_BASE}/{sport}/{league}/teams"
            resp = self.client.get(url, params={"limit": 100})
            resp.raise_for_status()
            data = resp.json()

            name_lower = team_name.lower()
            for team_data in data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
                team = team_data.get("team", {})
                names = [
                    team.get("displayName", "").lower(),
                    team.get("shortDisplayName", "").lower(),
                    team.get("nickname", "").lower(),
                    team.get("abbreviation", "").lower(),
                ]
                if name_lower in names or any(name_lower in n for n in names):
                    team_id = team.get("id")
                    self._set_cached(cache_key, team_id, ttl=3600)
                    return team_id

            return None

        except Exception:
            return None

    def _get_news(self, sport: str, league: str) -> list[str]:
        """Get latest news headlines for a sport."""
        cache_key = f"news_{sport}_{league}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            url = f"{ESPN_API_BASE}/{sport}/{league}/news"
            resp = self.client.get(url, params={"limit": 10})
            resp.raise_for_status()
            data = resp.json()

            headlines = []
            for article in data.get("articles", []):
                headline = article.get("headline", "")
                if headline:
                    headlines.append(headline)

            self._set_cached(cache_key, headlines)
            return headlines

        except Exception:
            return []

    def _get_cached(self, key: str):
        if key in self._cache:
            ts, data, ttl = self._cache[key]
            if time.time() - ts < ttl:
                return data
        return None

    def _set_cached(self, key: str, data, ttl: int = None):
        self._cache[key] = (time.time(), data, ttl or self._cache_ttl)

    def close(self):
        self.client.close()

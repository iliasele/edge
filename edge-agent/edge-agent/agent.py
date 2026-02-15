"""
Polymarket Autonomous Trading Agent v2
=======================================

Now with domain-specific data sources:
  - Weather: NOAA forecasts (free)
  - Sports: ESPN injury reports + scores (free)
  - Crypto: CoinGecko + on-chain metrics + Fear/Greed (free)

The loop every 10 minutes:
  1. Scans 500-1000 active markets
  2. Classifies markets by domain (weather/sports/crypto/general)
  3. Fetches real-time data from free APIs
  4. Sends markets + enrichment data to Claude for fair value estimation
  5. Finds mispricings > 8% edge
  6. Sizes positions with Kelly Criterion (max 6% bankroll)
  7. Executes trades via CLOB API
  8. Deducts API costs from bankroll
  9. If balance hits $0 → agent dies

Usage:
  python agent.py              # Live trading
  python agent.py --dry-run    # Paper trading (no real money)
  python agent.py --once       # Run one cycle then exit

VPS deployment:
  # On a $4.5/month Hetzner CX22 or similar:
  nohup python agent.py > agent.log 2>&1 &
  # Or use systemd (see deploy/agent.service)
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

import structlog

from models import Config, FairValueEstimate, Side, Market
from scanner import MarketScanner
from kalshi_scanner import KalshiScanner
from selector import HybridSelector
from enricher import DataEnricher
from analyst import Analyst
from pre_analyzer import PreAnalyzer
from sizer import PositionSizer
from executor import TradeExecutor
from kalshi_executor import KalshiExecutor
from accountant import Accountant
from settlement import SettlementMonitor
from position_monitor import PositionMonitor
from logger import TradeLogger

# Optional: Monad token interaction
try:
    from monad.token_trader import MonadTokenTrader
    MONAD_AVAILABLE = True
except ImportError:
    MONAD_AVAILABLE = False

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger()


class TradingAgent:
    """The autonomous trading agent with data enrichment."""

    # Analysis cache file — survives restarts
    ANALYSIS_CACHE_FILE = "analysis_cache.json"

    def __init__(self, config: Config, initial_balance: float = 100.0):
        self.config = config
        self.scanner = MarketScanner(config)
        self.kalshi_scanner = KalshiScanner(config)    # Kalshi markets
        self.selector = HybridSelector(config)    # Claude + learning
        self.enricher = DataEnricher()         # NEW: domain-specific data
        self.analyst = Analyst(config)
        self.pre_analyzer = PreAnalyzer()
        self.sizer = PositionSizer(config)
        self.executor = TradeExecutor(config)
        self.kalshi_executor = KalshiExecutor(config)  # Kalshi trading
        self.logger = TradeLogger()                    # NEW: CSV logging
        self.cycle_count = 0
        self._running = True

        # Polymarket mode: BUY_ONLY until USDC is deposited in exchange contract
        # Set to "disabled" to skip Polymarket entirely, "buy_only", or "full"
        self.polymarket_mode = os.getenv("POLYMARKET_MODE", "buy_only").lower()
        # Minimum order size for Polymarket ($1 minimum enforced by exchange)
        self.poly_min_order_usd = 1.0

        # Get real wallet balance if in live mode
        if not config.dry_run:
            real_balance = self.executor.get_balance()
            if real_balance > 0:
                log.info("using_real_wallet_balance", balance=f"${real_balance:.2f}")
                initial_balance = real_balance
            else:
                log.warning("could_not_read_wallet_using_provided_balance", balance=initial_balance)

        self.accountant = Accountant(initial_balance)
        self.settlement_monitor = SettlementMonitor(config)  # NEW: tracks outcomes
        self.position_monitor = PositionMonitor(config)    # NEW: exit management
        self.cycle_count = 0
        self._running = True

        # Monad token interaction (optional — for Moltiverse hackathon)
        self.token_trader = None
        if MONAD_AVAILABLE:
            try:
                self.token_trader = MonadTokenTrader()
                if self.token_trader.enabled:
                    log.info("monad_token_active", token=self.token_trader.config.get("token_symbol"))
            except Exception as e:
                log.debug("monad_token_init_skipped", error=str(e))

        # Track recently traded/analyzed markets to avoid re-analyzing
        # {market_id: {"cycle": N, "estimate": {...}, "timestamp": "..."}}
        self._analysis_cache: dict[str, dict] = {}
        self._load_analysis_cache()

        # IDs of markets we have open positions on — never re-analyze
        self._open_position_ids: set[str] = set()
        self._refresh_open_positions()

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        log.info("shutdown_signal_received", signal=signum)
        self._running = False

    def _is_on_cooldown(self, market_id: str, market_mid: float = 0.0) -> bool:
        """
        Check if a market should be skipped.
        
        Skip if:
        - We have an open position on it
        - We analyzed it recently AND price hasn't moved >5%
        """
        # Always skip markets with open positions
        if market_id in self._open_position_ids:
            return True

        if market_id not in self._analysis_cache:
            return False

        cached = self._analysis_cache[market_id]
        cycles_since = self.cycle_count - cached.get("cycle", 0)

        # Long cooldown: 50 cycles (~8 hours at 10min interval)
        cooldown_cycles = 50

        if cycles_since >= cooldown_cycles:
            # Cooldown expired
            del self._analysis_cache[market_id]
            return False

        # Check if price has moved significantly (>5%) — if so, re-analyze
        cached_mid = cached.get("mid_price", 0)
        if cached_mid > 0 and market_mid > 0:
            price_move = abs(market_mid - cached_mid)
            if price_move > 0.05:
                log.debug("price_moved_reanalyze", market_id=market_id[:20], move=f"{price_move:.3f}")
                del self._analysis_cache[market_id]
                return False

        return True

    def _cache_analysis(self, market_id: str, mid_price: float, estimate: dict = None):
        """Cache that we analyzed this market."""
        self._analysis_cache[market_id] = {
            "cycle": self.cycle_count,
            "mid_price": mid_price,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "estimate": estimate,
        }
        self._save_analysis_cache()

    def _refresh_open_positions(self):
        """Update the set of market IDs we have open positions on."""
        open_trades = self.accountant.get_open_trades()
        self._open_position_ids = set()
        for t in open_trades:
            self._open_position_ids.add(t.market_id)
            if t.ticker:
                self._open_position_ids.add(t.ticker)

    def _load_analysis_cache(self):
        """Load analysis cache from disk."""
        if os.path.exists(self.ANALYSIS_CACHE_FILE):
            try:
                with open(self.ANALYSIS_CACHE_FILE, "r") as f:
                    self._analysis_cache = json.load(f)
                log.info("analysis_cache_loaded", entries=len(self._analysis_cache))
            except Exception:
                self._analysis_cache = {}

    def _save_analysis_cache(self):
        """Save analysis cache to disk."""
        try:
            with open(self.ANALYSIS_CACHE_FILE, "w") as f:
                json.dump(self._analysis_cache, f)
        except Exception:
            pass

    def run(self, once: bool = False):
        """Main loop. Runs forever unless once=True."""
        log.info(
            "agent_starting",
            dry_run=self.config.dry_run,
            bankroll=f"${self.accountant.get_bankroll():.2f}",
            min_edge=f"{self.config.min_edge_pct:.0%}",
            max_position=f"{self.config.max_bankroll_pct:.0%}",
            scan_interval=f"{self.config.scan_interval}s",
            polymarket_mode=self.polymarket_mode,
            cached_analyses=len(self._analysis_cache),
        )

        while self._running:
            # === SURVIVAL CHECK ===
            if not self.accountant.is_alive():
                log.critical("AGENT_IS_DEAD. Shutting down.")
                print(self.accountant.get_report())
                break

            # === COST CHECK ===
            estimated_cycle_cost = 0.15  # Slightly higher with enrichment
            if not self.accountant.can_afford_cycle(estimated_cycle_cost):
                log.warning(
                    "low_balance_reducing_scope",
                    balance=f"${self.accountant.get_bankroll():.2f}",
                )
                self.config.max_markets = min(self.config.max_markets, 100)

            # === RUN CYCLE ===
            try:
                # Check for settled trades BEFORE running new cycle
                self._check_settlements()
                # Check open positions for exit opportunities
                self._check_positions()
                # Refresh open position IDs so we don't re-analyze them
                self._refresh_open_positions()

                # Polymarket cycle (skip if disabled)
                if self.polymarket_mode != "disabled":
                    self._run_cycle()
                else:
                    self.cycle_count += 1
                    log.info("polymarket_disabled_skipping", cycle=self.cycle_count)
            except Exception as e:
                log.error("cycle_failed", error=str(e), cycle=self.cycle_count)

            # === REPORT ===
            print(self.accountant.get_report())

            # === MONAD TOKEN INTERACTION ===
            if self.token_trader and self.token_trader.enabled:
                try:
                    stats = self.accountant.get_stats()
                    net_pnl = stats["total_pnl"] - stats["api_costs"]
                    self.token_trader.update_on_pnl(net_pnl, stats)
                except Exception as e:
                    log.debug("monad_token_update_failed", error=str(e))

            if once:
                break

            # === KALSHI MINI-CYCLE during Polymarket cooldown ===
            kalshi_wait = self.config.scan_interval // 2  # Run Kalshi halfway through
            log.info(f"sleeping {kalshi_wait}s then running Kalshi scan...")
            for _ in range(kalshi_wait):
                if not self._running:
                    break
                time.sleep(1)

            if self._running and self.accountant.is_alive():
                try:
                    self._run_kalshi_cycle()
                except Exception as e:
                    log.error("kalshi_cycle_failed", error=str(e))

                print(self.accountant.get_report())

            # Sleep remaining time
            remaining_sleep = self.config.scan_interval - kalshi_wait
            log.info(f"sleeping {remaining_sleep}s until next Polymarket cycle...")
            for _ in range(remaining_sleep):
                if not self._running:
                    break
                time.sleep(1)

        # Cleanup
        self.enricher.close()
        self.scanner.close()
        self.kalshi_scanner.close()
        self.kalshi_executor.close()
        self.settlement_monitor.close()
        log.info("agent_stopped")

    def _check_settlements(self):
        """
        Check for settled trades and feed results to learning layer.
        
        This is the KEY feedback loop:
          Settlement detected → Accountant updated → Selector learns
        """
        try:
            settlements = self.settlement_monitor.check_settlements(self.accountant)
            
            if not settlements:
                return
            
            # Feed each settlement to the selector's learning layer
            for s in settlements:
                if s.get("won") is None:
                    continue  # Cancelled orders, skip learning
                
                # Reconstruct a minimal Market object for the selector
                market = Market(
                    condition_id=s.get("ticker", s.get("trade_id", "")),
                    token_id=s.get("ticker", ""),
                    question=s.get("question", ""),
                    best_bid=0, best_ask=0, mid_price=0, spread=0,
                    volume_24h=0,
                    source=s.get("source", "unknown"),
                )
                
                self.selector.record_trade(
                    market=market,
                    won=s["won"],
                    edge=s.get("edge", 0.0),
                    pnl=s["pnl"],
                )
                
                # Log to CSV — update original row + append settlement row
                self.logger.update_trade_row(s.get("question", "")[:40], s)
                self.logger.log_settlement(s)
                
                emoji = "✅" if s["won"] else "❌"
                log.info(
                    f"{emoji} SETTLEMENT",
                    source=s["source"],
                    result=s["market_result"],
                    pnl=f"${s['pnl']:.2f}",
                    question=s["question"][:60],
                )
            
            # Print updated stats
            stats = self.accountant.get_stats()
            log.info(
                "settlement_summary",
                new_settlements=len([s for s in settlements if s.get("won") is not None]),
                total_settled=stats["settled_trades"],
                win_rate=f"{stats['win_rate']:.0%}",
                total_pnl=f"${stats['total_pnl']:.2f}",
            )
            
        except Exception as e:
            log.error("settlement_check_failed", error=str(e))

    def _check_positions(self):
        """
        Evaluate open positions for exit opportunities.
        
        Uses PositionMonitor to check if any positions should be
        closed (take profit, stop loss, edge decay, near expiry).
        """
        if not self.position_monitor.should_check():
            return

        open_trades = self.accountant.get_open_trades()
        if not open_trades:
            return

        log.info("checking_positions", open_count=len(open_trades))

        def price_fetcher(trade):
            """Get current price for an open trade."""
            try:
                if trade.source == "kalshi":
                    ticker = trade.ticker or trade.market_id
                    if ticker.startswith("kalshi_"):
                        ticker = ticker[7:]
                    return self.kalshi_executor.get_current_price(ticker)
                else:
                    # Polymarket: use token_id
                    return self.executor.get_current_price(trade.market_id)
            except Exception:
                return None

        # Evaluate all positions
        exit_signals = self.position_monitor.evaluate_positions(
            open_trades=open_trades,
            price_fetcher=price_fetcher,
        )

        if not exit_signals:
            return

        # Execute exits
        for sig in exit_signals:
            trade = sig["trade"]
            decision = sig["decision"]

            try:
                if trade.source == "kalshi":
                    result = self.kalshi_executor.exit_position(trade, decision.current_price)
                else:
                    result = self.executor.exit_position(trade, decision.current_price)

                if result.get("status") in ("exited", "exit_placed"):
                    exit_pnl = result.get("exit_pnl", 0.0)
                    self.accountant.record_exit(
                        trade_id=trade.trade_id,
                        exit_pnl=exit_pnl,
                        exit_price=decision.current_price,
                    )

                    # Also feed to position monitor for learning
                    self.position_monitor.record_outcome(trade.trade_id, exit_pnl)

                    # Feed to selector learning
                    market = Market(
                        condition_id=trade.market_id,
                        token_id=trade.ticker or "",
                        question=trade.question,
                        best_bid=0, best_ask=0, mid_price=0, spread=0,
                        volume_24h=0,
                        source=trade.source,
                    )
                    self.selector.record_trade(
                        market=market,
                        won=exit_pnl > 0,
                        edge=trade.edge,
                        pnl=exit_pnl,
                    )

                    emoji = "📤" if exit_pnl > 0 else "📉"
                    log.info(
                        f"{emoji} EXIT",
                        reason=decision.reason,
                        pnl=f"${exit_pnl:.2f}",
                        question=trade.question[:60],
                    )
                else:
                    log.warning(
                        "exit_failed",
                        status=result.get("status"),
                        error=result.get("error", ""),
                        question=trade.question[:40],
                    )

            except Exception as e:
                log.error("exit_execution_failed", error=str(e), trade_id=trade.trade_id[:20])

        # Log learning summary periodically
        if self.cycle_count % 10 == 0:
            summary = self.position_monitor.get_learning_summary()
            if summary.get("evaluated", 0) > 0:
                log.info("exit_learning", **summary)

    def _run_cycle(self):
        """Execute one full trading cycle."""
        self.cycle_count += 1
        cycle_start = time.time()
        log.info("cycle_start", cycle=self.cycle_count)

        # --- Step 1: Scan markets ---
        markets = self.scanner.fetch_active_markets()
        if not markets:
            log.warning("no_markets_found")
            return

        log.info("scan_complete", markets_found=len(markets))

        # Refresh real wallet balance each cycle (live mode only)
        if not self.config.dry_run:
            real_balance = self.executor.get_balance()
            if real_balance >= 0:
                self.accountant.balance_usd = real_balance
                log.info("balance_refreshed", balance=f"${real_balance:.2f}")

        # --- Step 2: SMART SELECTION - Claude picks where to look (NEW!) ---
        bankroll = self.accountant.get_bankroll()
        markets_to_analyze = self.selector.select_markets(markets, bankroll)

        # Deduct selector API cost
        selector_cost = self.selector.get_cost()
        if selector_cost > 0:
            self.accountant.deduct_api_cost(selector_cost)

        log.info("smart_selection_complete", selected=len(markets_to_analyze))

        # --- Filter out markets on cooldown or with open positions ---
        before_cooldown = len(markets_to_analyze)
        markets_to_analyze = [
            m for m in markets_to_analyze
            if not self._is_on_cooldown(m.condition_id, m.mid_price)
        ]
        skipped = before_cooldown - len(markets_to_analyze)
        if skipped > 0:
            log.info("markets_skipped_cooldown", skipped=skipped, remaining=len(markets_to_analyze))

        if not markets_to_analyze:
            log.info("all_markets_on_cooldown_skipping_cycle")
            return

        # --- Step 3: ENRICH with domain-specific data (NEW!) ---
        enrichment_start = time.time()
        enriched_markets = self.enricher.enrich_markets(markets_to_analyze)
        enrichment_time = time.time() - enrichment_start

        enrichment_stats = self.enricher.get_stats()
        log.info(
            "enrichment_complete",
            elapsed=f"{enrichment_time:.1f}s",
            weather=enrichment_stats.get("weather", 0),
            sports=enrichment_stats.get("sports", 0),
            crypto=enrichment_stats.get("crypto", 0),
            general=enrichment_stats.get("general", 0),
        )

        # --- Step 4a: Pre-analyze with MATH (crypto targets, etc.) ---
        enriched_first = sorted(
            enriched_markets,
            key=lambda e: 0 if e.domain == "general" else 1,
            reverse=True,
        )
        pre_analyses = self.pre_analyzer.analyze(markets_to_analyze, enriched_first)
        
        # Convert pre-analyses to FairValueEstimates
        pre_estimates = []
        markets_for_claude = []
        enriched_for_claude = []
        
        enriched_map = {e.market.condition_id: e for e in enriched_first}
        
        for m in markets_to_analyze:
            if m.condition_id in pre_analyses:
                pa = pre_analyses[m.condition_id]
                pre_estimates.append(FairValueEstimate(
                    market_id=pa.market_id,
                    question=m.question,
                    fair_probability=pa.fair_probability,
                    confidence=pa.confidence,
                    reasoning=pa.reasoning,
                    key_factors=[pa.method],
                ))
            else:
                markets_for_claude.append(m)
                enriched_for_claude.append(enriched_map.get(m.condition_id))

        # --- Step 4b: Analyze remaining with Claude ---
        claude_estimates = []
        if markets_for_claude:
            claude_estimates = self.analyst.estimate_batch(
                markets_for_claude,
                enriched=enriched_for_claude,
            )

        estimates = pre_estimates + claude_estimates

        # Deduct API cost
        api_cost = self.analyst.get_session_cost()
        self.accountant.deduct_api_cost(api_cost)
        self.analyst.total_cost = 0

        if not self.accountant.is_alive():
            log.critical("Agent died from API costs!")
            return

        # --- Step 5: Find trades (Kelly sizing) ---
        signals = self.sizer.find_trades(markets_to_analyze, estimates, bankroll)

        # Feed learning: record which markets produced signals and which didn't
        signal_ids = {s.market.condition_id for s in signals}
        for market in markets_to_analyze:
            had_signal = market.condition_id in signal_ids
            edge = 0.0
            if had_signal:
                sig = next((s for s in signals if s.market.condition_id == market.condition_id), None)
                if sig:
                    edge = sig.edge
            self.selector.record_analysis(market, had_signal=had_signal, edge=edge)
            # Cache analysis so we don't re-analyze next cycle
            self._cache_analysis(market.condition_id, market.mid_price)

        if not signals:
            log.info("no_trades_this_cycle")
            # Print what the agent has learned so far
            print(self.selector.get_learned_insights())
            return

        # --- Step 5b: Filter signals for Polymarket constraints ---
        filtered_signals = []
        for sig in signals:
            # Skip SELL signals only if no_token_id is missing
            if sig.side == Side.SELL and not sig.market.no_token_id:
                log.debug("poly_no_token_missing_skip", q=sig.market.question[:60])
                continue
            # Skip signals below $1 minimum
            if sig.position_size_usd < self.poly_min_order_usd:
                log.debug("poly_below_min_size", size=f"${sig.position_size_usd:.2f}", q=sig.market.question[:40])
                continue
            filtered_signals.append(sig)

        if not filtered_signals:
            log.info("no_viable_poly_trades_after_filter", original=len(signals))
            return

        signals = filtered_signals

        # --- Step 6: Execute top signals ---
        max_new_trades = min(self.config.trades_per_cycle, self.config.max_open_positions)
        signals_to_execute = signals[:max_new_trades]

        log.info(
            "executing_trades",
            count=len(signals_to_execute),
            total_signals=len(signals),
            max_per_cycle=self.config.trades_per_cycle,
            best_edge=f"{signals_to_execute[0].edge:.1%}" if signals_to_execute else "N/A",
        )

        results = self.executor.execute_signals(signals_to_execute)

        # Build enrichment lookup for logging
        enriched_map = {e.market.condition_id: e for e in enriched_markets}

        for signal, result in zip(signals_to_execute, results):
            self.accountant.record_trade(result)
            # Cache market so we don't re-analyze it
            self._cache_analysis(signal.market.condition_id, signal.market.mid_price)

            # Feed learning data to selector
            self.selector.record_analysis(signal.market, had_signal=True, edge=signal.edge)
            # Log trade to CSV
            domain = enriched_map.get(signal.market.condition_id)
            self.logger.log_trade(
                cycle=self.cycle_count,
                signal=signal,
                execution_result=result,
                bankroll=bankroll,
                domain=domain.domain if domain else "general",
            )
            log.info("trade_result", **result)

        # Log ALL analyzed markets to CSV (not just traded ones)
        self.logger.log_batch(
            cycle=self.cycle_count,
            markets=markets_to_analyze,
            estimates=estimates,
            enriched_list=enriched_markets,
            signals=signals_to_execute,
        )

        # --- Cycle complete ---
        elapsed = time.time() - cycle_start
        log.info(
            "cycle_complete",
            cycle=self.cycle_count,
            elapsed=f"{elapsed:.1f}s",
            markets_scanned=len(markets),
            markets_analyzed=len(markets_to_analyze),
            markets_enriched=len([e for e in enriched_markets if e.domain != "general"]),
            signals_found=len(signals),
            trades_executed=len(results),
        )

        # Print what the agent has learned
        if self.cycle_count % 3 == 0:  # Every 3 cycles
            print(self.selector.get_learned_insights())

    def _run_kalshi_cycle(self):
        """
        Run a mini analysis cycle on Kalshi markets.
        
        This runs BETWEEN Polymarket cycles to keep the agent productive.
        Kalshi markets are analyzed with the same Claude pipeline but
        execution is Kalshi-specific (or dry-run logged).
        """
        cycle_start = time.time()
        log.info("kalshi_cycle_start")

        # --- Fetch Kalshi markets ---
        kalshi_markets = self.kalshi_scanner.fetch_active_markets(max_markets=300)
        if not kalshi_markets:
            log.info("no_kalshi_markets_found")
            return

        log.info("kalshi_scan_complete", markets_found=len(kalshi_markets))

        # --- Select best markets using same adaptive selector ---
        bankroll = self.accountant.get_bankroll()
        # Smaller budget for Kalshi (it's the secondary source)
        kalshi_budget = min(50, len(kalshi_markets))
        selected = self.selector.select_markets(kalshi_markets, bankroll)[:kalshi_budget]

        # Filter cooldowns + open positions
        selected = [m for m in selected if not self._is_on_cooldown(m.condition_id, m.mid_price)]
        if not selected:
            log.info("kalshi_all_on_cooldown")
            return

        log.info("kalshi_selected", count=len(selected))

        # --- Enrich with domain data ---
        enriched = self.enricher.enrich_markets(selected)

        enriched_first = sorted(
            enriched,
            key=lambda e: 0 if e.domain == "general" else 1,
            reverse=True,
        )

        # --- Pre-analyze with math ---
        pre_analyses = self.pre_analyzer.analyze(selected, enriched_first)
        pre_estimates = []
        kalshi_for_claude = []
        kalshi_enriched_for_claude = []
        
        enriched_map = {e.market.condition_id: e for e in enriched_first}
        for m in selected:
            if m.condition_id in pre_analyses:
                pa = pre_analyses[m.condition_id]
                pre_estimates.append(FairValueEstimate(
                    market_id=pa.market_id,
                    question=m.question,
                    fair_probability=pa.fair_probability,
                    confidence=pa.confidence,
                    reasoning=pa.reasoning,
                    key_factors=[pa.method],
                ))
            else:
                kalshi_for_claude.append(m)
                kalshi_enriched_for_claude.append(enriched_map.get(m.condition_id))

        # --- Analyze remaining with Claude ---
        claude_estimates = []
        if kalshi_for_claude:
            claude_estimates = self.analyst.estimate_batch(kalshi_for_claude, enriched=kalshi_enriched_for_claude)
        estimates = pre_estimates + claude_estimates

        api_cost = self.analyst.get_session_cost()
        self.accountant.deduct_api_cost(api_cost)
        self.analyst.total_cost = 0

        if not self.accountant.is_alive():
            log.critical("Agent died from Kalshi API costs!")
            return

        # --- Find trades ---
        signals = self.sizer.find_trades(selected, estimates, bankroll)

        # Feed learning + cache analyses
        signal_ids = {s.market.condition_id for s in signals}
        for market in selected:
            had_signal = market.condition_id in signal_ids
            edge = 0.0
            if had_signal:
                sig = next((s for s in signals if s.market.condition_id == market.condition_id), None)
                if sig:
                    edge = sig.edge
            self.selector.record_analysis(market, had_signal=had_signal, edge=edge)
            # Cache so we don't re-analyze next cycle
            self._cache_analysis(market.condition_id, market.mid_price)

        if not signals:
            log.info("kalshi_no_trades_this_cycle")
            return

        # --- Execute on Kalshi ---
        max_trades = min(self.config.trades_per_cycle, len(signals))
        signals_to_execute = signals[:max_trades]

        log.info(
            "kalshi_executing_trades",
            count=len(signals_to_execute),
            total_signals=len(signals),
            best_edge=f"{signals_to_execute[0].edge:.1%}",
            live=self.kalshi_executor.is_configured(),
        )

        # Use Kalshi executor (real if configured, dry-run if not)
        results = self.kalshi_executor.execute_signals(signals_to_execute)

        enriched_map = {e.market.condition_id: e for e in enriched}

        for signal, result in zip(signals_to_execute, results):
            # Extract ticker from raw_result for settlement matching
            raw = result.get("raw_result", {})
            if isinstance(raw, dict):
                order = raw.get("order", {})
                if isinstance(order, dict) and order.get("ticker"):
                    result["ticker"] = order["ticker"]

            self.accountant.record_trade(result)
            self._cache_analysis(signal.market.condition_id, signal.market.mid_price)

            domain = enriched_map.get(signal.market.condition_id)
            self.logger.log_trade(
                cycle=self.cycle_count,
                signal=signal,
                execution_result=result,
                bankroll=bankroll,
                domain=domain.domain if domain else "general",
            )
            log.info("kalshi_trade_result", **result)

        # Log all analyzed
        self.logger.log_batch(
            cycle=self.cycle_count,
            markets=selected,
            estimates=estimates,
            enriched_list=enriched,
            signals=signals_to_execute,
        )

        elapsed = time.time() - cycle_start
        log.info(
            "kalshi_cycle_complete",
            elapsed=f"{elapsed:.1f}s",
            markets_scanned=len(kalshi_markets),
            markets_analyzed=len(selected),
            signals_found=len(signals),
            trades_logged=len(signals_to_execute),
        )


def main():
    parser = argparse.ArgumentParser(description="Polymarket Autonomous Trading Agent v2")
    parser.add_argument("--dry-run", action="store_true", help="Paper trading mode")
    parser.add_argument("--once", action="store_true", help="Run one cycle then exit")
    parser.add_argument("--balance", type=float, default=100.0, help="Initial bankroll in USD")
    args = parser.parse_args()

    config = Config()
    if args.dry_run:
        config.dry_run = True

    poly_mode = os.getenv("POLYMARKET_MODE", "buy_only").lower()

    print(f"""
    ╔══════════════════════════════════════════════════════╗
    ║   PREDICTION MARKET TRADING AGENT v4                 ║
    ║                                                      ║
    ║   Mode:     {'DRY RUN (paper)' if config.dry_run else 'LIVE TRADING ⚠️'}                ║
    ║   Balance:  ${args.balance:.2f}                               ║
    ║   Min Edge: {config.min_edge_pct:.0%}                                  ║
    ║   Max Size: {config.max_bankroll_pct:.0%} of bankroll                      ║
    ║   Interval: {config.scan_interval}s                                ║
    ║                                                      ║
    ║   Exchanges:                                         ║
    ║     📈  Polymarket → {poly_mode:<35}║
    ║     📊  Kalshi     → active (primary)                ║
    ║                                                      ║
    ║   Smart features:                                    ║
    ║     🧠  Analysis cache (skip re-analyzed markets)    ║
    ║     📊  Settlement tracking + P&L logging            ║
    ║     📈  Learning from outcomes                       ║
    ║     🪙  Monad EDGE token interaction                 ║
    ║                                                      ║
    ║   If balance → $0, the agent dies.                   ║
    ╚══════════════════════════════════════════════════════╝
    """)

    if not config.dry_run:
        print("⚠️  LIVE MODE: Real money will be used. Ctrl+C to abort.")
        print("    Starting in 5 seconds...")
        time.sleep(5)

    agent = TradingAgent(config, initial_balance=args.balance)
    agent.run(once=args.once)


if __name__ == "__main__":
    main()

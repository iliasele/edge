"""
Settlement Monitor: Closes the feedback loop.

Checks both Kalshi and Polymarket for resolved positions, computes P&L,
and feeds results back to the Accountant and Selector (learning layer).

Kalshi flow:
  GET /portfolio/settlements → list of settled markets with revenue
  GET /portfolio/positions → check for position changes

Polymarket flow:
  Check market resolution status via CLOB API or Gamma API
  Compare against open positions

Runs every cycle as part of the agent loop.
"""

import os
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import structlog

from models import Config, Market
from accountant import Accountant, TradeRecord

log = structlog.get_logger()


class SettlementMonitor:
    """
    Monitors open positions and detects settlements on both exchanges.
    
    When a market resolves:
    1. Computes P&L
    2. Updates Accountant (trade record)
    3. Returns settlement info so the agent can update the Selector (learning)
    """

    def __init__(self, config: Config):
        self.config = config
        
        # Kalshi auth
        self.kalshi_api_key_id = os.getenv("KALSHI_API_KEY_ID", "")
        self.kalshi_private_key = None
        self.kalshi_base_url = "https://api.elections.kalshi.com"
        self.kalshi_client = httpx.Client(timeout=30.0)
        
        if self.kalshi_api_key_id:
            self._load_kalshi_key()
        
        # Polymarket client (reuse from executor)
        self.poly_client = None
        if not config.dry_run:
            self._init_poly_client()
        
        # Track which settlements we've already processed
        self._processed_settlements: set[str] = set()
        self._load_processed()

    def _load_kalshi_key(self):
        """Load Kalshi RSA private key."""
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "./kalshi-key.pem")
        if not os.path.exists(key_path):
            return
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.backends import default_backend
            with open(key_path, "rb") as f:
                self.kalshi_private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )
            log.info("settlement_monitor_kalshi_key_loaded")
        except Exception as e:
            log.error("settlement_kalshi_key_failed", error=str(e))

    def _init_poly_client(self):
        """Initialize Polymarket CLOB client for position checking."""
        try:
            from py_clob_client.client import ClobClient
            sig_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0"))
            funder = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
            
            init_kwargs = {
                "host": "https://clob.polymarket.com",
                "key": self.config.private_key,
                "chain_id": 137,
            }
            if sig_type > 0 and funder:
                init_kwargs["signature_type"] = sig_type
                init_kwargs["funder"] = funder
            
            self.poly_client = ClobClient(**init_kwargs)
            api_creds = self.poly_client.create_or_derive_api_creds()
            self.poly_client.set_api_creds(api_creds)
        except Exception as e:
            log.warning("settlement_poly_client_failed", error=str(e))

    def _kalshi_sign_request(self, method: str, path: str) -> dict:
        """Create Kalshi auth headers."""
        import base64
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        
        timestamp = str(int(datetime.now().timestamp() * 1000))
        path_clean = path.split("?")[0]
        message = f"{timestamp}{method}{path_clean}".encode("utf-8")
        
        signature = self.kalshi_private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        
        return {
            "KALSHI-ACCESS-KEY": self.kalshi_api_key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }

    def _kalshi_get(self, path: str) -> dict:
        """Authenticated GET to Kalshi."""
        headers = self._kalshi_sign_request("GET", path)
        url = self.kalshi_base_url + path
        resp = self.kalshi_client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()

    def check_settlements(self, accountant: Accountant) -> list[dict]:
        """
        Main entry point. Check both exchanges for settled trades.
        
        Returns list of settlement dicts:
            {trade_id, won, pnl, market_result, source, question}
        """
        settlements = []
        
        # Check Kalshi
        try:
            kalshi_settlements = self._check_kalshi_settlements(accountant)
            settlements.extend(kalshi_settlements)
        except Exception as e:
            log.error("kalshi_settlement_check_failed", error=str(e))
        
        # Check Polymarket
        try:
            poly_settlements = self._check_poly_settlements(accountant)
            settlements.extend(poly_settlements)
        except Exception as e:
            log.error("poly_settlement_check_failed", error=str(e))
        
        # Also check for cancelled/expired resting orders
        try:
            cancelled = self._check_cancelled_orders(accountant)
            settlements.extend(cancelled)
        except Exception as e:
            log.error("cancelled_order_check_failed", error=str(e))
        
        if settlements:
            log.info("settlements_found", count=len(settlements))
            self._save_processed()
        
        return settlements

    def _check_kalshi_settlements(self, accountant: Accountant) -> list[dict]:
        """
        Check Kalshi portfolio/settlements endpoint for resolved trades.
        
        The settlements endpoint returns:
        {
            "settlements": [{
                "ticker": "...",
                "market_result": "yes" or "no",
                "yes_count": N, "no_count": N,
                "yes_total_cost": N, "no_total_cost": N,
                "revenue": N (cents),
                "settled_time": "...",
                "fee_cost": "...",
                "value": N (cents)
            }]
        }
        """
        if not (self.kalshi_api_key_id and self.kalshi_private_key):
            return []
        
        results = []
        
        # Get all settlements from Kalshi
        data = self._kalshi_get("/trade-api/v2/portfolio/settlements")
        settlements = data.get("settlements", [])
        
        # Get our open Kalshi trades
        open_kalshi = [
            t for t in accountant.get_open_trades()
            if t.source == "kalshi"
        ]
        
        if not open_kalshi:
            return []
        
        # Build lookup by ticker
        open_by_ticker = {}
        for trade in open_kalshi:
            ticker = trade.ticker or trade.market_id
            if ticker not in open_by_ticker:
                open_by_ticker[ticker] = []
            open_by_ticker[ticker].append(trade)
        
        for settlement in settlements:
            ticker = settlement.get("ticker", "")
            if ticker not in open_by_ticker:
                continue
            
            # Skip if already processed
            settle_key = f"kalshi_{ticker}_{settlement.get('settled_time', '')}"
            if settle_key in self._processed_settlements:
                continue
            
            market_result = settlement.get("market_result", "")  # "yes" or "no"
            revenue_cents = settlement.get("revenue", 0)
            revenue = revenue_cents / 100.0
            fee_cents = float(settlement.get("fee_cost", "0"))
            
            for trade in open_by_ticker[ticker]:
                # Calculate P&L
                # Revenue from Kalshi is net revenue for our position
                # We need to figure out if we won based on our side and the result
                
                if trade.side == "BUY":
                    # We bought YES tokens
                    won = (market_result == "yes")
                    if won:
                        pnl = (1.0 - trade.price) * trade.count  # Each contract pays $1
                    else:
                        pnl = -trade.price * trade.count
                elif trade.side == "SELL":
                    # We sold (bought NO)
                    won = (market_result == "no")
                    if won:
                        pnl = trade.price * trade.count  # We keep what we sold at
                    else:
                        pnl = -(1.0 - trade.price) * trade.count
                else:
                    won = False
                    pnl = 0.0
                
                # Use Kalshi's own revenue figure if available and reasonable
                if revenue_cents != 0 and abs(revenue - pnl) < trade.size_usd:
                    pnl = revenue
                
                # Record in accountant
                accountant.record_settlement(
                    trade_id=trade.trade_id,
                    won=won,
                    pnl=pnl,
                    market_result=market_result,
                )
                
                results.append({
                    "trade_id": trade.trade_id,
                    "won": won,
                    "pnl": pnl,
                    "market_result": market_result,
                    "source": "kalshi",
                    "question": trade.question,
                    "ticker": ticker,
                    "edge": trade.edge,
                })
                
                self._processed_settlements.add(settle_key)
                
                log.info(
                    "kalshi_settlement_detected",
                    ticker=ticker,
                    result=market_result,
                    won=won,
                    pnl=f"${pnl:.2f}",
                    question=trade.question[:60],
                )
        
        return results

    def _check_poly_settlements(self, accountant: Accountant) -> list[dict]:
        """
        Check Polymarket for resolved trades.
        
        Uses the Gamma Markets API to check if a market has resolved,
        since the CLOB client doesn't have a direct settlement endpoint.
        """
        open_poly = [
            t for t in accountant.get_open_trades()
            if t.source == "polymarket"
        ]
        
        if not open_poly:
            return []
        
        results = []
        
        for trade in open_poly:
            settle_key = f"poly_{trade.trade_id}"
            if settle_key in self._processed_settlements:
                continue
            
            try:
                # Check market status via Gamma API
                resolved, result_price = self._check_poly_market_status(trade.market_id)
                
                if not resolved:
                    continue
                
                # Calculate P&L
                if trade.side == "BUY":
                    # Bought YES at trade.price
                    if result_price > 0.5:
                        # Market resolved YES
                        won = True
                        pnl = (1.0 - trade.price) * (trade.size_usd / trade.price)
                    else:
                        won = False
                        pnl = -trade.size_usd
                else:
                    # Sold YES (effectively bought NO) at trade.price
                    if result_price < 0.5:
                        # Market resolved NO
                        won = True
                        pnl = trade.price * (trade.size_usd / (1.0 - trade.price))
                    else:
                        won = False
                        pnl = -trade.size_usd
                
                market_result = "yes" if result_price > 0.5 else "no"
                
                accountant.record_settlement(
                    trade_id=trade.trade_id,
                    won=won,
                    pnl=pnl,
                    market_result=market_result,
                )
                
                results.append({
                    "trade_id": trade.trade_id,
                    "won": won,
                    "pnl": pnl,
                    "market_result": market_result,
                    "source": "polymarket",
                    "question": trade.question,
                    "edge": trade.edge,
                })
                
                self._processed_settlements.add(settle_key)
                
                log.info(
                    "poly_settlement_detected",
                    result=market_result,
                    won=won,
                    pnl=f"${pnl:.2f}",
                    question=trade.question[:60],
                )
            except Exception as e:
                log.debug("poly_market_check_failed", error=str(e), market=trade.market_id[:20])
        
        return results

    def _check_poly_market_status(self, condition_id: str) -> tuple[bool, float]:
        """
        Check if a Polymarket market has resolved.
        
        Returns (is_resolved, result_price).
        result_price is 1.0 for YES, 0.0 for NO.
        """
        # Try Gamma Markets API (public, no auth needed)
        try:
            url = f"https://gamma-api.polymarket.com/markets?condition_id={condition_id}"
            resp = httpx.get(url, timeout=10.0)
            resp.raise_for_status()
            markets = resp.json()
            
            if markets and len(markets) > 0:
                market = markets[0]
                # Check if resolved
                resolved = market.get("resolved", False)
                if resolved:
                    # outcome_prices is a JSON string like "[1.0, 0.0]"
                    outcome_str = market.get("outcome_prices", "")
                    if outcome_str:
                        import json
                        prices = json.loads(outcome_str) if isinstance(outcome_str, str) else outcome_str
                        return True, float(prices[0])  # First outcome = YES
                
                # Also check if end_date has passed and price went to 0 or 1
                end_date = market.get("end_date_iso", "")
                if end_date:
                    from datetime import datetime, timezone
                    try:
                        end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                        if end < datetime.now(timezone.utc):
                            # Market past end date — check if price indicates resolution
                            price_str = market.get("best_bid", "0.5")
                            price = float(price_str) if price_str else 0.5
                            if price >= 0.95 or price <= 0.05:
                                return True, round(price)
                    except (ValueError, TypeError):
                        pass
            
            return False, 0.5
        except Exception:
            return False, 0.5

    def _check_cancelled_orders(self, accountant: Accountant) -> list[dict]:
        """
        Check if any resting orders have been cancelled or expired.
        
        For Kalshi: check GET /portfolio/orders for order status.
        """
        results = []
        
        open_trades = accountant.get_open_trades()
        if not open_trades:
            return []
        
        # Check Kalshi orders
        kalshi_trades = [t for t in open_trades if t.source == "kalshi"]
        if kalshi_trades and self.kalshi_api_key_id and self.kalshi_private_key:
            try:
                data = self._kalshi_get("/trade-api/v2/portfolio/orders")
                orders = data.get("orders", [])
                order_map = {o.get("order_id", ""): o for o in orders}
                
                for trade in kalshi_trades:
                    order = order_map.get(trade.trade_id)
                    if order:
                        status = order.get("status", "")
                        if status in ("canceled", "expired"):
                            accountant.mark_cancelled(trade.trade_id)
                            results.append({
                                "trade_id": trade.trade_id,
                                "won": None,
                                "pnl": 0.0,
                                "market_result": "cancelled",
                                "source": "kalshi",
                                "question": trade.question,
                            })
                            log.info(
                                "order_cancelled",
                                trade_id=trade.trade_id[:20],
                                status=status,
                            )
                    else:
                        # Order not in active orders — might have been filled or settled
                        # Check positions to see if we have a position
                        pass  # Handled by settlement check above
            except Exception as e:
                log.debug("kalshi_order_check_failed", error=str(e))
        
        return results

    def _load_processed(self):
        """Load set of already-processed settlements."""
        path = "processed_settlements.json"
        if os.path.exists(path):
            try:
                import json
                with open(path, "r") as f:
                    self._processed_settlements = set(json.load(f))
            except Exception:
                pass

    def _save_processed(self):
        """Save processed settlements to disk."""
        import json
        try:
            with open("processed_settlements.json", "w") as f:
                json.dump(list(self._processed_settlements), f)
        except Exception:
            pass

    def close(self):
        """Clean up."""
        self.kalshi_client.close()

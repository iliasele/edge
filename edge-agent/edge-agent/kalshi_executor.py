"""
Kalshi Executor: Places orders on Kalshi with RSA-PSS authentication.

Auth flow:
  1. Load RSA private key from file
  2. For each request: sign(timestamp + method + path) with RSA-PSS SHA256
  3. Send headers: KALSHI-ACCESS-KEY, KALSHI-ACCESS-SIGNATURE, KALSHI-ACCESS-TIMESTAMP

Order types:
  - "limit" (default): Resting order, may not fill immediately
  - "market": Instant fill at best available price (pays spread)

Set KALSHI_ORDER_TYPE=market in .env for instant fills.

Setup:
  1. Go to kalshi.com → Settings → API
  2. Create API key → save private key file + key ID
  3. Add to .env:
     KALSHI_API_KEY_ID=your-key-id
     KALSHI_PRIVATE_KEY_PATH=./kalshi-key.pem
     KALSHI_ORDER_TYPE=market   # or "limit" for resting orders
"""

import os
import uuid
import base64
import datetime
import time
import structlog
import httpx
from typing import Optional

from models import Config, TradeSignal, Side

log = structlog.get_logger()

KALSHI_BASE_URL = "https://api.elections.kalshi.com"
KALSHI_DEMO_URL = "https://demo-api.kalshi.co"


class KalshiExecutor:
    """Executes trades on Kalshi with RSA-PSS signed requests."""

    def __init__(self, config: Config):
        self.config = config
        self.api_key_id = os.getenv("KALSHI_API_KEY_ID", "")
        self.private_key = None
        self.client = httpx.Client(timeout=30.0)

        # Order type: "market" for instant fills, "limit" for resting orders
        self.order_type = os.getenv("KALSHI_ORDER_TYPE", "market").lower()

        # Use demo URL if dry run, production if live
        self.base_url = KALSHI_DEMO_URL if config.dry_run else KALSHI_BASE_URL

        if self.api_key_id:
            self._load_private_key()

    def _load_private_key(self):
        """Load RSA private key from file."""
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "./kalshi-key.pem")

        if not os.path.exists(key_path):
            log.warning("kalshi_key_not_found", path=key_path)
            return

        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.backends import default_backend

            with open(key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend(),
                )
            log.info("kalshi_key_loaded", order_type=self.order_type)
        except ImportError:
            log.error("cryptography_not_installed. Run: pip install cryptography")
        except Exception as e:
            log.error("kalshi_key_load_failed", error=str(e))

    def _sign_request(self, method: str, path: str) -> dict:
        """Create authentication headers with RSA-PSS signature."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        timestamp = str(int(datetime.datetime.now().timestamp() * 1000))

        # Sign: timestamp + METHOD + path (without query params)
        path_clean = path.split("?")[0]
        message = f"{timestamp}{method}{path_clean}".encode("utf-8")

        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )

        sig_b64 = base64.b64encode(signature).decode("utf-8")

        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": sig_b64,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }

    def _authenticated_request(self, method: str, path: str, json_data: dict = None) -> dict:
        """Make an authenticated request to Kalshi API."""
        headers = self._sign_request(method, path)
        url = self.base_url + path

        if method == "GET":
            resp = self.client.get(url, headers=headers)
        elif method == "POST":
            resp = self.client.post(url, headers=headers, json=json_data)
        else:
            raise ValueError(f"Unsupported method: {method}")

        resp.raise_for_status()
        return resp.json()

    def is_configured(self) -> bool:
        """Check if Kalshi trading is properly configured."""
        return bool(self.api_key_id and self.private_key)

    def execute_signals(self, signals: list[TradeSignal]) -> list[dict]:
        """Execute trade signals on Kalshi."""
        results = []
        for signal in signals:
            result = self._execute_single(signal)
            results.append(result)
            time.sleep(0.3)
        return results

    def _execute_single(self, signal: TradeSignal) -> dict:
        """Execute a single trade on Kalshi."""
        log.info(
            "kalshi_executing_trade",
            question=signal.market.question[:80],
            side=signal.side.value,
            edge=f"{signal.edge:.1%}",
            size=f"${signal.position_size_usd:.2f}",
            order_type=self.order_type,
        )

        # If not configured or dry run, simulate
        if not self.is_configured() or self.config.dry_run:
            return self._dry_run(signal)

        return self._live_execution(signal)

    def _dry_run(self, signal: TradeSignal) -> dict:
        """Simulate Kalshi trade."""
        price = signal.market.best_ask if signal.side == Side.BUY else signal.market.best_bid
        return {
            "status": "dry_run_kalshi",
            "market": signal.market.question[:80],
            "side": signal.side.value,
            "price": price,
            "size_usd": signal.position_size_usd,
            "edge": signal.edge,
            "kelly": signal.kelly_fraction,
            "source": "kalshi",
        }

    def _live_execution(self, signal: TradeSignal) -> dict:
        """Place a real order on Kalshi."""
        try:
            # Kalshi ticker is stored in token_id
            ticker = signal.market.token_id

            # Determine side and action
            # Our Side.BUY + market underpriced = buy YES
            # Our Side.SELL + market overpriced = buy NO (or sell YES)
            if signal.side == Side.BUY:
                side = "yes"
                action = "buy"
                price = signal.market.best_ask
            else:
                side = "no"
                action = "buy"
                price = 1.0 - signal.market.best_bid  # NO price = 1 - YES price

            # Kalshi prices are in cents (1-99)
            yes_price_cents = max(1, min(99, int(round(price * 100))))

            # Count = number of contracts
            count = max(1, int(signal.position_size_usd / price))

            # Skip dust bets — 1 contract at low price is not worth it
            total_cost = count * price
            if total_cost < 0.50:
                log.debug("kalshi_dust_bet_skipped", cost=f"${total_cost:.2f}", count=count, price=price)
                return {
                    "status": "skipped",
                    "reason": f"dust bet (${total_cost:.2f})",
                    "market": signal.market.question[:80],
                    "source": "kalshi",
                }

            order_data = {
                "ticker": ticker,
                "side": side,
                "action": action,
                "count": count,
                "type": self.order_type,  # "market" or "limit"
                "client_order_id": str(uuid.uuid4()),
            }

            # For limit orders, include price. For market orders, no price needed.
            if self.order_type == "limit":
                if side == "yes":
                    order_data["yes_price"] = yes_price_cents
                else:
                    order_data["no_price"] = 100 - yes_price_cents

            # Remove None values
            order_data = {k: v for k, v in order_data.items() if v is not None}

            path = "/trade-api/v2/portfolio/orders"
            resp = self._authenticated_request("POST", path, json_data=order_data)

            order = resp.get("order", {})
            status = order.get("status", "unknown")
            fill_count = order.get("fill_count", 0)

            log.info(
                "kalshi_order_placed",
                order_id=order.get("order_id", ""),
                status=status,
                fill_count=fill_count,
                order_type=self.order_type,
            )

            return {
                "status": "filled" if fill_count > 0 else "placed",
                "order_id": order.get("order_id", ""),
                "ticker": order.get("ticker", ticker),
                "market": signal.market.question[:80],
                "side": signal.side.value,
                "price": price,
                "size_usd": signal.position_size_usd,
                "count": count,
                "fill_count": fill_count,
                "source": "kalshi",
                "raw_result": resp,
            }

        except Exception as e:
            log.error("kalshi_execution_failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
                "market": signal.market.question[:80],
                "source": "kalshi",
            }

    def get_balance(self) -> float:
        """Get Kalshi account balance."""
        if not self.is_configured():
            return -1.0

        try:
            path = "/trade-api/v2/portfolio/balance"
            data = self._authenticated_request("GET", path)
            # Balance is in cents
            balance_cents = data.get("balance", 0)
            balance = balance_cents / 100.0
            log.info("kalshi_balance", balance=f"${balance:.2f}")
            return balance
        except Exception as e:
            log.error("kalshi_balance_failed", error=str(e))
            return -1.0

    def get_current_price(self, ticker: str) -> float:
        """Get current mid price for a Kalshi market by ticker."""
        try:
            path = f"/trade-api/v2/markets/{ticker}"
            data = self._authenticated_request("GET", path)
            market = data.get("market", data)

            yes_bid = float(market.get("yes_bid_dollars", 0) or 0)
            yes_ask = float(market.get("yes_ask_dollars", 0) or 0)

            if yes_bid > 0 and yes_ask > 0:
                return (yes_bid + yes_ask) / 2
            elif yes_bid > 0:
                return yes_bid
            elif yes_ask > 0:
                return yes_ask

            # Fallback to cents
            yb = int(market.get("yes_bid", 0) or 0)
            ya = int(market.get("yes_ask", 0) or 0)
            if yb > 0 and ya > 0:
                return (yb + ya) / 200.0
            return 0.0
        except Exception as e:
            log.debug("kalshi_price_fetch_failed", ticker=ticker[:30], error=str(e))
            return 0.0

    def exit_position(self, trade, current_price: float) -> dict:
        """
        Exit a Kalshi position by selling contracts we own.

        If we bought YES → sell YES
        If we bought NO → sell NO
        """
        if not self.is_configured():
            return {"status": "error", "error": "Kalshi not configured"}

        try:
            ticker = trade.ticker or trade.market_id
            # Remove "kalshi_" prefix if present
            if ticker.startswith("kalshi_"):
                ticker = ticker[7:]

            # Determine what we own
            if trade.side == "BUY":
                side = "yes"
            else:
                side = "no"

            count = trade.count if trade.count > 0 else max(1, int(trade.size_usd / trade.price))

            # Sell price in cents
            sell_price_cents = max(1, min(99, int(round(current_price * 100))))

            order_data = {
                "ticker": ticker,
                "side": side,
                "action": "sell",
                "count": count,
                "type": self.order_type,
                "client_order_id": str(__import__("uuid").uuid4()),
            }

            if self.order_type == "limit":
                if side == "yes":
                    order_data["yes_price"] = sell_price_cents
                else:
                    order_data["no_price"] = 100 - sell_price_cents

            order_data = {k: v for k, v in order_data.items() if v is not None}

            path = "/trade-api/v2/portfolio/orders"
            resp = self._authenticated_request("POST", path, json_data=order_data)

            order = resp.get("order", {})
            fill_count = order.get("fill_count", 0)

            # Calculate exit P&L
            if trade.side == "BUY":
                exit_pnl = (current_price - trade.price) * count
            else:
                exit_pnl = (trade.price - current_price) * count

            log.info(
                "kalshi_position_exited",
                ticker=ticker[:30],
                entry=trade.price,
                exit=current_price,
                pnl=f"${exit_pnl:.2f}",
                fill_count=fill_count,
            )

            return {
                "status": "exited" if fill_count > 0 else "exit_placed",
                "trade_id": trade.trade_id,
                "exit_price": current_price,
                "exit_pnl": exit_pnl,
                "count": count,
                "source": "kalshi",
                "raw_result": resp,
            }

        except Exception as e:
            log.error("kalshi_exit_failed", error=str(e), trade_id=trade.trade_id[:20])
            return {"status": "error", "error": str(e), "trade_id": trade.trade_id}

    def close(self):
        self.client.close()

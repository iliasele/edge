"""
Executor: places orders on Polymarket via the CLOB API.

Updated to match py-clob-client v0.29+ API:
- ClobClient init uses signature_type + funder (not ApiCreds in constructor)
- Orders use OrderArgs + create_order() (not create_and_sign_order)
- API creds derived via create_or_derive_api_creds()
- post_order takes OrderType (GTC, FOK, GTD)
"""

import os
import time
import structlog
from typing import Optional

from models import Config, TradeSignal, Side

log = structlog.get_logger()


class TradeExecutor:
    """
    Executes trades on Polymarket's CLOB (Central Limit Order Book).
    In production, this uses py-clob-client. For dry runs, it logs trades.
    """

    def __init__(self, config: Config):
        self.config = config
        self.clob_client = None

        if not config.dry_run:
            self._init_clob_client()

    def _init_clob_client(self):
        """Initialize the Polymarket CLOB client (v0.29+ API)."""
        try:
            from py_clob_client.client import ClobClient

            # Determine signature type:
            # 0 = EOA / MetaMask / hardware wallet (default)
            # 1 = Email / Magic wallet
            # 2 = Browser wallet proxy (Coinbase Wallet, etc.)
            sig_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0"))
            funder = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")

            # Initialize client
            init_kwargs = {
                "host": "https://clob.polymarket.com",
                "key": self.config.private_key,
                "chain_id": 137,  # Polygon mainnet
            }

            # Add proxy wallet params if needed
            if sig_type > 0 and funder:
                init_kwargs["signature_type"] = sig_type
                init_kwargs["funder"] = funder

            self.clob_client = ClobClient(**init_kwargs)

            # Derive API credentials (does NOT create new ones if they exist)
            api_creds = self.clob_client.create_or_derive_api_creds()
            self.clob_client.set_api_creds(api_creds)

            log.info(
                "clob_client_initialized",
                signature_type=sig_type,
                funder=funder[:10] + "..." if funder else "none (EOA mode)",
                wallet=self.config.wallet_address[:10] + "..." if self.config.wallet_address else "not set",
            )

        except ImportError:
            log.error("py_clob_client_not_installed. Run: pip install py-clob-client")
            self.config.dry_run = True
        except Exception as e:
            log.error("clob_client_init_failed", error=str(e))
            self.config.dry_run = True

    def execute_signals(self, signals: list[TradeSignal]) -> list[dict]:
        """Execute a list of trade signals. Returns execution results."""
        results = []

        for signal in signals:
            result = self._execute_single(signal)
            results.append(result)
            time.sleep(0.5)  # Small delay between orders

        return results

    def _execute_single(self, signal: TradeSignal) -> dict:
        """Execute a single trade."""
        log.info(
            "executing_trade",
            question=signal.market.question[:80],
            side=signal.side.value,
            edge=f"{signal.edge:.1%}",
            size=f"${signal.position_size_usd:.2f}",
            kelly=f"{signal.kelly_fraction:.2%}",
            ev=f"${signal.expected_value:.4f}",
            dry_run=self.config.dry_run,
        )

        if self.config.dry_run:
            return self._dry_run_execution(signal)

        return self._live_execution(signal)

    def _dry_run_execution(self, signal: TradeSignal) -> dict:
        """Simulate trade execution for dry runs."""
        return {
            "status": "dry_run",
            "market": signal.market.question[:80],
            "side": signal.side.value,
            "price": signal.market.best_ask if signal.side == Side.BUY else signal.market.best_bid,
            "size_usd": signal.position_size_usd,
            "edge": signal.edge,
            "kelly": signal.kelly_fraction,
        }

    def _live_execution(self, signal: TradeSignal) -> dict:
        """Execute a real trade on Polymarket using current py-clob-client API."""
        if not self.clob_client:
            log.error("no_clob_client")
            return {"status": "error", "error": "CLOB client not initialized"}

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL

            # Determine order parameters
            if signal.side == Side.BUY:
                # BUY signal = market is underpriced = buy YES token
                side = BUY
                price = signal.market.best_ask
                token_id = signal.market.token_id
            else:
                # SELL signal = market is overpriced = buy NO token
                # Instead of selling YES (which needs USDC allowance),
                # we BUY the NO token directly (same economic exposure)
                side = BUY
                price = 1.0 - signal.market.best_bid  # NO price = 1 - YES bid
                token_id = signal.market.no_token_id

                if not token_id:
                    log.warning("no_token_id_missing", market=signal.market.question[:60])
                    return {
                        "status": "error",
                        "error": "NO token ID not available for this market",
                        "market": signal.market.question[:80],
                    }

            # Calculate size in shares
            size = signal.position_size_usd / price

            # Build order using OrderArgs (the correct API)
            order_args = OrderArgs(
                token_id=token_id,
                price=round(price, 2),
                size=round(size, 2),
                side=side,
            )

            # Create (sign) the order
            signed_order = self.clob_client.create_order(order_args)

            # Post as GTC (Good-Till-Cancelled) limit order
            resp = self.clob_client.post_order(signed_order, OrderType.GTC)

            log.info("order_placed", resp=resp)

            # Response format: {"success": true, "orderID": "..."}
            success = resp.get("success", False) if isinstance(resp, dict) else False

            return {
                "status": "placed" if success else "failed",
                "order_id": resp.get("orderID", "") if isinstance(resp, dict) else "",
                "market": signal.market.question[:80],
                "side": signal.side.value,
                "price": price,
                "size_usd": signal.position_size_usd,
                "raw_result": resp,
            }

        except Exception as e:
            error_str = str(e)
            if "not enough balance" in error_str or "allowance" in error_str:
                log.error(
                    "insufficient_balance",
                    error=error_str,
                    hint="Deposit USDC into your Polymarket account at polymarket.com first",
                    market=signal.market.question[:60],
                )
            elif "Request exception" in error_str:
                log.error(
                    "network_error",
                    error=error_str,
                    hint="Cannot reach clob.polymarket.com — check DNS/firewall on your VPS",
                    market=signal.market.question[:60],
                )
            else:
                log.error("execution_failed", error=error_str)
            return {
                "status": "error",
                "error": error_str,
                "market": signal.market.question[:80],
            }

    def get_balance(self) -> float:
        """Get current USDC balance on Polymarket (Polygon network).
        
        For proxy/email wallets: checks the FUNDER address (proxy wallet).
        For EOA wallets: checks the WALLET_ADDRESS directly.
        """
        if self.config.dry_run:
            return -1.0  # Sentinel for dry run

        # Use funder address (proxy wallet) if set, otherwise wallet address
        funder = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
        check_address = funder if funder else self.config.wallet_address

        if not check_address:
            log.error("no_wallet_address_configured")
            return -1.0

        try:
            from web3 import Web3

            polygon_rpc = "https://polygon-rpc.com"
            w3 = Web3(Web3.HTTPProvider(polygon_rpc))

            # USDC.e on Polygon (what Polymarket uses)
            USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

            ERC20_ABI = [
                {
                    "constant": True,
                    "inputs": [{"name": "_owner", "type": "address"}],
                    "name": "balanceOf",
                    "outputs": [{"name": "balance", "type": "uint256"}],
                    "type": "function",
                },
                {
                    "constant": True,
                    "inputs": [],
                    "name": "decimals",
                    "outputs": [{"name": "", "type": "uint8"}],
                    "type": "function",
                },
            ]

            usdc = w3.eth.contract(
                address=Web3.to_checksum_address(USDC_ADDRESS),
                abi=ERC20_ABI,
            )

            wallet = Web3.to_checksum_address(check_address)
            raw_balance = usdc.functions.balanceOf(wallet).call()
            decimals = usdc.functions.decimals().call()

            balance = raw_balance / (10 ** decimals)
            is_proxy = "proxy" if funder else "EOA"
            log.info("wallet_balance", usdc=f"${balance:.2f}", type=is_proxy, address=check_address[:10] + "...")
            return balance

        except ImportError:
            log.error("web3_not_installed, run: pip install web3")
            return -1.0
        except Exception as e:
            log.error("balance_check_failed", error=str(e))
            return -1.0

    def get_open_orders(self) -> list[dict]:
        """Get current open orders."""
        if self.config.dry_run or not self.clob_client:
            return []

        try:
            orders = self.clob_client.get_orders()
            return orders if orders else []
        except Exception as e:
            log.error("get_orders_failed", error=str(e))
            return []

    def cancel_all_orders(self) -> bool:
        """Emergency: cancel all open orders."""
        if self.config.dry_run:
            log.info("dry_run_cancel_all")
            return True

        try:
            if self.clob_client:
                result = self.clob_client.cancel_all()
                log.info("cancelled_all_orders", result=result)
                return True
            return False
        except Exception as e:
            log.error("cancel_all_failed", error=str(e))
            return False

    def get_current_price(self, token_id: str) -> Optional[float]:
        """Get current mid price for a token from the CLOB order book."""
        if self.config.dry_run or not self.clob_client:
            return None

        try:
            book = self.clob_client.get_order_book(token_id)
            if not book:
                return None

            # Extract best bid/ask from order book
            bids = book.get("bids", [])
            asks = book.get("asks", [])

            best_bid = float(bids[0]["price"]) if bids else 0
            best_ask = float(asks[0]["price"]) if asks else 0

            if best_bid > 0 and best_ask > 0:
                return (best_bid + best_ask) / 2
            elif best_bid > 0:
                return best_bid
            elif best_ask > 0:
                return best_ask
            return None
        except Exception as e:
            log.debug("price_fetch_failed", token_id=token_id[:20], error=str(e))
            return None

    def exit_position(self, trade, current_price: float) -> dict:
        """
        Exit an existing position by selling what we own.

        If we bought YES tokens → sell YES tokens
        If we bought NO tokens (SELL signal) → sell NO tokens
        """
        if self.config.dry_run:
            return {
                "status": "dry_run_exit",
                "trade_id": trade.trade_id,
                "exit_price": current_price,
                "source": "polymarket",
            }

        if not self.clob_client:
            return {"status": "error", "error": "No CLOB client"}

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import SELL

            # Determine which token to sell
            token_id = trade.market_id  # This should be the token we bought
            # If we bought YES (BUY signal), sell YES token
            # If we bought NO (SELL signal), we need the NO token ID
            # For now, use the token_id from the original trade

            # Calculate size: original shares = size_usd / entry_price
            shares = trade.size_usd / trade.price if trade.price > 0 else 0
            if shares <= 0:
                return {"status": "error", "error": "Cannot calculate position size"}

            # Sell at current bid (slightly below mid for faster fill)
            sell_price = max(0.01, round(current_price - 0.01, 2))

            order_args = OrderArgs(
                token_id=token_id,
                price=sell_price,
                size=round(shares, 2),
                side=SELL,
            )

            signed_order = self.clob_client.create_order(order_args)
            resp = self.clob_client.post_order(signed_order, OrderType.GTC)

            success = resp.get("success", False) if isinstance(resp, dict) else False

            exit_pnl = (current_price - trade.price) * shares if trade.side == "BUY" else (trade.price - current_price) * shares

            log.info(
                "position_exited",
                question=trade.question[:60],
                entry=trade.price,
                exit=sell_price,
                pnl=f"${exit_pnl:.2f}",
            )

            return {
                "status": "exited" if success else "exit_failed",
                "trade_id": trade.trade_id,
                "exit_price": sell_price,
                "exit_pnl": exit_pnl,
                "shares": shares,
                "source": "polymarket",
                "raw_result": resp,
            }

        except Exception as e:
            log.error("exit_failed", error=str(e), trade_id=trade.trade_id[:20])
            return {"status": "error", "error": str(e), "trade_id": trade.trade_id}

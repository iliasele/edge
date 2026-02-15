"""
Monad Token Trader: The agent interacts with its own token on nad.fun.

How it works:
  - After each trading cycle, check net P&L
  - If profitable: agent buys its own token (drives price up)
  - If losing: agent sells token (reflects poor performance)
  - Token price becomes a live indicator of agent performance

This creates a "fund token" — holders are betting on the agent's ability.

Requirements:
  - monad/token_config.json (created by deploy_token.mjs)
  - MONAD_PRIVATE_KEY in .env
  - web3.py installed (pip install web3)
"""

import json
import os
import time
import structlog
from datetime import datetime, timezone

log = structlog.get_logger()

TOKEN_CONFIG_FILE = "monad/token_config.json"


class MonadTokenTrader:
    """Buys/sells the agent's own token on nad.fun based on performance."""

    # How much MON to spend per buy/sell action
    TRADE_AMOUNT_MON = 0.05  # 0.05 MON per trade
    
    # Only trade token if P&L changed by at least this much since last trade
    MIN_PNL_CHANGE = 0.50  # $0.50 change triggers token trade
    
    # Cooldown between token trades (seconds)
    TRADE_COOLDOWN = 600  # 10 minutes

    def __init__(self):
        self.config = None
        self.w3 = None
        self.account = None
        self.last_trade_time = 0
        self.last_pnl = 0.0
        self.enabled = False
        
        self._load_config()

    def _load_config(self):
        """Load token config from deploy script output."""
        if not os.path.exists(TOKEN_CONFIG_FILE):
            log.info("monad_token_not_deployed", hint="Run: node monad/deploy_token.mjs")
            return

        try:
            with open(TOKEN_CONFIG_FILE, "r") as f:
                self.config = json.load(f)

            private_key = os.getenv("MONAD_PRIVATE_KEY", "")
            if not private_key:
                log.warning("MONAD_PRIVATE_KEY not set, token trading disabled")
                return

            from web3 import Web3

            self.w3 = Web3(Web3.HTTPProvider(self.config["rpc_url"]))
            if not private_key.startswith("0x"):
                private_key = f"0x{private_key}"
            self.account = self.w3.eth.account.from_key(private_key)
            self.enabled = True

            log.info(
                "monad_token_trader_ready",
                token=self.config["token_symbol"],
                address=self.config["token_address"][:20] + "...",
                wallet=self.account.address[:12] + "...",
            )

        except ImportError:
            log.warning("web3 not installed, run: pip install web3")
        except Exception as e:
            log.error("monad_config_load_failed", error=str(e))

    def should_trade(self) -> bool:
        """Check if enough time has passed since last token trade."""
        if not self.enabled:
            return False
        return time.time() - self.last_trade_time >= self.TRADE_COOLDOWN

    def update_on_pnl(self, current_pnl: float, stats: dict = None):
        """
        Called after each cycle with current P&L.
        Decides whether to buy or sell the agent's token.
        """
        if not self.enabled or not self.should_trade():
            return

        pnl_change = current_pnl - self.last_pnl

        # Only trade if P&L changed significantly
        if abs(pnl_change) < self.MIN_PNL_CHANGE:
            return

        try:
            if pnl_change > 0:
                # Agent is making money → buy own token (bullish signal)
                self._buy_token(pnl_change)
            else:
                # Agent is losing money → sell own token (bearish signal)
                self._sell_token(abs(pnl_change))

            self.last_pnl = current_pnl
            self.last_trade_time = time.time()

        except Exception as e:
            log.error("monad_token_trade_failed", error=str(e))

    def _buy_token(self, pnl_gain: float):
        """Buy the agent's token on nad.fun bonding curve."""
        if not self.w3 or not self.account:
            return

        try:
            amount_mon = self.w3.to_wei(self.TRADE_AMOUNT_MON, 'ether')
            token_address = self.config["token_address"]
            router = self.config["bonding_curve_router"]

            # Check MON balance first
            balance = self.w3.eth.get_balance(self.account.address)
            if balance < amount_mon + self.w3.to_wei(0.01, 'ether'):  # need gas too
                log.debug("insufficient_mon_for_buy", balance=f"{self.w3.from_wei(balance, 'ether'):.4f} MON")
                return

            # BondingCurveRouter.buy() ABI
            buy_abi = [{
                "type": "function",
                "name": "buy",
                "stateMutability": "payable",
                "inputs": [{
                    "name": "params",
                    "type": "tuple",
                    "components": [
                        {"name": "amountOutMin", "type": "uint256"},
                        {"name": "token", "type": "address"},
                        {"name": "to", "type": "address"},
                        {"name": "deadline", "type": "uint256"},
                    ],
                }],
                "outputs": [],
            }]

            contract = self.w3.eth.contract(
                address=self.w3.to_checksum_address(router),
                abi=buy_abi,
            )

            deadline = int(time.time()) + 300  # 5 min deadline
            nonce = self.w3.eth.get_transaction_count(self.account.address)

            tx = contract.functions.buy((
                0,  # amountOutMin (0 = accept any, simplest for small amounts)
                self.w3.to_checksum_address(token_address),
                self.account.address,
                deadline,
            )).build_transaction({
                'from': self.account.address,
                'value': amount_mon,
                'nonce': nonce,
                'chainId': self.config['chain_id'],
            })

            # Estimate gas dynamically
            try:
                gas_estimate = self.w3.eth.estimate_gas(tx)
                tx['gas'] = int(gas_estimate * 1.2)  # 20% buffer
            except Exception:
                tx['gas'] = 300000  # fallback

            signed = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)

            log.info(
                "🟢 AGENT_BUY_TOKEN",
                token=self.config["token_symbol"],
                amount=f"{self.TRADE_AMOUNT_MON} MON",
                reason=f"P&L up ${pnl_gain:.2f}",
                tx=tx_hash.hex()[:20] + "...",
            )

        except Exception as e:
            log.error("token_buy_failed", error=str(e))

    def _sell_token(self, pnl_loss: float):
        """Sell the agent's token on nad.fun bonding curve."""
        if not self.w3 or not self.account:
            return

        try:
            token_address = self.config["token_address"]
            router = self.config["bonding_curve_router"]

            # Check token balance first
            erc20_abi = [{
                "type": "function", "name": "balanceOf",
                "inputs": [{"name": "account", "type": "address"}],
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
            }, {
                "type": "function", "name": "approve",
                "inputs": [
                    {"name": "spender", "type": "address"},
                    {"name": "amount", "type": "uint256"},
                ],
                "outputs": [{"name": "", "type": "bool"}],
                "stateMutability": "nonpayable",
            }]

            token_contract = self.w3.eth.contract(
                address=self.w3.to_checksum_address(token_address),
                abi=erc20_abi,
            )

            balance = token_contract.functions.balanceOf(self.account.address).call()
            if balance == 0:
                log.debug("no_tokens_to_sell")
                return

            # Sell 10% of holdings (don't dump everything)
            sell_amount = balance // 10
            if sell_amount == 0:
                sell_amount = balance  # If very small, sell all

            # Approve router to spend tokens
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            approve_tx = token_contract.functions.approve(
                self.w3.to_checksum_address(router),
                sell_amount,
            ).build_transaction({
                'from': self.account.address,
                'nonce': nonce,
                'chainId': self.config['chain_id'],
            })
            try:
                gas_est = self.w3.eth.estimate_gas(approve_tx)
                approve_tx['gas'] = int(gas_est * 1.2)
            except Exception:
                approve_tx['gas'] = 100000

            signed_approve = self.account.sign_transaction(approve_tx)
            approve_hash = self.w3.eth.send_raw_transaction(signed_approve.raw_transaction)
            
            # Wait for approve to confirm
            self.w3.eth.wait_for_transaction_receipt(approve_hash, timeout=30)

            # Sell via BondingCurveRouter
            sell_abi = [{
                "type": "function",
                "name": "sell",
                "stateMutability": "nonpayable",
                "inputs": [{
                    "name": "params",
                    "type": "tuple",
                    "components": [
                        {"name": "amountIn", "type": "uint256"},
                        {"name": "amountOutMin", "type": "uint256"},
                        {"name": "token", "type": "address"},
                        {"name": "to", "type": "address"},
                        {"name": "deadline", "type": "uint256"},
                    ],
                }],
                "outputs": [],
            }]

            sell_contract = self.w3.eth.contract(
                address=self.w3.to_checksum_address(router),
                abi=sell_abi,
            )

            deadline = int(time.time()) + 300

            sell_tx = sell_contract.functions.sell((
                sell_amount,
                0,  # amountOutMin
                self.w3.to_checksum_address(token_address),
                self.account.address,
                deadline,
            )).build_transaction({
                'from': self.account.address,
                'nonce': nonce + 1,
                'chainId': self.config['chain_id'],
            })
            
            try:
                gas_est = self.w3.eth.estimate_gas(sell_tx)
                sell_tx['gas'] = int(gas_est * 1.2)
            except Exception:
                sell_tx['gas'] = 300000

            signed_sell = self.account.sign_transaction(sell_tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_sell.raw_transaction)

            log.info(
                "🔴 AGENT_SELL_TOKEN",
                token=self.config["token_symbol"],
                amount=f"{sell_amount / 1e18:.2f} tokens",
                reason=f"P&L down ${pnl_loss:.2f}",
                tx=tx_hash.hex()[:20] + "...",
            )

        except Exception as e:
            log.error("token_sell_failed", error=str(e))

    def get_token_info(self) -> dict:
        """Get current token status for reporting."""
        if not self.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "token": self.config.get("token_symbol", "?"),
            "address": self.config.get("token_address", "?"),
            "network": self.config.get("network", "?"),
            "last_trade": self.last_trade_time,
            "last_pnl": self.last_pnl,
        }

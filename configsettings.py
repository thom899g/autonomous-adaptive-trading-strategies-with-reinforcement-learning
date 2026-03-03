"""
Configuration management for the trading system.
Centralizes all configurable parameters with environment variable support.
"""
import os
from typing import Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class FirebaseConfig:
    """Firebase configuration with validation."""
    credential_path: str
    project_id: str
    
    def __post_init__(self):
        if not os.path.exists(self.credential_path):
            raise FileNotFoundError(f"Firebase credential file not found: {self.credential_path}")

@dataclass
class ExchangeConfig:
    """Exchange API configuration."""
    name: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    sandbox: bool = True
    
    def __post_init__(self):
        if not self.sandbox and (not self.api_key or not self.api_secret):
            raise ValueError("Live trading requires API credentials")

@dataclass
class RLConfig:
    """Reinforcement Learning hyperparameters."""
    learning_rate: float = 0.001
    gamma: float = 0.99
    batch_size: int = 64
    memory_size: int = 10000
    tau: float = 0.005  # Target network update rate

@dataclass
class TradingConfig:
    """Trading system parameters."""
    symbols: tuple = ("BTC/USDT", "ETH/USDT")
    timeframe: str = "1h"
    initial_balance: float = 10000.0
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit

class Settings:
    """Singleton configuration manager."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize all configuration sections."""
        self.firebase = FirebaseConfig(
            credential_path=os.getenv("FIREBASE_CREDENTIAL_PATH", "./firebase-creds.json"),
            project_id=os.getenv("FIREBASE_PROJECT_ID", "trading-system-prod")
        )
        self.exchange = ExchangeConfig(
            name=os.getenv("EXCHANGE_NAME", "binance"),
            api_key=os.getenv("EXCHANGE_API_KEY", ""),
            api_secret=os.getenv("EXCHANGE_API_SECRET", ""),
            sandbox=os.getenv("EXCHANGE_SANDBOX", "true").lower() == "true"
        )
        self.rl = RLConfig()
        self.trading = TradingConfig()
        
        # Validation
        self._validate()
    
    def _validate(self):
        """Validate configuration integrity."""
        required_env_vars = ["FIREBASE_PROJECT_ID"]
        for var in required_env_vars:
            if not os.getenv(var):
                raise ValueError(f"Required environment variable not set: {var}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary for logging."""
        return {
            "firebase": {
                "project_id": self.firebase.project_id,
                "credential_path": self.firebase.credential_path
            },
            "exchange": {
                "name": self.exchange.name,
                "sandbox": self.exchange.sandbox
            },
            "rl": self.rl.__dict__,
            "trading": self.trading.__dict__
        }

# Global configuration instance
settings = Settings()
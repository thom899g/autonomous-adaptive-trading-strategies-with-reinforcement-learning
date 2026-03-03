"""
Market data ingestion with error handling and connection pooling.
Handles multiple data sources and formats with retry logic.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging
from datetime import datetime, timedelta
import ccxt
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Structured market data container."""
    symbol: str
    timeframe: str
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    timestamp: np.ndarray
    
    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.close), 5)  # OHLCV features

class DataIngestor:
    """Robust market data ingestor with retry logic and caching."""
    
    def __init__(self, exchange_id: str = "binance", sandbox: bool = True):
        """
        Initialize data ingestor with exchange connection.
        
        Args:
            exchange_id: Exchange identifier (binance, coinbase, etc.)
            sandbox: Use sandbox/testnet if available
        """
        self.exchange
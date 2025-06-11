"""Configuration and constants for the RedFlagProfits pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _Config:
    """Configuration settings."""

    # File paths
    DATA_DIR = Path("data")
    DICT_DIR = Path("data/dictionaries")
    PARQUET_FILE = Path("data/all_billionaires.parquet")
    LOG_FILE = Path("update.log")

    # API endpoints
    FORBES_API = "https://www.forbes.com/forbesapi/person/rtb/0/-estWorthPrev/true.json"
    FRED_API = "https://api.stlouisfed.org/fred/series/observations"

    # FRED series IDs
    CPI_SERIES = "CPIAUCNS"
    PCE_SERIES = "PCEPI"

    # Network settings
    REQUEST_TIMEOUT = 15
    MAX_RETRIES = 3
    RETRY_DELAY = 1

    # Parquet settings
    COMPRESSION_LEVEL = 22
    DATA_PAGE_SIZE = 1048576

    # Data processing
    FORBES_COLUMNS = [
        "finalWorth",
        "estWorthPrev",
        "privateAssetsWorth",
        "archivedWorth",
        "personName",
        "gender",
        "birthDate",
        "countryOfCitizenship",
        "state",
        "city",
        "source",
        "industries",
        "financialAssets",
    ]

    ASSET_COLUMNS = [
        "exchanges",
        "tickers",
        "companies",
        "shares",
        "prices",
        "currencies",
        "exchange_rates",
    ]

    DICTIONARY_NAMES = [
        "exchanges",
        "currencies",
        "industries",
        "companies",
        "countries",
        "sources",
    ]

    # Constants
    GENDER_MAP = {"M": 0, "F": 1}
    INFLATION_BUFFER_DAYS = 90
    INVALID_CODE = -1
    TRILLION = int(1e6)  # Conversion factor: millions → trillions

    # Field mappings for data processing
    ASSET_FIELD_MAPPINGS = [
        ("shares", "numberOfShares", 0.0),
        ("prices", "sharePrice", 0.0),
        ("exchange_rates", "exchangeRate", 1.0),
    ]

    COLUMN_MAPPINGS = [
        ("countryOfCitizenship", "countries", "country_code"),
        ("source", "sources", "source_code"),
    ]

    # Request headers
    HEADERS = {
        "authority": "www.forbes.com",
        "cache-control": "max-age=0",
        "user-agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36",
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "accept-language": "en-US,en;q=0.9",
    }


Config = _Config()

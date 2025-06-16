# redflags-data

A dataset of billionaire wealth and financial assets scraped from the Forbes Real Time Billionaires API.

## Dataset Structure

The data is stored in two parquet files:

### `billionaires.parquet`
| Column | Type | Description |
|--------|------|-------------|
| `date` | Date | Snapshot date (YYYYMMDD) |
| `personName` | Categorical | Full name |
| `lastName` | Categorical | Last name |
| `birthDate` | Date | Date of birth |
| `gender` | Categorical | Gender |
| `countryOfCitizenship` | Categorical | Country of citizenship |
| `city` | Categorical | City of residence |
| `state` | Categorical | State/region |
| `source` | Categorical | Primary source of wealth |
| `industries` | Categorical | Industry categories |
| `finalWorth` | Decimal(18,8) | Current net worth (millions USD) |
| `estWorthPrev` | Decimal(18,8) | Previous estimated worth |
| `archivedWorth` | Decimal(18,8) | Archived worth value |
| `privateAssetsWorth` | Decimal(18,8) | Private assets value |

### `assets.parquet`
| Column | Type | Description |
|--------|------|-------------|
| `date` | Date | Snapshot date (YYYYMMDD) |
| `personName` | Categorical | Billionaire name (links to billionaires table) |
| `companyName` | Categorical | Company name |
| `ticker` | Categorical | Stock ticker symbol |
| `exchange` | Categorical | Stock exchange |
| `currencyCode` | Categorical | Currency code |
| `currentPrice` | Decimal(18,11) | Current stock price |
| `sharePrice` | Decimal(18,11) | Share price at time of data |
| `numberOfShares` | Decimal(18,2) | Number of shares owned |
| `exchangeRate` | Decimal(18,8) | Currency exchange rate to USD |
| `exerciseOptionPrice` | Decimal(18,11) | Option exercise price |
| `interactive` | Boolean | Whether holding is interactive/liquid |

## How it works

1. **Fetch**: Scrapes live JSON data from Forbes API endpoints
2. **Process**: Extracts billionaire profiles and their financial assets
3. **Transform**: Applies proper data types (decimals for precision, categoricals for efficiency)
4. **Store**: Saves as compressed parquet files with date tracking

## Data Format

- **High precision**: Decimal types prevent floating-point rounding errors
- **Categorical encoding**: Efficient storage for repeated string values
- **Date tracking**: Each record tagged with snapshot date for time series analysis
- **Compression**: Brotli level 11 for minimal file size

The script handles data updates by replacing existing records for the same date, allowing you to build a historical dataset over time.

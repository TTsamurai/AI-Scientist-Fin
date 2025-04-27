import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import requests
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    Configuration parameters for the data pipeline.
    Load sensitive values from environment variables.
    """

    refresh_token: str = os.getenv("JQUANTS_TOKEN", "")
    target_ticker: str = "86040"
    start_date: str = "20230201"
    end_date: str = "20250101"
    output_dir: str = os.path.join("..", "price_data", "processed", target_ticker)
    train_ratio: float = 0.7
    val_ratio: float = 0.15


class JQuantsClient:
    """
    Client for interacting with the J-Quants API.
    Handles authentication and data retrieval.
    """

    BASE_URL = "https://api.jquants.com/v1"

    def __init__(self, refresh_token: str):
        """
        Initialize the client and acquire an access token.

        Args:
            refresh_token: The refresh token for J-Quants API authentication.
        Raises:
            EnvironmentError: If no refresh token is provided.
        """
        if not refresh_token:
            raise EnvironmentError("JQUANTS_TOKEN environment variable is not set.")
        self.token = self._get_id_token(refresh_token)
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def _get_id_token(self, refresh_token: str) -> str:
        """
        Exchange a refresh token for an ID token.

        Args:
            refresh_token: The J-Quants refresh token.
        Returns:
            A valid ID token string.
        Raises:
            HTTPError: If the request fails.
            ValueError: If the token is missing in the response.
        """
        url = f"{self.BASE_URL}/token/auth_refresh?refreshtoken={refresh_token}"
        response = requests.post(url)
        response.raise_for_status()
        token = response.json().get("idToken")
        if not token:
            raise ValueError("idToken not found in response.")
        return token

    def get_stock_quotes(self, ticker: str, start: str, end: str) -> List[dict]:
        """
        Fetch daily stock price quotes for a given ticker and date range.

        Args:
            ticker: The stock code (e.g., "86040").
            start: Start date in YYYYMMDD format.
            end: End date in YYYYMMDD format.
        Returns:
            A list of dictionaries containing daily quote data.
        Raises:
            HTTPError: If the request fails.
            ValueError: If expected data is missing in the response.
        """
        url = f"{self.BASE_URL}/prices/daily_quotes?code={ticker}&from={start}&to={end}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        quotes = response.json().get("daily_quotes")
        if quotes is None:
            raise ValueError("daily_quotes not found in response.")
        return quotes

    def get_listed_companies(self) -> List[dict]:
        """
        Retrieve information on all listed companies.

        Returns:
            A list of dictionaries with company info.
        Raises:
            HTTPError: If the request fails.
            ValueError: If expected data is missing in the response.
        """
        url = f"{self.BASE_URL}/listed/info"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        info = response.json().get("info")
        if info is None:
            raise ValueError("info not found in response.")
        return info


class StockDataProcessor:
    """
    Processes raw stock data into feature-engineered datasets,
    splits, scales, and saves them.
    """

    def __init__(self, raw_data: List[dict]):
        """
        Initialize the processor with raw data.

        Args:
            raw_data: A list of dictionaries from J-Quants daily quotes.
        """
        df = pd.DataFrame(raw_data)
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        self.df = df

    def engineer_features(self) -> pd.DataFrame:
        """
        Generate technical indicators and labels from raw price data.

        Returns:
            A DataFrame with engineered features and binary labels for
            3, 7, and 15 days ahead movements.
        """
        df = self.df.copy()
        df["Return"] = df["Close"].pct_change()
        df["SMA_5"] = df["Close"].rolling(5).mean()
        df["SMA_10"] = df["Close"].rolling(10).mean()
        df["Volatility_5"] = df["Close"].rolling(5).std()
        df["Momentum_5"] = df["Close"] - df["Close"].shift(5)
        for i in range(1, 15):
            df[f"Close_day{i}"] = df["Close"].shift(i)
        # Create labels for 3, 7, and 15 days ahead
        for h in [3, 7, 15]:
            df[f"Label_{h}"] = (df["Close"].shift(-h) > df["Close"]).astype(int)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def split_and_scale(
        self, df: pd.DataFrame, train_ratio: float, val_ratio: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into train/validation/test sets and apply standard scaling.

        Args:
            df: The feature-engineered DataFrame.
            train_ratio: Fraction for training set.
            val_ratio: Fraction for validation set.
        Returns:
            A tuple of scaled DataFrames: (train_df, val_df, test_df).
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        feature_cols = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Return",
            "SMA_5",
            "SMA_10",
            "Volatility_5",
            "Momentum_5",
        ] + [f"Close_day{i}" for i in range(1, 15)]

        scaler = StandardScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])

        return train_df, val_df, test_df

    @staticmethod
    def save_datasets(
        datasets: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], output_dir: str
    ) -> None:
        """
        Save the train, validation, and test datasets as CSV files.

        Args:
            datasets: Tuple containing (train_df, val_df, test_df).
            output_dir: Directory path to save CSV files.
        """
        os.makedirs(output_dir, exist_ok=True)
        names = ["train", "val", "test"]
        for name, df in zip(names, datasets):
            path = os.path.join(output_dir, f"{name}_stock_dataset.csv")
            df.to_csv(path, index=False)
            logger.info("Saved %s dataset with %d rows to %s", name, len(df), path)


def main() -> None:
    """
    Execute the full data retrieval and processing pipeline.

    Steps:
    1. Load configuration and authenticate client.
    2. Fetch raw stock data.
    3. Engineer features and labels.
    4. Split, scale, and save datasets.
    """
    cfg = Config()
    client = JQuantsClient(cfg.refresh_token)
    raw_data = client.get_stock_quotes(cfg.target_ticker, cfg.start_date, cfg.end_date)
    processor = StockDataProcessor(raw_data)
    features_df = processor.engineer_features()
    datasets = processor.split_and_scale(features_df, cfg.train_ratio, cfg.val_ratio)
    StockDataProcessor.save_datasets(datasets, cfg.output_dir)
    logger.info("Dataset creation complete.")


if __name__ == "__main__":
    main()

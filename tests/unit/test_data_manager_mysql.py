from unittest.mock import MagicMock, patch

import pandas as pd

from futureQuant.data.manager import DataManager


class TestDataManagerMySQL:
    def test_get_daily_data_uses_mysql_before_network_fetch(self):
        mysql_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "symbol": ["RB", "RB"],
                "open": [3800.0, 3810.0],
                "high": [3850.0, 3860.0],
                "low": [3750.0, 3760.0],
                "close": [3820.0, 3830.0],
                "volume": [100000, 110000],
                "open_interest": [500000, 501000],
            }
        )

        with patch("futureQuant.data.manager.DBManager") as mock_db_class, \
             patch("futureQuant.data.manager.FuturesCalendar"), \
             patch("futureQuant.data.manager.DataCleaner") as mock_cleaner_class, \
             patch("futureQuant.data.manager.ContractManager"), \
             patch("futureQuant.data.manager.AKShareFetcher") as mock_ak_class, \
             patch("futureQuant.data.manager.MySQLPriceLoaderConfig.from_env") as mock_mysql_config, \
             patch("futureQuant.data.manager.MySQLPriceLoader") as mock_mysql_loader_class:
            mock_db = MagicMock()
            mock_db.load_price_data.return_value = pd.DataFrame()
            mock_db_class.return_value = mock_db

            mock_cleaner = MagicMock()
            mock_cleaner.clean_ohlc.return_value = mysql_df
            mock_cleaner_class.return_value = mock_cleaner

            mock_mysql_config.return_value = MagicMock()
            mock_mysql_loader = MagicMock()
            mock_mysql_loader.load_daily_data.return_value = mysql_df.copy()
            mock_mysql_loader_class.return_value = mock_mysql_loader

            mock_fetcher = MagicMock()
            mock_ak_class.return_value = mock_fetcher

            dm = DataManager()
            result = dm.get_daily_data("RB", "2024-01-01", "2024-01-02")

            mock_mysql_loader.load_daily_data.assert_called_once_with(
                "RB", "2024-01-01", "2024-01-02"
            )
            mock_fetcher.fetch_daily.assert_not_called()
            assert not result.empty
            assert len(result) == 2

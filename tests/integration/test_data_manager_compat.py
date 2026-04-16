"""
Compatibility tests for DataManager legacy entry points.
"""

from unittest.mock import patch

import pandas as pd

from futureQuant.data.manager import DataManager


class TestDataManagerCompatibility:
    def test_get_daily_alias_delegates_to_get_daily_data(self):
        expected = pd.DataFrame(
            {
                "date": ["2024-08-01"],
                "close": [3800.0],
            }
        )

        with patch("futureQuant.data.manager.DBManager"), \
             patch("futureQuant.data.manager.FuturesCalendar"), \
             patch("futureQuant.data.manager.DataCleaner"), \
             patch("futureQuant.data.manager.ContractManager"), \
             patch("futureQuant.data.manager.AKShareFetcher"):
            dm = DataManager()

            with patch.object(dm, "get_daily_data", return_value=expected) as mock_get_daily_data:
                result = dm.get_daily(
                    symbol="RB2501",
                    start_date="2024-08-01",
                    end_date="2024-08-01",
                )

            mock_get_daily_data.assert_called_once_with(
                symbol="RB2501",
                start_date="2024-08-01",
                end_date="2024-08-01",
                source="akshare",
                use_cache=True,
                auto_clean=True,
            )
            assert result.equals(expected)

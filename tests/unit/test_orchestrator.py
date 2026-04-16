"""
Tests for MultiAgentFactorMiner orchestration behavior.
"""

from unittest.mock import MagicMock, patch

import pandas as pd

from futureQuant.agent.orchestrator import MultiAgentFactorMiner


class TestMultiAgentFactorMinerDataLoading:
    """Validate data loading and compatibility behavior."""

    def test_prepare_data_uses_data_manager_get_daily_data(self):
        """The orchestrator should fetch via DataManager.get_daily_data()."""
        sample_df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "open": [3800.0, 3810.0],
                "high": [3850.0, 3860.0],
                "low": [3750.0, 3760.0],
                "close": [3820.0, 3830.0],
                "volume": [100000, 110000],
            }
        )

        with patch("futureQuant.data.manager.DataManager") as mock_dm_class:

            mock_dm = MagicMock()
            mock_dm.get_daily_data.return_value = sample_df.copy()
            mock_dm_class.return_value = mock_dm

            miner = MultiAgentFactorMiner(
                symbols=["RB2501"],
                start_date="2024-01-01",
                end_date="2024-01-02",
            )

            data = miner._prepare_data()

            mock_dm.get_daily_data.assert_called_once_with(
                "RB2501",
                "2024-01-01",
                "2024-01-02",
            )
            assert not data.empty
            assert "symbol" in data.columns
            assert set(data["symbol"]) == {"RB2501"}

    def test_prepare_data_uses_continuous_contract_for_variety_code(self):
        sample_df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "open": [3800.0, 3810.0],
                "high": [3850.0, 3860.0],
                "low": [3750.0, 3760.0],
                "close": [3820.0, 3830.0],
                "volume": [100000, 110000],
            }
        )

        with patch("futureQuant.data.manager.DataManager") as mock_dm_class:
            mock_dm = MagicMock()
            mock_dm.get_continuous_contract.return_value = sample_df.copy()
            mock_dm_class.return_value = mock_dm

            miner = MultiAgentFactorMiner(
                symbols=["RB"],
                start_date="2024-01-01",
                end_date="2024-01-02",
            )

            data = miner._prepare_data()

            mock_dm.get_continuous_contract.assert_called_once_with(
                "RB",
                "2024-01-01",
                "2024-01-02",
            )
            assert not data.empty
            assert "symbol" in data.columns

import pandas as pd

from futureQuant.agent.context import MiningContext


class TestMiningContext:
    def test_returns_index_aligns_with_data_date_index(self):
        data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "close": [100.0, 101.0, 102.0],
            }
        )
        returns = pd.Series([0.01, -0.02, 0.03], name="future_returns")

        ctx = MiningContext(
            data=data,
            returns=returns,
            symbols=["RB"],
            start_date="2024-01-01",
            end_date="2024-01-03",
            config={},
        )

        assert isinstance(ctx.data.index, pd.DatetimeIndex)
        assert isinstance(ctx.returns.index, pd.DatetimeIndex)
        assert ctx.returns.index.equals(ctx.data.index)

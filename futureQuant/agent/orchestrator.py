"""
多 Agent 因子挖掘编排器

实现 MultiAgentFactorMiner 主入口类，协调各挖掘 Agent 的工作流程：
1. 数据准备
2. 并行/串行运行挖掘 Agent（技术、基本面、宏观）
3. 因子融合与去相关
4. 因子验证
5. 回测评估
6. 汇总结果
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.logger import get_logger
from .base import AgentResult, AgentStatus
from .context import MiningContext
from .miners.technical_agent import TechnicalMiningAgent
from .miners.fundamental_agent import FundamentalMiningAgent
from .miners.macro_agent import MacroMiningAgent
from .miners.fusion_agent import FusionAgent

logger = get_logger('agent.orchestrator')


@dataclass
class MiningResult:
    """
    挖掘流程最终结果

    Attributes:
        selected_factors: 最终筛选出的因子列表
        factor_scores: 因子评分字典
        factor_data: 因子值 DataFrame
        backtest_results: 回测结果字典
        agent_results: 各 Agent 的执行结果
        elapsed_seconds: 总耗时
        summary: 摘要信息字典
    """

    selected_factors: List[Any] = field(default_factory=list)
    factor_scores: Dict[str, float] = field(default_factory=dict)
    factor_data: Optional[pd.DataFrame] = None
    backtest_results: Dict[str, Any] = field(default_factory=dict)
    agent_results: Dict[str, AgentResult] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_factors(self) -> int:
        """最终因子数量"""
        return len(self.selected_factors)

    def __repr__(self) -> str:
        return (
            f"MiningResult("
            f"n_factors={self.n_factors}, "
            f"elapsed={self.elapsed_seconds:.1f}s)"
        )


class MultiAgentFactorMiner:
    """
    多 Agent 因子挖掘主入口类

    协调技术、基本面、宏观三类挖掘 Agent，通过 FusionAgent 进行因子融合，
    最终输出经过筛选和评估的高质量因子集合。

    工作流程:
        1. 准备数据（OHLCV + 收益率）
        2. 并行运行挖掘 Agent（技术/基本面/宏观）
        3. FusionAgent 去相关 + ICIR 加权合成
        4. 汇总结果

    使用示例:
        >>> miner = MultiAgentFactorMiner(
        ...     symbols=['RB', 'HC'],
        ...     start_date='2020-01-01',
        ...     end_date='2023-12-31',
        ...     data=price_df,
        ... )
        >>> result = miner.run(n_workers=4)
        >>> print(f"Found {result.n_factors} factors")
    """

    DEFAULT_CONFIG = {
        # 技术因子配置
        'technical': {
            'ic_threshold': 0.02,
            'momentum_windows': [5, 10, 20, 60, 120],
            'volatility_windows': [10, 20, 60],
            'volume_windows': [5, 10, 20],
            'rsi_windows': [6, 14, 21],
        },
        # 基本面因子配置
        'fundamental': {
            'ic_threshold': 0.02,
            'basis_lag': 1,
            'inventory_lag': 3,
            'warehouse_lag': 2,
        },
        # 宏观因子配置
        'macro': {
            'ic_threshold': 0.01,
        },
        # 融合配置
        'fusion': {
            'corr_threshold': 0.8,
            'min_icir': 0.3,
        },
    }

    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        data: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化 MultiAgentFactorMiner

        Args:
            symbols: 品种代码列表，如 ['RB', 'HC', 'I']
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            data: 价格数据 DataFrame（包含 OHLCV 列）。
                  若为 None，将尝试通过 DataManager 获取。
            config: 自定义配置，会与 DEFAULT_CONFIG 合并
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self._data = data

        # 合并配置
        self.config = self._merge_config(self.DEFAULT_CONFIG, config or {})

        # 初始化各 Agent
        self._technical_agent = TechnicalMiningAgent(
            name='technical_miner',
            config=self.config.get('technical', {}),
        )
        self._fundamental_agent = FundamentalMiningAgent(
            name='fundamental_miner',
            config=self.config.get('fundamental', {}),
        )
        self._macro_agent = MacroMiningAgent(
            name='macro_miner',
            config=self.config.get('macro', {}),
        )
        self._fusion_agent = FusionAgent(
            name='fusion_agent',
            config=self.config.get('fusion', {}),
        )

        logger.info(
            f"MultiAgentFactorMiner initialized: "
            f"symbols={symbols}, date=[{start_date}, {end_date}]"
        )

    @staticmethod
    def _merge_config(base: Dict, override: Dict) -> Dict:
        """
        深度合并配置字典

        Args:
            base: 基础配置
            override: 覆盖配置

        Returns:
            合并后的配置
        """
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = MultiAgentFactorMiner._merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def _prepare_data(self) -> pd.DataFrame:
        """
        准备价格数据

        若初始化时未传入 data，尝试通过 DataManager 获取。

        Returns:
            价格 DataFrame（OHLCV）

        Raises:
            ValueError: 数据获取失败或数据为空
        """
        if self._data is not None:
            logger.info(f"Using provided data: shape={self._data.shape}")
            return self._data

        # 尝试通过 DataManager 获取数据
        logger.info("No data provided, attempting to fetch via DataManager...")
        try:
            from ..data.manager import DataManager
            dm = DataManager()
            dfs = []
            for symbol in self.symbols:
                df = dm.get_daily(symbol, self.start_date, self.end_date)
                if df is not None and not df.empty:
                    df['symbol'] = symbol
                    dfs.append(df)
            if dfs:
                data = pd.concat(dfs, axis=0)
                logger.info(f"Fetched data: shape={data.shape}")
                return data
        except Exception as e:
            logger.warning(f"DataManager fetch failed: {e}")

        raise ValueError(
            "No data available. Please provide data via the 'data' parameter "
            "or ensure DataManager is configured."
        )

    def _compute_returns(self, data: pd.DataFrame) -> pd.Series:
        """
        计算未来收益率（shift(-1)）

        Args:
            data: 价格 DataFrame，需要包含 'close' 列

        Returns:
            未来收益率 Series
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column to compute returns")

        close = data['close']
        returns = close.pct_change().shift(-1)
        returns.name = 'future_returns'
        logger.debug(f"Computed returns: {returns.notna().sum()} valid values")
        return returns

    def run(self, n_workers: int = 4) -> MiningResult:
        """
        运行完整的因子挖掘流程

        流程:
        1. 准备数据
        2. 并行运行挖掘 Agent（技术/基本面/宏观）
        3. 收集所有发现的因子
        4. FusionAgent 去相关 + 合成
        5. 汇总结果

        Args:
            n_workers: 并行工作线程数（用于并行运行挖掘 Agent）

        Returns:
            MiningResult: 完整挖掘结果
        """
        start_time = time.time()
        logger.info(f"Starting mining pipeline with {n_workers} workers...")

        # 1. 准备数据
        try:
            data = self._prepare_data()
        except ValueError as e:
            logger.error(f"Data preparation failed: {e}")
            return MiningResult(
                summary={'error': str(e)},
                elapsed_seconds=time.time() - start_time,
            )

        # 2. 计算收益率
        try:
            returns = self._compute_returns(data)
        except ValueError as e:
            logger.error(f"Returns computation failed: {e}")
            return MiningResult(
                summary={'error': str(e)},
                elapsed_seconds=time.time() - start_time,
            )

        # 3. 构建上下文
        context = MiningContext(
            data=data,
            returns=returns,
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date,
            config=self.config,
        )

        # 4. 并行运行挖掘 Agent
        mining_agents = [
            self._technical_agent,
            self._fundamental_agent,
            self._macro_agent,
        ]

        agent_results: Dict[str, AgentResult] = {}
        all_factors = []
        all_factor_data_frames = []

        if n_workers > 1:
            # 并行执行
            logger.info(f"Running {len(mining_agents)} mining agents in parallel...")
            with ThreadPoolExecutor(max_workers=min(n_workers, len(mining_agents))) as executor:
                future_to_agent = {
                    executor.submit(agent.run, {'context': context}): agent
                    for agent in mining_agents
                }
                for future in as_completed(future_to_agent):
                    agent = future_to_agent[future]
                    try:
                        result = future.result()
                        agent_results[agent.name] = result
                        if result.is_success and result.factors:
                            all_factors.extend(result.factors)
                        if result.is_success and result.data is not None:
                            all_factor_data_frames.append(result.data)
                        logger.info(
                            f"Agent [{agent.name}] completed: "
                            f"n_factors={result.n_factors}"
                        )
                    except Exception as e:
                        logger.error(f"Agent [{agent.name}] raised exception: {e}")
                        agent_results[agent.name] = AgentResult(
                            agent_name=agent.name,
                            status=AgentStatus.FAILED,
                            errors=[str(e)],
                        )
        else:
            # 串行执行
            logger.info("Running mining agents sequentially...")
            for agent in mining_agents:
                result = agent.run({'context': context})
                agent_results[agent.name] = result
                if result.is_success and result.factors:
                    all_factors.extend(result.factors)
                if result.is_success and result.data is not None:
                    all_factor_data_frames.append(result.data)

        logger.info(f"Mining agents completed. Total discovered factors: {len(all_factors)}")

        # 5. 合并因子数据
        combined_factor_data: Optional[pd.DataFrame] = None
        if all_factor_data_frames:
            try:
                combined_factor_data = pd.concat(all_factor_data_frames, axis=1)
                # 去除重复列名
                combined_factor_data = combined_factor_data.loc[
                    :, ~combined_factor_data.columns.duplicated()
                ]
                logger.info(f"Combined factor data: shape={combined_factor_data.shape}")
            except Exception as e:
                logger.warning(f"Failed to combine factor data: {e}")

        # 6. 运行 FusionAgent
        fusion_context = {
            'context': context,
            'factors': all_factors,
            'factor_data': combined_factor_data,
            'returns': returns,
        }
        fusion_result = self._fusion_agent.run(fusion_context)
        agent_results['fusion_agent'] = fusion_result

        selected_factors = fusion_result.factors if fusion_result.is_success else all_factors
        final_factor_data = fusion_result.data if fusion_result.is_success else combined_factor_data

        # 7. 更新上下文
        context.add_discovered_factors(all_factors)
        context.add_validated_factors(selected_factors)
        if fusion_result.metrics:
            context.set_factor_scores(
                {k: v for k, v in fusion_result.metrics.items() if isinstance(v, float)}
            )

        elapsed = time.time() - start_time

        # 8. 构建摘要
        summary = {
            'total_discovered': len(all_factors),
            'total_selected': len(selected_factors),
            'agent_stats': {
                name: {
                    'status': r.status.value,
                    'n_factors': r.n_factors,
                    'elapsed': r.elapsed_seconds,
                }
                for name, r in agent_results.items()
            },
            'elapsed_seconds': elapsed,
        }

        logger.info(
            f"Mining pipeline completed: "
            f"discovered={len(all_factors)}, "
            f"selected={len(selected_factors)}, "
            f"elapsed={elapsed:.1f}s"
        )

        return MiningResult(
            selected_factors=selected_factors,
            factor_scores=context.factor_scores,
            factor_data=final_factor_data,
            backtest_results=context.backtest_results,
            agent_results=agent_results,
            elapsed_seconds=elapsed,
            summary=summary,
        )

    def run_backtest(
        self,
        factors: Optional[List[Any]] = None,
        risk_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        用筛选出的因子运行回测

        Args:
            factors: 因子列表。若为 None，使用最近一次 run() 的结果。
            risk_config: 风险控制配置，如最大回撤、波动率目标等

        Returns:
            回测结果字典，包含收益率、夏普比率、最大回撤等指标
        """
        if factors is None:
            logger.warning("No factors provided for backtest")
            return {}

        risk_config = risk_config or {}
        logger.info(f"Running backtest with {len(factors)} factors...")

        try:
            data = self._prepare_data()
            returns = self._compute_returns(data)

            # 计算因子值
            factor_values = {}
            for factor in factors:
                try:
                    fv = factor.compute(data)
                    factor_values[factor.name] = fv
                except Exception as e:
                    logger.warning(f"Failed to compute factor {factor.name}: {e}")

            if not factor_values:
                logger.warning("No valid factor values computed")
                return {}

            factor_df = pd.DataFrame(factor_values)

            # 简单等权组合回测
            # 对每个因子值做 z-score 标准化
            z_scores = factor_df.apply(
                lambda col: (col - col.rolling(60, min_periods=20).mean())
                / col.rolling(60, min_periods=20).std()
            )

            # 等权合成信号
            composite_signal = z_scores.mean(axis=1)

            # 计算策略收益（信号 * 未来收益）
            strategy_returns = composite_signal.shift(1).fillna(0).clip(-3, 3)
            strategy_returns = strategy_returns / strategy_returns.abs().rolling(20).mean().clip(lower=0.1)
            strategy_returns = strategy_returns * returns

            # 计算绩效指标
            valid_returns = strategy_returns.dropna()
            if len(valid_returns) < 10:
                return {'error': 'insufficient data for backtest'}

            total_return = (1 + valid_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(valid_returns)) - 1
            volatility = valid_returns.std() * np.sqrt(252)
            sharpe = annual_return / volatility if volatility > 0 else 0

            # 最大回撤
            cumulative = (1 + valid_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()

            results = {
                'total_return': float(total_return),
                'annual_return': float(annual_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe),
                'max_drawdown': float(max_drawdown),
                'n_factors': len(factors),
                'n_periods': len(valid_returns),
            }

            logger.info(
                f"Backtest completed: "
                f"annual_return={annual_return:.2%}, "
                f"sharpe={sharpe:.2f}, "
                f"max_drawdown={max_drawdown:.2%}"
            )
            return results

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {'error': str(e)}

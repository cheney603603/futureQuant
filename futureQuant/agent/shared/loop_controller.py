"""
AgentLoopController - 通用 Agent Loop 控制器

提供：
- 状态机管理（PENDING/RUNNING/SUCCESS/FAILED/PAUSED）
- 指数退避重试机制
- 超时控制
- 回退策略（graceful degradation）
- 结构化日志记录

使用示例：
    >>> controller = AgentLoopController(
    ...     name="data_collector",
    ...     retry_strategy=RetryStrategy(max_retries=3, base_delay=2.0),
    ...     timeout=300,
    ... )
    >>> async with controller.run():
    ...     await fetch_data()
"""
from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Awaitable
from contextlib import asynccontextmanager

from ...core.logger import get_logger

logger = get_logger('agent.shared.loop')

T = TypeVar('T')


class LoopState(Enum):
    """Agent Loop 状态枚举"""
    PENDING = "pending"        # 等待开始
    RUNNING = "running"       # 执行中
    SUCCESS = "success"       # 成功完成
    FAILED = "failed"         # 执行失败
    PAUSED = "paused"         # 已暂停（可恢复）
    RETRYING = "retrying"     # 重试中


@dataclass
class RetryStrategy:
    """
    重试策略配置

    Attributes:
        max_retries: 最大重试次数
        base_delay: 基础重试延迟（秒）
        exponential_base: 指数退避底数（默认2）
        max_delay: 最大延迟上限（秒）
        jitter: 随机抖动系数（0-1）
    """
    max_retries: int = 3
    base_delay: float = 1.0
    exponential_base: float = 2.0
    max_delay: float = 60.0
    jitter: float = 0.1

    def get_delay(self, attempt: int) -> float:
        """
        计算第 attempt 次重试的延迟时间

        Args:
            attempt: 当前重试次数（从1开始）

        Returns:
            延迟秒数
        """
        import random
        delay = min(
            self.base_delay * (self.exponential_base ** (attempt - 1)),
            self.max_delay
        )
        # 添加随机抖动
        if self.jitter > 0:
            jitter_range = delay * self.jitter
            delay += random.uniform(-jitter_range, jitter_range)
        return max(0, delay)


@dataclass
class LoopMetrics:
    """Loop 执行指标"""
    attempts: int = 0
    retries: int = 0
    total_elapsed: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def record_stage(self, stage: str, elapsed: float):
        """记录阶段耗时"""
        self.stage_times[stage] = elapsed

    def record_error(self, error: str):
        """记录错误"""
        self.errors.append(error)

    def record_warning(self, warning: str):
        """记录警告"""
        self.warnings.append(warning)

    def record_retry(self):
        """记录一次重试"""
        self.retries += 1

    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {
            'attempts': self.attempts,
            'retries': self.retries,
            'total_elapsed': self.total_elapsed,
            'stage_times': self.stage_times,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
        }


@dataclass
class FallbackResult:
    """
    回退策略结果

    Attributes:
        used_fallback: 是否使用了回退方案
        degraded_level: 降级程度（0=无降级, 1=轻度, 2=中度, 3=重度）
        result: 回退后的执行结果
        message: 回退说明
    """
    used_fallback: bool = False
    degraded_level: int = 0
    result: Any = None
    message: str = ""


class AgentLoopController:
    """
    通用 Agent Loop 控制器

    管理 Agent 的完整执行生命周期：
    1. 状态机流转
    2. 重试机制（指数退避）
    3. 超时控制
    4. 回退策略（graceful degradation）
    5. 结构化日志

    支持同步和异步两种执行模式。

    使用示例（异步）：
        >>> controller = AgentLoopController(
        ...     name="fetcher",
        ...     retry_strategy=RetryStrategy(max_retries=3),
        ...     timeout=120,
        ... )
        >>> result = await controller.run_async(fetch_coro)
        >>> if result.is_fallback:
        ...     logger.warning(f"Used fallback: {result.message}")

    使用示例（同步）：
        >>> result = controller.run_sync(heavy_task)
    """

    def __init__(
        self,
        name: str,
        retry_strategy: Optional[RetryStrategy] = None,
        timeout: Optional[float] = None,
        on_stage_complete: Optional[Callable[[str, float], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_fallback: Optional[Callable[[FallbackResult], None]] = None,
    ):
        """
        初始化 AgentLoopController

        Args:
            name: Agent 名称（用于日志）
            retry_strategy: 重试策略，为 None 则不重试
            timeout: 单次执行超时（秒），为 None 则不设限
            on_stage_complete: 阶段完成回调 (stage_name, elapsed)
            on_error: 错误回调 (exception)
            on_fallback: 回退触发回调 (FallbackResult)
        """
        self.name = name
        self.retry_strategy = retry_strategy or RetryStrategy(max_retries=0)
        self.timeout = timeout
        self.on_stage_complete = on_stage_complete
        self.on_error = on_error
        self.on_fallback = on_fallback

        self._state: LoopState = LoopState.PENDING
        self._metrics = LoopMetrics()
        self._current_stage: Optional[str] = None
        self._start_time: Optional[float] = None
        self._fallback_result: Optional[FallbackResult] = None

    @property
    def state(self) -> LoopState:
        """当前状态"""
        return self._state

    @property
    def metrics(self) -> LoopMetrics:
        """执行指标"""
        return self._metrics

    @property
    def is_running(self) -> bool:
        """是否运行中"""
        return self._state == LoopState.RUNNING

    @property
    def is_fallback(self) -> bool:
        """是否使用了回退方案"""
        return (
            self._fallback_result is not None
            and self._fallback_result.used_fallback
        )

    @property
    def fallback_result(self) -> Optional[FallbackResult]:
        """回退结果"""
        return self._fallback_result

    def set_state(self, state: LoopState, reason: str = ""):
        """
        设置状态（带日志）

        Args:
            state: 新状态
            reason: 状态变更原因
        """
        old = self._state
        self._state = state
        if reason:
            logger.debug(
                f"[{self.name}] State: {old.value} → {state.value} ({reason})"
            )
        else:
            logger.debug(f"[{self.name}] State: {old.value} → {state.value}")

    def _log_stage_start(self, stage: str):
        """记录阶段开始"""
        self._current_stage = stage
        logger.info(f"[{self.name}] ▶ Stage: {stage}")

    def _log_stage_end(self, stage: str, elapsed: float, success: bool = True):
        """记录阶段结束"""
        self._metrics.record_stage(stage, elapsed)
        if self.on_stage_complete:
            self.on_stage_complete(stage, elapsed)
        status = "✓" if success else "✗"
        logger.info(
            f"[{self.name}] {status} Stage: {stage} ({elapsed:.2f}s)"
        )
        self._current_stage = None

    def _log_error(self, exc: Exception, context: str = ""):
        """记录错误"""
        msg = f"{type(exc).__name__}: {exc}"
        if context:
            msg = f"[{context}] {msg}"
        self._metrics.record_error(msg)
        logger.error(f"[{self.name}] Error: {msg}")
        if self.on_error:
            self.on_error(exc)

    def _log_warning(self, warning: str):
        """记录警告"""
        self._metrics.record_warning(warning)
        logger.warning(f"[{self.name}] Warning: {warning}")

    # ---- 同步执行模式 ----

    def run_sync(
        self,
        task: Callable[[], T],
        fallback: Optional[Callable[[Exception], FallbackResult]] = None,
    ) -> T:
        """
        同步执行任务（带重试+超时）

        Args:
            task: 要执行的同步任务
            fallback: 回退策略函数，接收异常返回 FallbackResult

        Returns:
            任务返回值

        Raises:
            任务本身抛出的异常（所有重试失败后）
        """
        self._start_time = time.time()
        self.set_state(LoopState.RUNNING)
        self._metrics.attempts = 0

        last_error: Optional[Exception] = None

        # 重试循环
        for attempt in range(1, self.retry_strategy.max_retries + 2):
            self._metrics.attempts = attempt

            if attempt > 1:
                delay = self.retry_strategy.get_delay(attempt - 1)
                logger.info(
                    f"[{self.name}] Retry {attempt - 1}/{self.retry_strategy.max_retries} "
                    f"after {delay:.1f}s..."
                )
                self._metrics.record_retry()
                self.set_state(LoopState.RETRYING)
                time.sleep(delay)

            self.set_state(LoopState.RUNNING)

            # 超时包装
            try:
                if self.timeout:
                    result = self._run_with_timeout(task, self.timeout)
                else:
                    result = task()
                # 成功
                elapsed = time.time() - self._start_time
                self._metrics.total_elapsed = elapsed
                self.set_state(LoopState.SUCCESS)
                return result

            except Exception as exc:
                last_error = exc
                self._log_error(exc)
                tb = traceback.format_exc()
                logger.debug(f"[{self.name}] Traceback:\n{tb}")

                # 检查是否应触发回退
                if fallback and attempt <= self.retry_strategy.max_retries + 1:
                    fb = self._try_fallback(fallback, exc)
                    if fb.used_fallback:
                        return fb.result

        # 所有重试都失败了
        self.set_state(LoopState.FAILED)
        elapsed = time.time() - self._start_time
        self._metrics.total_elapsed = elapsed
        self._log_error(last_error or RuntimeError("Unknown error"))
        raise last_error

    def _run_with_timeout(
        self,
        task: Callable[[], T],
        timeout: float,
    ) -> T:
        """带超时的执行（轮询检测，适用于同步代码）"""
        import threading
        result = [None]
        error = [None]
        done = threading.Event()

        def target():
            try:
                result[0] = task()
            except Exception as e:
                error[0] = e
            finally:
                done.set()

        t = threading.Thread(target=target)
        t.start()
        if not done.wait(timeout):
            t.join(0.001)
            raise TimeoutError(
                f"[{self.name}] Task timed out after {timeout}s"
            )
        if error[0]:
            raise error[0]
        return result[0]

    # ---- 异步执行模式 ----

    async def run_async(self, task: Awaitable[T]) -> T:
        """
        异步执行任务（带重试+超时）

        Args:
            task: 要执行的异步协程

        Returns:
            任务返回值

        Raises:
            任务本身抛出的异常
        """
        self._start_time = time.time()
        self.set_state(LoopState.RUNNING)
        self._metrics.attempts = 0

        last_error: Optional[Exception] = None

        for attempt in range(1, self.retry_strategy.max_retries + 2):
            self._metrics.attempts = attempt

            if attempt > 1:
                delay = self.retry_strategy.get_delay(attempt - 1)
                logger.info(
                    f"[{self.name}] Retry {attempt - 1}/{self.retry_strategy.max_retries} "
                    f"after {delay:.1f}s..."
                )
                self._metrics.record_retry()
                self.set_state(LoopState.RETRYING)
                await asyncio.sleep(delay)

            self.set_state(LoopState.RUNNING)

            # 超时包装
            try:
                if self.timeout:
                    result = await asyncio.wait_for(task, timeout=self.timeout)
                else:
                    result = await task
                # 成功
                elapsed = time.time() - self._start_time
                self._metrics.total_elapsed = elapsed
                self.set_state(LoopState.SUCCESS)
                return result

            except asyncio.TimeoutError:
                exc = TimeoutError(
                    f"[{self.name}] Task timed out after {self.timeout}s"
                )
                last_error = exc
                self._log_error(exc)

            except Exception as exc:
                last_error = exc
                self._log_error(exc)
                tb = traceback.format_exc()
                logger.debug(f"[{self.name}] Traceback:\n{tb}")

        # 所有重试失败
        self.set_state(LoopState.FAILED)
        elapsed = time.time() - self._start_time
        self._metrics.total_elapsed = elapsed
        if last_error:
            self._log_error(last_error)
        raise last_error

    def _try_fallback(
        self,
        fallback_fn: Callable[[Exception], FallbackResult],
        exc: Exception,
    ) -> FallbackResult:
        """尝试执行回退策略"""
        try:
            fb = fallback_fn(exc)
            self._fallback_result = fb
            if fb.used_fallback:
                self._log_warning(
                    f"Fallback triggered: {fb.message} "
                    f"(degraded_level={fb.degraded_level})"
                )
                if self.on_fallback:
                    self.on_fallback(fb)
            return fb
        except Exception as fallback_exc:
            self._log_warning(
                f"Fallback itself failed: {type(fallback_exc).__name__}: {fallback_exc}"
            )
            return FallbackResult(used_fallback=False)

    # ---- 分阶段执行 ----

    def run_stages(
        self,
        stages: Dict[str, Callable[[], Any]],
        stop_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        分阶段执行任务（带独立计时）

        Args:
            stages: {阶段名: 任务函数} 的字典
            stop_on_error: 遇到错误是否停止后续阶段

        Returns:
            {阶段名: 结果} 的字典
        """
        results: Dict[str, Any] = {}
        for stage_name, stage_fn in stages.items():
            self._log_stage_start(stage_name)
            start = time.time()

            try:
                result = self.run_sync(stage_fn)
                results[stage_name] = result
                elapsed = time.time() - start
                self._log_stage_end(stage_name, elapsed, success=True)
            except Exception as exc:
                elapsed = time.time() - start
                self._log_stage_end(stage_name, elapsed, success=False)
                self._log_error(exc, context=stage_name)
                if stop_on_error:
                    break

        return results

    async def run_stages_async(
        self,
        stages: Dict[str, Awaitable[Any]],
        stop_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        异步分阶段执行

        Args:
            stages: {阶段名: 协程} 的字典
            stop_on_error: 遇到错误是否停止

        Returns:
            {阶段名: 结果} 的字典
        """
        results: Dict[str, Any] = {}
        for stage_name, stage_coro in stages.items():
            self._log_stage_start(stage_name)
            start = time.time()

            try:
                result = await self.run_async(stage_coro)
                results[stage_name] = result
                elapsed = time.time() - start
                self._log_stage_end(stage_name, elapsed, success=True)
            except Exception as exc:
                elapsed = time.time() - start
                self._log_stage_end(stage_name, elapsed, success=False)
                self._log_error(exc, context=stage_name)
                if stop_on_error:
                    break

        return results

    @asynccontextmanager
    async def context(self):
        """异步上下文管理器"""
        self._start_time = time.time()
        self.set_state(LoopState.RUNNING)
        try:
            yield self
            elapsed = time.time() - self._start_time
            self._metrics.total_elapsed = elapsed
            if self._state == LoopState.RUNNING:
                self.set_state(LoopState.SUCCESS)
        except Exception as exc:
            elapsed = time.time() - self._start_time
            self._metrics.total_elapsed = elapsed
            self.set_state(LoopState.FAILED)
            self._log_error(exc)
            raise

    def pause(self, reason: str = ""):
        """暂停 Loop"""
        if self._state == LoopState.RUNNING:
            self.set_state(LoopState.PAUSED, reason)
            logger.info(f"[{self.name}] Paused: {reason}")

    def resume(self):
        """恢复 Loop"""
        if self._state == LoopState.PAUSED:
            self.set_state(LoopState.RUNNING, "resume")
            logger.info(f"[{self.name}] Resumed")

    def get_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            'name': self.name,
            'state': self._state.value,
            'metrics': self._metrics.to_dict(),
            'fallback_used': self.is_fallback,
            'fallback_message': (
                self._fallback_result.message
                if self._fallback_result
                else ""
            ),
        }

    def __repr__(self) -> str:
        return (
            f"AgentLoopController(name={self.name!r}, "
            f"state={self._state.value}, "
            f"attempts={self._metrics.attempts}, "
            f"elapsed={self._metrics.total_elapsed:.2f}s)"
        )

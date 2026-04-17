"""
Blackboard Controller - 黑板控制器

协调多个知识源（Agent）的执行：
- 按优先级调度知识源
- 监控执行状态
- 处理人类介入
- 支持并行和串行执行

使用示例：
    >>> from futureQuant.agent.blackboard import Blackboard, BlackboardController
    >>> from futureQuant.agent.blackboard import KnowledgeSourceAdapter
    >>> 
    >>> # 创建黑板和控制器
    >>> bb = Blackboard()
    >>> controller = BlackboardController(blackboard=bb)
    >>> 
    >>> # 注册 Agent
    >>> controller.register(KnowledgeSourceAdapter.wrap(data_collector, input_keys=[], output_key="price_data"))
    >>> controller.register(KnowledgeSourceAdapter.wrap(factor_mining, input_keys=["price_data"], output_key="factors"))
    >>> controller.register(KnowledgeSourceAdapter.wrap(backtest, input_keys=["factors"], output_key="backtest_result"))
    >>> 
    >>> # 执行
    >>> result = await controller.execute()
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .blackboard import Blackboard, BlackboardState
    from .knowledge_source import KnowledgeSource, KnowledgeSourceResult
    from .human_intervention import InterventionRequest

from .blackboard import Blackboard, BlackboardState
from .knowledge_source import KnowledgeSource, KnowledgeSourceResult, KnowledgeSourceAdapter
from .human_intervention import HumanInterventionPoint, InterventionRequest

from ...core.logger import get_logger

logger = get_logger('agent.blackboard.controller')


class ExecutionMode(Enum):
    """执行模式"""
    SEQUENTIAL = "sequential"       # 串行（按优先级依次执行）
    PARALLEL = "parallel"           # 并行（无依赖的并行执行）
    ADAPTIVE = "adaptive"           # 自适应（根据依赖关系自动选择）


@dataclass
class ControllerResult:
    """
    控制器执行结果
    
    Attributes:
        success: 是否成功
        state: 最终状态
        n_executed: 执行的知识源数量
        n_skipped: 跳过的知识源数量
        n_failed: 失败的知识源数量
        results: 各知识源的执行结果
        elapsed_seconds: 总耗时
        snapshot: 最终黑板快照
    """
    success: bool
    state: BlackboardState
    n_executed: int = 0
    n_skipped: int = 0
    n_failed: int = 0
    results: Dict[str, KnowledgeSourceResult] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    snapshot: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'state': self.state.value,
            'n_executed': self.n_executed,
            'n_skipped': self.n_skipped,
            'n_failed': self.n_failed,
            'results': {k: v.to_dict() for k, v in self.results.items()},
            'elapsed_seconds': self.elapsed_seconds,
        }


class BlackboardController:
    """
    黑板控制器
    
    协调多个知识源的执行，实现黑板模式的核心控制逻辑：
    
    1. 注册知识源（Agent）
    2. 按优先级或依赖关系调度执行
    3. 监控执行状态
    4. 处理人类介入请求
    5. 生成执行报告
    
    使用示例：
        >>> controller = BlackboardController(
        ...     blackboard=Blackboard(),
        ...     mode=ExecutionMode.ADAPTIVE,
        ...     human_intervention_enabled=True,
        ... )
        >>> 
        >>> # 注册 Agent
        >>> controller.register(agent1, priority=10)
        >>> controller.register(agent2, priority=5)
        >>> 
        >>> # 执行
        >>> result = await controller.execute()
    """
    
    def __init__(
        self,
        blackboard: Optional[Blackboard] = None,
        mode: ExecutionMode = ExecutionMode.ADAPTIVE,
        max_workers: int = 4,
        human_intervention_enabled: bool = True,
        auto_save: bool = False,
        save_dir: Optional[Path] = None,
        on_progress: Optional[Callable[[str, str, float], None]] = None,
    ):
        """
        初始化控制器
        
        Args:
            blackboard: 黑板实例（None 则创建新的）
            mode: 执行模式
            max_workers: 最大并行工作线程数
            human_intervention_enabled: 是否启用人类介入
            auto_save: 是否自动保存黑板
            save_dir: 保存目录
            on_progress: 进度回调 (agent_name, status, elapsed)
        """
        self.blackboard = blackboard or Blackboard()
        self.mode = mode
        self.max_workers = max_workers
        self.auto_save = auto_save
        self.save_dir = Path(save_dir) if save_dir else None
        self.on_progress = on_progress
        
        # 知识源注册表
        self._sources: Dict[str, KnowledgeSource] = {}
        self._dependencies: Dict[str, List[str]] = {}  # agent -> [依赖的 agent]
        
        # 人类介入
        self.human_intervention = HumanInterventionPoint(
            enabled=human_intervention_enabled,
            on_request=self._on_intervention_request,
        )
        
        # 执行状态
        self._is_running = False
        self._current_agent: Optional[str] = None
        
        logger.info(
            f"BlackboardController initialized: "
            f"mode={mode.value}, max_workers={max_workers}, "
            f"human_intervention={human_intervention_enabled}"
        )
    
    # ---- 注册知识源 ----
    
    def register(
        self,
        source: KnowledgeSource,
        dependencies: Optional[List[str]] = None,
    ) -> str:
        """
        注册知识源
        
        Args:
            source: 知识源实例
            dependencies: 依赖的其他知识源名称列表
        
        Returns:
            知识源名称
        """
        name = source.name
        self._sources[name] = source
        self._dependencies[name] = dependencies or []
        
        logger.info(
            f"[Controller] Registered: {name}, "
            f"priority={source.priority}, dependencies={self._dependencies[name]}"
        )
        
        return name
    
    def register_agent(
        self,
        agent: Any,
        name: Optional[str] = None,
        priority: int = 0,
        input_keys: Optional[List[str]] = None,
        output_key: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
    ) -> str:
        """
        注册 Agent（自动包装为 KnowledgeSource）
        
        Args:
            agent: Agent 实例
            name: 名称
            priority: 优先级
            input_keys: 输入数据键
            output_key: 输出数据键
            dependencies: 依赖
        
        Returns:
            知识源名称
        """
        source = KnowledgeSourceAdapter.wrap(
            agent=agent,
            name=name,
            priority=priority,
            input_keys=input_keys,
            output_key=output_key,
        )
        return self.register(source, dependencies)
    
    def unregister(self, name: str) -> bool:
        """注销知识源"""
        if name in self._sources:
            del self._sources[name]
            self._dependencies.pop(name, None)
            logger.info(f"[Controller] Unregistered: {name}")
            return True
        return False
    
    def get_registered_sources(self) -> List[str]:
        """获取已注册的知识源列表"""
        return list(self._sources.keys())
    
    # ---- 执行 ----
    
    def execute(self) -> ControllerResult:
        """
        同步执行所有知识源
        
        如果 Blackboard 上存在 execution_plan，则优先按 Plan 调度执行。
        否则按默认的注册顺序/依赖关系执行。
        
        Returns:
            ControllerResult
        """
        plan = self.blackboard.read("execution_plan", default=None)
        if plan and isinstance(plan, dict) and plan.get("steps"):
            return self.execute_plan(plan)
        return self._execute_sync()
    
    def execute_plan(self, plan: Dict[str, Any]) -> ControllerResult:
        """
        按 ExecutionPlan 调度执行
        
        Args:
            plan: ExecutionPlan 字典，包含 goal 和 steps
        
        Returns:
            ControllerResult
        """
        import time
        from collections import deque
        
        start_time = time.time()
        self._is_running = True
        self.blackboard.set_state(BlackboardState.RUNNING)
        
        result = ControllerResult(
            success=False,
            state=BlackboardState.RUNNING,
        )
        
        steps = plan.get("steps", [])
        goal = plan.get("goal", "")
        logger.info(f"[Controller] Executing plan: {goal} with {len(steps)} steps")
        
        # 构建 step_id -> source 映射
        step_map = {s["step_id"]: s for s in steps}
        completed = set()
        failed_steps = []
        
        # 拓扑排序执行
        remaining = set(step_map.keys())
        
        try:
            while remaining:
                # 找出当前可执行的 steps（依赖已满足）
                ready = [
                    sid for sid in remaining
                    if all(d in completed for d in step_map[sid].get("depends_on", []))
                ]
                
                if not ready:
                    # 存在循环依赖或无法执行的 step
                    logger.error(f"[Controller] Plan deadlock: remaining={remaining}, completed={completed}")
                    failed_steps.extend(list(remaining))
                    break
                
                for sid in ready:
                    step = step_map[sid]
                    agent_name = step.get("agent", "")
                    task_desc = step.get("task", "")
                    remaining.remove(sid)
                    
                    self._current_agent = agent_name
                    logger.info(f"[Controller] Plan step {sid}: {agent_name} -> {task_desc}")
                    
                    # 写入进度
                    self.blackboard.write(
                        "plan_progress",
                        {
                            "current_step": sid,
                            "agent": agent_name,
                            "status": "running",
                            "completed": list(completed),
                        },
                        agent="blackboard_controller",
                    )
                    
                    source = self._sources.get(agent_name)
                    if source is None:
                        logger.warning(f"[Controller] Agent '{agent_name}' not registered, skipping step {sid}")
                        result.n_skipped += 1
                        failed_steps.append(sid)
                        continue
                    
                    # 检查人类介入
                    while self.blackboard.has_pending_interventions():
                        logger.info("[Controller] Waiting for human intervention...")
                        time.sleep(1)
                        if self._check_intervention_timeout():
                            break
                    
                    # 将 step 上下文写入黑板（inputs 映射）
                    for out_key in step.get("outputs", []):
                        # 预占 outputs 位置，初始化为 pending
                        self.blackboard.write(
                            out_key,
                            {"status": "pending", "step_id": sid},
                            agent="blackboard_controller",
                        )
                    
                    # 观察并执行
                    if not source.observe(self.blackboard):
                        result.n_skipped += 1
                        result.results[f"{agent_name}_step{sid}"] = KnowledgeSourceResult(
                            source_name=agent_name,
                            state=source.state,
                        )
                        failed_steps.append(sid)
                        continue
                    
                    ks_result = source.execute(self.blackboard)
                    result.results[f"{agent_name}_step{sid}"] = ks_result
                    
                    if ks_result.state.value == 'success':
                        result.n_executed += 1
                        completed.add(sid)
                    else:
                        result.n_failed += 1
                        failed_steps.append(sid)
                    
                    if self.on_progress:
                        self.on_progress(agent_name, ks_result.state.value, ks_result.elapsed_seconds)
            
            result.success = result.n_failed == 0 and len(failed_steps) == 0
            result.state = BlackboardState.SUCCESS if result.success else BlackboardState.FAILED
            result.elapsed_seconds = time.time() - start_time
            result.snapshot = self.blackboard.snapshot()
            self.blackboard.set_state(result.state)
            
            # 最终进度
            self.blackboard.write(
                "plan_progress",
                {
                    "status": "completed" if result.success else "failed",
                    "completed_steps": list(completed),
                    "failed_steps": failed_steps,
                },
                agent="blackboard_controller",
            )
            
        except Exception as e:
            result.state = BlackboardState.FAILED
            result.elapsed_seconds = time.time() - start_time
            logger.error(f"[Controller] Plan execution failed: {e}")
        
        finally:
            self._is_running = False
            self._current_agent = None
            if self.auto_save and self.save_dir:
                self._save_result(result)
        
        logger.info(
            f"[Controller] Plan execution complete: "
            f"success={result.success}, "
            f"executed={result.n_executed}, "
            f"skipped={result.n_skipped}, "
            f"failed={result.n_failed}"
        )
        
        return result
    
    async def execute_async(self) -> ControllerResult:
        """
        异步执行所有知识源
        
        Returns:
            ControllerResult
        """
        return await self._execute_async()
    
    def _execute_sync(self) -> ControllerResult:
        """同步执行实现"""
        start_time = time.time()
        self._is_running = True
        self.blackboard.set_state(BlackboardState.RUNNING)
        
        result = ControllerResult(
            success=False,
            state=BlackboardState.RUNNING,
        )
        
        try:
            # 获取执行顺序
            execution_order = self._get_execution_order()
            logger.info(f"[Controller] Execution order: {execution_order}")
            
            # 按顺序执行
            for name in execution_order:
                if not self._is_running:
                    break
                
                # 检查人类介入
                while self.blackboard.has_pending_interventions():
                    logger.info("[Controller] Waiting for human intervention...")
                    time.sleep(1)
                    # 检查超时
                    if self._check_intervention_timeout():
                        break
                
                source = self._sources[name]
                self._current_agent = name
                
                # 观察黑板
                if not source.observe(self.blackboard):
                    result.n_skipped += 1
                    result.results[name] = KnowledgeSourceResult(
                        source_name=name,
                        state=source.state,
                    )
                    continue
                
                # 执行
                ks_result = source.execute(self.blackboard)
                result.results[name] = ks_result
                
                if ks_result.state.value == 'success':
                    result.n_executed += 1
                else:
                    result.n_failed += 1
                
                # 进度回调
                if self.on_progress:
                    self.on_progress(name, ks_result.state.value, ks_result.elapsed_seconds)
            
            # 完成
            result.success = result.n_failed == 0
            result.state = BlackboardState.SUCCESS if result.success else BlackboardState.FAILED
            result.elapsed_seconds = time.time() - start_time
            result.snapshot = self.blackboard.snapshot()
            
            self.blackboard.set_state(result.state)
            
        except Exception as e:
            result.state = BlackboardState.FAILED
            result.elapsed_seconds = time.time() - start_time
            logger.error(f"[Controller] Execution failed: {e}")
        
        finally:
            self._is_running = False
            self._current_agent = None
            
            # 自动保存
            if self.auto_save and self.save_dir:
                self._save_result(result)
        
        logger.info(
            f"[Controller] Execution complete: "
            f"success={result.success}, "
            f"executed={result.n_executed}, "
            f"skipped={result.n_skipped}, "
            f"failed={result.n_failed}, "
            f"elapsed={result.elapsed_seconds:.2f}s"
        )
        
        return result
    
    async def _execute_async(self) -> ControllerResult:
        """异步执行实现"""
        start_time = time.time()
        self._is_running = True
        self.blackboard.set_state(BlackboardState.RUNNING)
        
        result = ControllerResult(
            success=False,
            state=BlackboardState.RUNNING,
        )
        
        try:
            # 获取执行顺序
            execution_order = self._get_execution_order()
            logger.info(f"[Controller] Execution order: {execution_order}")
            
            # 根据模式选择执行方式
            if self.mode == ExecutionMode.PARALLEL:
                # 并行执行（无依赖的）
                result = await self._execute_parallel(execution_order, result)
            elif self.mode == ExecutionMode.ADAPTIVE:
                # 自适应执行
                result = await self._execute_adaptive(execution_order, result)
            else:
                # 串行执行
                for name in execution_order:
                    if not self._is_running:
                        break
                    
                    # 检查人类介入
                    while self.blackboard.has_pending_interventions():
                        logger.info("[Controller] Waiting for human intervention...")
                        await asyncio.sleep(1)
                    
                    source = self._sources[name]
                    self._current_agent = name
                    
                    if not source.observe(self.blackboard):
                        result.n_skipped += 1
                        result.results[name] = KnowledgeSourceResult(
                            source_name=name,
                            state=source.state,
                        )
                        continue
                    
                    ks_result = source.execute(self.blackboard)
                    result.results[name] = ks_result
                    
                    if ks_result.state.value == 'success':
                        result.n_executed += 1
                    else:
                        result.n_failed += 1
                    
                    if self.on_progress:
                        self.on_progress(name, ks_result.state.value, ks_result.elapsed_seconds)
            
            result.success = result.n_failed == 0
            result.state = BlackboardState.SUCCESS if result.success else BlackboardState.FAILED
            result.elapsed_seconds = time.time() - start_time
            result.snapshot = self.blackboard.snapshot()
            
            self.blackboard.set_state(result.state)
            
        except Exception as e:
            result.state = BlackboardState.FAILED
            result.elapsed_seconds = time.time() - start_time
            logger.error(f"[Controller] Async execution failed: {e}")
        
        finally:
            self._is_running = False
            self._current_agent = None
            
            if self.auto_save and self.save_dir:
                self._save_result(result)
        
        return result
    
    async def _execute_parallel(
        self,
        execution_order: List[str],
        result: ControllerResult,
    ) -> ControllerResult:
        """并行执行"""
        # 按依赖分组
        groups = self._group_by_dependencies(execution_order)
        
        for group in groups:
            # 同一组内的可以并行执行
            tasks = []
            for name in group:
                source = self._sources[name]
                if source.observe(self.blackboard):
                    tasks.append((name, source))
                else:
                    result.n_skipped += 1
                    result.results[name] = KnowledgeSourceResult(
                        source_name=name,
                        state=source.state,
                    )
            
            # 并行执行
            if tasks:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {
                        executor.submit(s.execute, self.blackboard): n
                        for n, s in tasks
                    }
                    
                    for future in futures:
                        name = futures[future]
                        try:
                            ks_result = future.result()
                            result.results[name] = ks_result
                            if ks_result.state.value == 'success':
                                result.n_executed += 1
                            else:
                                result.n_failed += 1
                        except Exception as e:
                            result.n_failed += 1
                            result.results[name] = KnowledgeSourceResult(
                                source_name=name,
                                state=source.state,
                                errors=[str(e)],
                            )
        
        return result
    
    async def _execute_adaptive(
        self,
        execution_order: List[str],
        result: ControllerResult,
    ) -> ControllerResult:
        """自适应执行（根据依赖自动选择串行或并行）"""
        # 检查是否有依赖
        has_deps = any(self._dependencies.values())
        
        if has_deps:
            # 有依赖，按依赖分组执行
            return await self._execute_parallel(execution_order, result)
        else:
            # 无依赖，串行执行
            for name in execution_order:
                if not self._is_running:
                    break
                
                source = self._sources[name]
                self._current_agent = name
                
                if not source.observe(self.blackboard):
                    result.n_skipped += 1
                    result.results[name] = KnowledgeSourceResult(
                        source_name=name,
                        state=source.state,
                    )
                    continue
                
                ks_result = source.execute(self.blackboard)
                result.results[name] = ks_result
                
                if ks_result.state.value == 'success':
                    result.n_executed += 1
                else:
                    result.n_failed += 1
                
                if self.on_progress:
                    self.on_progress(name, ks_result.state.value, ks_result.elapsed_seconds)
            
            return result
    
    # ---- 调度逻辑 ----
    
    def _get_execution_order(self) -> List[str]:
        """
        获取执行顺序
        
        按优先级和依赖关系排序。
        """
        # 拓扑排序（考虑依赖）
        visited: Set[str] = set()
        order: List[str] = []
        
        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            
            # 先访问依赖
            for dep in self._dependencies.get(name, []):
                if dep in self._sources:
                    visit(dep)
            
            order.append(name)
        
        # 按优先级排序
        sorted_sources = sorted(
            self._sources.items(),
            key=lambda x: -x[1].priority  # 优先级高的先执行
        )
        
        for name, _ in sorted_sources:
            visit(name)
        
        return order
    
    def _group_by_dependencies(self, order: List[str]) -> List[List[str]]:
        """
        按依赖分组
        
        同一组内的可以并行执行。
        """
        groups: List[List[str]] = []
        current_group: List[str] = []
        processed: Set[str] = set()
        
        for name in order:
            deps = self._dependencies.get(name, [])
            
            # 检查依赖是否都已处理
            if all(d in processed for d in deps):
                current_group.append(name)
            else:
                # 开始新组
                if current_group:
                    groups.append(current_group)
                current_group = [name]
            
            processed.add(name)
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    # ---- 人类介入 ----
    
    def _on_intervention_request(self, request: InterventionRequest):
        """介入请求回调"""
        # 写入黑板
        self.blackboard.request_intervention(request)
        
        # 暂停执行
        self.blackboard.set_state(BlackboardState.HUMAN_WAITING)
        
        logger.info(
            f"[Controller] Human intervention requested: "
            f"id={request.request_id}, type={request.intervention_type.value}"
        )
    
    def respond_intervention(
        self,
        request_id: str,
        response: Any,
        approved: bool = True,
        reason: str = "",
    ):
        """
        响应介入请求
        
        Args:
            request_id: 请求 ID
            response: 响应内容
            approved: 是否批准
            reason: 理由
        """
        self.human_intervention.respond(
            request_id=request_id,
            response=response,
            approved=approved,
            reason=reason,
        )
        
        # 写入黑板
        self.blackboard.respond_intervention(request_id, response)
        
        logger.info(
            f"[Controller] Human intervention responded: "
            f"id={request_id}, approved={approved}"
        )
    
    def _check_intervention_timeout(self) -> bool:
        """检查介入请求是否超时"""
        pending = self.human_intervention.get_pending_requests()
        for req in pending:
            if req.is_expired():
                # 使用默认响应
                self.human_intervention.respond(
                    request_id=req.request_id,
                    response=req.default_response,
                    approved=bool(req.default_response),
                    reason="超时，使用默认响应",
                    responder="timeout",
                )
                return True
        return False
    
    # ---- 控制 ----
    
    def pause(self):
        """暂停执行"""
        self._is_running = False
        self.blackboard.set_state(BlackboardState.PAUSED)
        logger.info("[Controller] Paused")
    
    def resume(self):
        """恢复执行"""
        self._is_running = True
        self.blackboard.set_state(BlackboardState.RUNNING)
        logger.info("[Controller] Resumed")
    
    def stop(self):
        """停止执行"""
        self._is_running = False
        self.blackboard.set_state(BlackboardState.FAILED)
        logger.info("[Controller] Stopped")
    
    # ---- 持久化 ----
    
    def _save_result(self, result: ControllerResult):
        """保存执行结果"""
        if not self.save_dir:
            return
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.save_dir / f"controller_result_{timestamp}.json"
        
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"[Controller] Result saved to {filepath}")
    
    # ---- 工具方法 ----
    
    def get_status(self) -> Dict[str, Any]:
        """获取控制器状态"""
        return {
            'is_running': self._is_running,
            'current_agent': self._current_agent,
            'blackboard_state': self.blackboard.state.value,
            'n_sources': len(self._sources),
            'sources': list(self._sources.keys()),
            'pending_interventions': len(self.human_intervention.get_pending_requests()),
        }
    
    def __repr__(self) -> str:
        return (
            f"BlackboardController("
            f"sources={len(self._sources)}, "
            f"mode={self.mode.value}, "
            f"running={self._is_running})"
        )

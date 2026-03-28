"""
å¹¶è¡è®¡ç®å¼ææ¨¡å

æä¾é«æçå¹¶è¡å å­è®¡ç®è½åï¼
- å¤è¿ç¨åå¤çº¿ç¨æ¯æ
- ä»»å¡éåç®¡ç
- è¿åº¦è·è¸ª
- å¼å¸¸å¤çåå®¹é
- æ§è½çæ§
"""

import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """æ§è¡æ¨¡å¼æä¸¾"""
    PROCESS = "process"  # å¤è¿ç¨
    THREAD = "thread"    # å¤çº¿ç¨
    SEQUENTIAL = "sequential"  # ä¸²è¡


@dataclass
class TaskResult:
    """ä»»å¡æ§è¡ç»æ"""
    task_id: str
    factor_name: str
    data: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    success: bool = True

    def __repr__(self) -> str:
        status = "â" if self.success else "â"
        return f"{status} {self.factor_name} ({self.elapsed_seconds:.2f}s)"


@dataclass
class ProgressTracker:
    """è¿åº¦è·è¸ªå¨"""
    total_tasks: int
    completed_tasks: int = 0
    failed_tasks: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def progress_percent(self) -> float:
        """è¿åº¦ç¾åæ¯"""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100
    
    @property
    def elapsed_seconds(self) -> float:
        """å·²èæ¶ï¼ç§ï¼"""
        return time.time() - self.start_time
    
    @property
    def estimated_remaining_seconds(self) -> float:
        """é¢è®¡å©ä½æ¶é´ï¼ç§ï¼"""
        if self.completed_tasks == 0:
            return 0.0
        avg_time_per_task = self.elapsed_seconds / self.completed_tasks
        remaining_tasks = self.total_tasks - self.completed_tasks
        return avg_time_per_task * remaining_tasks
    
    def update(self, success: bool = True) -> None:
        """æ´æ°è¿åº¦"""
        self.completed_tasks += 1
        if not success:
            self.failed_tasks += 1
    
    def __repr__(self) -> str:
        return (
            f"Progress({self.completed_tasks}/{self.total_tasks}, "
            f"{self.progress_percent:.1f}%, "
            f"ETA: {self.estimated_remaining_seconds:.1f}s)"
        )


class ParallelCalculator:
    """
    å¹¶è¡è®¡ç®å¼æ
    
    æ¯æå¤è¿ç¨ãå¤çº¿ç¨åä¸²è¡ä¸ç§æ§è¡æ¨¡å¼ï¼
    æä¾ä»»å¡éåç®¡çãè¿åº¦è·è¸ªåå¼å¸¸å¤çã
    """
    
    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.PROCESS,
        n_jobs: int = -1,
        timeout: Optional[float] = None,
        verbose: int = 0,
    ):
        """
        åå§åå¹¶è¡è®¡ç®å¼æ
        
        Args:
            mode: æ§è¡æ¨¡å¼ï¼PROCESS/THREAD/SEQUENTIALï¼
            n_jobs: å¹¶è¡ä»»å¡æ°ï¼-1 è¡¨ç¤ºä½¿ç¨ææ CPU æ ¸å¿ï¼
            timeout: åä¸ªä»»å¡è¶æ¶æ¶é´ï¼ç§ï¼
            verbose: æ¥å¿è¯¦ç»ç¨åº¦ï¼0-2ï¼
        """
        self.mode = mode
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.verbose = verbose
        self.results: List[TaskResult] = []
        self.progress: Optional[ProgressTracker] = None
        
        logger.info(
            f"ParallelCalculator initialized: mode={mode.value}, "
            f"n_jobs={n_jobs}, timeout={timeout}"
        )
    
    def calculate_factors(
        self,
        factor_functions: Dict[str, Callable],
        data: pd.DataFrame,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        å¹¶è¡è®¡ç®å¤ä¸ªå å­
        
        Args:
            factor_functions: å å­è®¡ç®å½æ°å­å¸ {å å­å: è®¡ç®å½æ°}
            data: è¾å¥æ°æ® DataFrame
            **kwargs: ä¼ éç»è®¡ç®å½æ°çé¢å¤åæ°
        
        Returns:
            å å­è®¡ç®ç»æå­å¸ {å å­å: å å­å¼ DataFrame}
        """
        self.results = []
        self.progress = ProgressTracker(total_tasks=len(factor_functions))
        
        logger.info(
            f"Starting parallel factor calculation: {len(factor_functions)} factors, "
            f"mode={self.mode.value}"
        )
        
        if self.mode == ExecutionMode.SEQUENTIAL:
            return self._calculate_sequential(factor_functions, data, **kwargs)
        elif self.mode == ExecutionMode.THREAD:
            return self._calculate_threaded(factor_functions, data, **kwargs)
        else:  # PROCESS
            return self._calculate_parallel(factor_functions, data, **kwargs)
    
    def _calculate_sequential(
        self,
        factor_functions: Dict[str, Callable],
        data: pd.DataFrame,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """ä¸²è¡è®¡ç®å å­"""
        results = {}
        
        for factor_name, func in factor_functions.items():
            try:
                start_time = time.time()
                result = func(data, **kwargs)
                elapsed = time.time() - start_time
                
                results[factor_name] = result
                self.results.append(TaskResult(
                    task_id=factor_name,
                    factor_name=factor_name,
                    data=result,
                    elapsed_seconds=elapsed,
                    success=True
                ))
                self.progress.update(success=True)
                
                if self.verbose > 0:
                    logger.info(f"â {factor_name}: {elapsed:.2f}s")
                    
            except Exception as e:
                error_msg = f"Error calculating {factor_name}: {str(e)}"
                logger.error(error_msg)
                self.results.append(TaskResult(
                    task_id=factor_name,
                    factor_name=factor_name,
                    error=error_msg,
                    success=False
                ))
                self.progress.update(success=False)
        
        return results
    
    def _calculate_threaded(
        self,
        factor_functions: Dict[str, Callable],
        data: pd.DataFrame,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """å¤çº¿ç¨è®¡ç®å å­"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {
                executor.submit(
                    self._execute_task,
                    factor_name,
                    func,
                    data,
                    **kwargs
                ): factor_name
                for factor_name, func in factor_functions.items()
            }
            
            for future in as_completed(futures, timeout=self.timeout):
                factor_name = futures[future]
                try:
                    result = future.result()
                    if result.success:
                        results[factor_name] = result.data
                    self.results.append(result)
                    self.progress.update(success=result.success)
                    
                    if self.verbose > 0:
                        logger.info(str(result))
                        
                except Exception as e:
                    error_msg = f"Error in threaded execution: {str(e)}"
                    logger.error(error_msg)
                    self.results.append(TaskResult(
                        task_id=factor_name,
                        factor_name=factor_name,
                        error=error_msg,
                        success=False
                    ))
                    self.progress.update(success=False)
        
        return results
    
    def _calculate_parallel(
        self,
        factor_functions: Dict[str, Callable],
        data: pd.DataFrame,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """å¤è¿ç¨è®¡ç®å å­ï¼ä½¿ç¨ joblibï¼"""
        results = {}
        
        try:
            # ä½¿ç¨ joblib ç Parallel è¿è¡å¹¶è¡è®¡ç®
            delayed_tasks = [
                delayed(self._execute_task)(
                    factor_name,
                    func,
                    data,
                    **kwargs
                )
                for factor_name, func in factor_functions.items()
            ]
            
            task_results = Parallel(
                n_jobs=self.n_jobs,
                timeout=self.timeout,
                verbose=self.verbose
            )(delayed_tasks)
            
            for result in task_results:
                if result.success:
                    results[result.factor_name] = result.data
                self.results.append(result)
                self.progress.update(success=result.success)
                
                if self.verbose > 0:
                    logger.info(str(result))
                    
        except Exception as e:
            logger.error(f"Error in parallel execution: {str(e)}")
            # éçº§å°å¤çº¿ç¨
            logger.info("Falling back to threaded execution")
            return self._calculate_threaded(factor_functions, data, **kwargs)
        
        return results
    
    @staticmethod
    def _execute_task(
        factor_name: str,
        func: Callable,
        data: pd.DataFrame,
        **kwargs
    ) -> TaskResult:
        """æ§è¡åä¸ªä»»å¡"""
        try:
            start_time = time.time()
            result = func(data, **kwargs)
            elapsed = time.time() - start_time
            
            return TaskResult(
                task_id=factor_name,
                factor_name=factor_name,
                data=result,
                elapsed_seconds=elapsed,
                success=True
            )
        except Exception as e:
            error_msg = f"Error calculating {factor_name}: {str(e)}"
            return TaskResult(
                task_id=factor_name,
                factor_name=factor_name,
                error=error_msg,
                success=False
            )
    
    def get_summary(self) -> Dict[str, Any]:
        """è·åæ§è¡æè¦"""
        if not self.progress:
            return {}
        
        successful = sum(1 for r in self.results if r.success)
        failed = sum(1 for r in self.results if not r.success)
        total_time = sum(r.elapsed_seconds for r in self.results)
        
        return {
            "total_tasks": self.progress.total_tasks,
            "successful_tasks": successful,
            "failed_tasks": failed,
            "total_elapsed_seconds": self.progress.elapsed_seconds,
            "computation_time_seconds": total_time,
            "speedup": total_time / self.progress.elapsed_seconds if self.progress.elapsed_seconds > 0 else 1.0,
            "mode": self.mode.value,
        }
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """è·åç»æ DataFrame"""
        data = []
        for result in self.results:
            data.append({
                "factor_name": result.factor_name,
                "success": result.success,
                "elapsed_seconds": result.elapsed_seconds,
                "error": result.error or "N/A",
            })
        return pd.DataFrame(data)


class BatchCalculator:
    """
    æ¹éè®¡ç®å¨
    
    æ¯æå°å¤§éå å­åæ¹è®¡ç®ï¼é¿ååå­æº¢åºã
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        mode: ExecutionMode = ExecutionMode.PROCESS,
        n_jobs: int = -1,
    ):
        """
        åå§åæ¹éè®¡ç®å¨
        
        Args:
            batch_size: æ¯æ¹è®¡ç®çå å­æ°
            mode: æ§è¡æ¨¡å¼
            n_jobs: å¹¶è¡ä»»å¡æ°
        """
        self.batch_size = batch_size
        self.calculator = ParallelCalculator(mode=mode, n_jobs=n_jobs)
        logger.info(f"BatchCalculator initialized: batch_size={batch_size}")
    
    def calculate_factors_batched(
        self,
        factor_functions: Dict[str, Callable],
        data: pd.DataFrame,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        åæ¹è®¡ç®å å­
        
        Args:
            factor_functions: å å­è®¡ç®å½æ°å­å¸
            data: è¾å¥æ°æ®
            **kwargs: é¢å¤åæ°
        
        Returns:
            ææå å­çè®¡ç®ç»æ
        """
        all_results = {}
        factor_items = list(factor_functions.items())
        
        for i in range(0, len(factor_items), self.batch_size):
            batch = dict(factor_items[i:i + self.batch_size])
            logger.info(
                f"Processing batch {i // self.batch_size + 1}: "
                f"{len(batch)} factors"
            )
            
            batch_results = self.calculator.calculate_factors(batch, data, **kwargs)
            all_results.update(batch_results)
        
        return all_results


def create_calculator(
    mode: str = "process",
    n_jobs: int = -1,
    batch_size: Optional[int] = None,
) -> Any:
    """
    å·¥åå½æ°ï¼åå»ºè®¡ç®å¨å®ä¾
    
    Args:
        mode: æ§è¡æ¨¡å¼ ("process", "thread", "sequential")
        n_jobs: å¹¶è¡ä»»å¡æ°
        batch_size: æ¹éå¤§å°ï¼å¦ææå®åè¿å BatchCalculatorï¼
    
    Returns:
        è®¡ç®å¨å®ä¾
    """
    execution_mode = ExecutionMode[mode.upper()]
    
    if batch_size:
        return BatchCalculator(
            batch_size=batch_size,
            mode=execution_mode,
            n_jobs=n_jobs
        )
    else:
        return ParallelCalculator(
            mode=execution_mode,
            n_jobs=n_jobs
        )

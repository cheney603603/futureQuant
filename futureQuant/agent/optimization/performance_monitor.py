"""
æ§è½çæ§å¨æ¨¡å

æä¾æ§è½çæ§è½åï¼
- æ§è½ææ æ¶é
- æ§è½æ¥åçæ
- æ§è½åè­¦
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """æ§è½ææ """
    name: str
    value: float
    unit: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def __repr__(self) -> str:
        return f"{self.name}: {self.value:.2f} {self.unit}"


class PerformanceMonitor:
    """
    æ§è½çæ§å¨
    
    æ¶éååæç³»ç»æ§è½ææ ã
    """
    
    def __init__(self, name: str = "default"):
        """
        åå§åæ§è½çæ§å¨
        
        Args:
            name: çæ§å¨åç§°
        """
        self.name = name
        self.metrics: List[PerformanceMetric] = []
        self.start_time = time.time()
        logger.info(f"PerformanceMonitor initialized: {name}")
    
    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
    ) -> None:
        """
        è®°å½æ§è½ææ 
        
        Args:
            name: ææ åç§°
            value: ææ å¼
            unit: åä½
        """
        metric = PerformanceMetric(name=name, value=value, unit=unit)
        self.metrics.append(metric)
        logger.debug(f"Recorded metric: {metric}")
    
    def measure_time(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> tuple:
        """
        æµéå½æ°æ§è¡æ¶é´
        
        Args:
            func: è¦æµéçå½æ°
            *args: å½æ°çä½ç½®åæ°
            **kwargs: å½æ°çå³é®å­åæ°
        
        Returns:
            (æ§è¡ç»æ, æ§è¡æ¶é´(ç§))
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        self.record_metric(
            name=f"{func.__name__}_time",
            value=elapsed,
            unit="seconds"
        )
        
        return result, elapsed
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """è·åææ  DataFrame"""
        data = []
        for metric in self.metrics:
            data.append({
                "name": metric.name,
                "value": metric.value,
                "unit": metric.unit,
                "timestamp": datetime.fromtimestamp(metric.timestamp),
            })
        return pd.DataFrame(data)
    
    def get_summary(self) -> Dict[str, Any]:
        """è·åçæ§æè¦"""
        if not self.metrics:
            return {}
        
        df = self.get_metrics_dataframe()
        
        summary = {
            "monitor_name": self.name,
            "total_metrics": len(self.metrics),
            "elapsed_seconds": time.time() - self.start_time,
            "metrics_by_name": {},
        }
        
        for name in df["name"].unique():
            name_metrics = df[df["name"] == name]["value"].tolist()
            summary["metrics_by_name"][name] = {
                "count": len(name_metrics),
                "min": min(name_metrics),
                "max": max(name_metrics),
                "avg": sum(name_metrics) / len(name_metrics),
            }
        
        return summary
    
    def __repr__(self) -> str:
        return f"PerformanceMonitor({self.name}, metrics={len(self.metrics)})"


class PerformanceReporter:
    """
    æ§è½æ¥åçæå¨
    
    çæè¯¦ç»çæ§è½æ¥åã
    """
    
    def __init__(self, output_dir: str = "./reports"):
        """
        åå§åæ§è½æ¥åçæå¨
        
        Args:
            output_dir: æ¥åè¾åºç®å½
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PerformanceReporter initialized: output_dir={output_dir}")
    
    def generate_report(
        self,
        monitor: PerformanceMonitor,
        report_name: Optional[str] = None,
    ) -> Path:
        """
        çææ§è½æ¥å
        
        Args:
            monitor: æ§è½çæ§å¨å®ä¾
            report_name: æ¥ååç§°
        
        Returns:
            æ¥åæä»¶è·¯å¾
        """
        if report_name is None:
            report_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # çæ JSON æ¥å
        summary = monitor.get_summary()
        json_path = self.output_dir / f"{report_name}.json"
        
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Generated JSON report: {json_path}")
        
        # çæ CSV æ¥å
        df = monitor.get_metrics_dataframe()
        csv_path = self.output_dir / f"{report_name}.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Generated CSV report: {csv_path}")
        
        # çæææ¬æ¥å
        text_path = self.output_dir / f"{report_name}.txt"
        self._generate_text_report(monitor, text_path)
        
        logger.info(f"Generated text report: {text_path}")
        
        return json_path
    
    def _generate_text_report(
        self,
        monitor: PerformanceMonitor,
        output_path: Path,
    ) -> None:
        """çæææ¬æ ¼å¼çæ¥å"""
        summary = monitor.get_summary()
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Performance Report: {summary['monitor_name']}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total Metrics: {summary['total_metrics']}\n")
            f.write(f"Elapsed Time: {summary['elapsed_seconds']:.2f}s\n\n")
            
            f.write("Metrics Summary:\n")
            f.write("-" * 80 + "\n")
            
            for name, stats in summary['metrics_by_name'].items():
                f.write(f"\n{name}:\n")
                f.write(f"  Count: {stats['count']}\n")
                f.write(f"  Min:   {stats['min']:.4f}\n")
                f.write(f"  Max:   {stats['max']:.4f}\n")
                f.write(f"  Avg:   {stats['avg']:.4f}\n")
    
    def compare_reports(
        self,
        report1_path: Path,
        report2_path: Path,
    ) -> Dict[str, Any]:
        """
        æ¯è¾ä¸¤ä¸ªæ§è½æ¥å
        
        Args:
            report1_path: ç¬¬ä¸ä¸ªæ¥åè·¯å¾
            report2_path: ç¬¬äºä¸ªæ¥åè·¯å¾
        
        Returns:
            æ¯è¾ç»æ
        """
        with open(report1_path, 'r') as f:
            report1 = json.load(f)
        
        with open(report2_path, 'r') as f:
            report2 = json.load(f)
        
        comparison = {
            "report1": report1_path.name,
            "report2": report2_path.name,
            "metrics_comparison": {},
        }
        
        for name in report1['metrics_by_name']:
            if name in report2['metrics_by_name']:
                m1 = report1['metrics_by_name'][name]
                m2 = report2['metrics_by_name'][name]
                
                improvement = ((m1['avg'] - m2['avg']) / m1['avg'] * 100) if m1['avg'] > 0 else 0
                
                comparison['metrics_comparison'][name] = {
                    "report1_avg": m1['avg'],
                    "report2_avg": m2['avg'],
                    "improvement_percent": improvement,
                }
        
        return comparison


class PerformanceAlert:
    """
    æ§è½åè­¦
    
    çæ§æ§è½ææ å¹¶å¨è¶è¿éå¼æ¶ååºåè­¦ã
    """
    
    def __init__(self):
        """åå§åæ§è½åè­¦"""
        self.thresholds: Dict[str, float] = {}
        self.alerts: List[Dict[str, Any]] = []
        logger.info("PerformanceAlert initialized")
    
    def set_threshold(self, metric_name: str, threshold: float) -> None:
        """
        è®¾ç½®æ§è½ææ éå¼
        
        Args:
            metric_name: ææ åç§°
            threshold: éå¼
        """
        self.thresholds[metric_name] = threshold
        logger.info(f"Set threshold for {metric_name}: {threshold}")
    
    def check_metric(
        self,
        metric_name: str,
        value: float,
    ) -> bool:
        """
        æ£æ¥æ§è½ææ 
        
        Args:
            metric_name: ææ åç§°
            value: ææ å¼
        
        Returns:
            æ¯å¦è¶è¿éå¼
        """
        if metric_name not in self.thresholds:
            return False
        
        threshold = self.thresholds[metric_name]
        
        if value > threshold:
            alert = {
                "metric_name": metric_name,
                "value": value,
                "threshold": threshold,
                "timestamp": datetime.now().isoformat(),
                "severity": "warning" if value < threshold * 1.5 else "critical",
            }
            self.alerts.append(alert)
            
            logger.warning(
                f"Performance alert: {metric_name}={value:.2f} > {threshold:.2f}"
            )
            
            return True
        
        return False
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """è·åææåè­¦"""
        return self.alerts
    
    def clear_alerts(self) -> None:
        """æ¸ç©ºåè­¦"""
        self.alerts.clear()
    
    def __repr__(self) -> str:
        return f"PerformanceAlert(thresholds={len(self.thresholds)}, alerts={len(self.alerts)})"


class PerformanceBenchmark:
    """
    æ§è½åºåæµè¯
    
    ç¨äºæ§è½åºåæµè¯åå¯¹æ¯ã
    """
    
    def __init__(self, name: str = "benchmark"):
        """
        åå§åæ§è½åºåæµè¯
        
        Args:
            name: åºåæµè¯åç§°
        """
        self.name = name
        self.results: Dict[str, List[float]] = {}
        logger.info(f"PerformanceBenchmark initialized: {name}")
    
    def run_benchmark(
        self,
        func: Callable,
        iterations: int = 10,
        *args,
        **kwargs
    ) -> Dict[str, float]:
        """
        è¿è¡åºåæµè¯
        
        Args:
            func: è¦æµè¯çå½æ°
            iterations: è¿­ä»£æ¬¡æ°
            *args: å½æ°çä½ç½®åæ°
            **kwargs: å½æ°çå³é®å­åæ°
        
        Returns:
            åºåæµè¯ç»æ
        """
        times = []
        
        for i in range(iterations):
            start_time = time.time()
            func(*args, **kwargs)
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        func_name = func.__name__
        self.results[func_name] = times
        
        result = {
            "function": func_name,
            "iterations": iterations,
            "min_time": min(times),
            "max_time": max(times),
            "avg_time": sum(times) / len(times),
            "total_time": sum(times),
        }
        
        logger.info(f"Benchmark {func_name}: avg={result['avg_time']:.4f}s")
        
        return result
    
    def compare_functions(
        self,
        func1: Callable,
        func2: Callable,
        iterations: int = 10,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        æ¯è¾ä¸¤ä¸ªå½æ°çæ§è½
        
        Args:
            func1: ç¬¬ä¸ä¸ªå½æ°
            func2: ç¬¬äºä¸ªå½æ°
            iterations: è¿­ä»£æ¬¡æ°
            *args: å½æ°çä½ç½®åæ°
            **kwargs: å½æ°çå³é®å­åæ°
        
        Returns:
            æ¯è¾ç»æ
        """
        result1 = self.run_benchmark(func1, iterations, *args, **kwargs)
        result2 = self.run_benchmark(func2, iterations, *args, **kwargs)
        
        speedup = result1['avg_time'] / result2['avg_time']
        improvement = (1 - result2['avg_time'] / result1['avg_time']) * 100
        
        return {
            "func1": result1,
            "func2": result2,
            "speedup": speedup,
            "improvement_percent": improvement,
        }
    
    def __repr__(self) -> str:
        return f"PerformanceBenchmark({self.name}, results={len(self.results)})"

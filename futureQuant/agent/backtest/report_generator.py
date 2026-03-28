"""
åæµæ¥åçæå¨æ¨¡å

çæå¤ç§æ ¼å¼çåæµæ¥åï¼ææ¬ãHTMLãJSONï¼ã
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import pandas as pd
import numpy as np
from pathlib import Path

from ...core.logger import get_logger

logger = get_logger('agent.backtest.report_generator')


class BacktestReportGenerator:
    """
    åæµæ¥åçæå¨
    
    æ¯æçæææ¬ãHTMLãJSON ä¸ç§æ ¼å¼çåæµæ¥åã
    """
    
    def __init__(self):
        """åå§åæ¥åçæå¨"""
        self.logger = logger
    
    def generate(
        self,
        backtest_result: Dict[str, Any],
        format: str = 'text',
        output_path: Optional[str] = None
    ) -> str:
        """
        çæåæµæ¥å
        
        Args:
            backtest_result: åæµç»æå­å¸
            format: æ¥åæ ¼å¼ ('text', 'html', 'json')
            output_path: è¾åºæä»¶è·¯å¾ï¼å¯éï¼
            
        Returns:
            æ¥ååå®¹å­ç¬¦ä¸²
        """
        if format == 'text':
            report = self._generate_text_report(backtest_result)
        elif format == 'html':
            report = self._generate_html_report(backtest_result)
        elif format == 'json':
            report = self._generate_json_report(backtest_result)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # ä¿å­å°æä»¶
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"Report saved to {output_path}")
        
        return report
    
    def _generate_text_report(self, result: Dict[str, Any]) -> str:
        """çæææ¬æ ¼å¼æ¥å"""
        lines = []
        lines.append("=" * 80)
        lines.append("æè´§éåç­ç¥åæµæ¥å")
        lines.append("=" * 80)
        lines.append("")
        
        # åºæ¬ä¿¡æ¯
        lines.append("ãåæµéç½®ã")
        lines.append(f"åæµæ¨¡å¼: {result.get('mode', 'unknown')}")
        lines.append(f"åå§èµé: Â¥{result.get('initial_capital', 0):,.2f}")
        lines.append(f"ææ«æç: Â¥{result.get('final_equity', 0):,.2f}")
        lines.append("")
        
        # æ¶çææ 
        lines.append("ãæ¶çææ ã")
        total_return = result.get('total_return', 0)
        annual_return = result.get('annual_return', 0)
        lines.append(f"æ»æ¶çç: {total_return*100:.2f}%")
        lines.append(f"å¹´åæ¶çç: {annual_return*100:.2f}%")
        lines.append("")
        
        # é£é©ææ 
        lines.append("ãé£é©ææ ã")
        volatility = result.get('volatility', 0)
        max_drawdown = result.get('max_drawdown', 0)
        lines.append(f"å¹´åæ³¢å¨ç: {volatility*100:.2f}%")
        lines.append(f"æå¤§åæ¤: {max_drawdown*100:.2f}%")
        lines.append("")
        
        # é£é©è°æ´æ¶ç
        lines.append("ãé£é©è°æ´æ¶çã")
        sharpe = result.get('sharpe_ratio', 0)
        sortino = result.get('sortino_ratio', 0)
        calmar = result.get('calmar_ratio', 0)
        lines.append(f"å¤æ®æ¯ç: {sharpe:.3f}")
        lines.append(f"ç´¢æè¯ºæ¯ç: {sortino:.3f}")
        lines.append(f"å¡çæ¯ç: {calmar:.3f}")
        lines.append("")
        
        # äº¤æç»è®¡
        lines.append("ãäº¤æç»è®¡ã")
        total_trades = result.get('total_trades', 0)
        win_rate = result.get('win_rate', 0)
        profit_factor = result.get('profit_factor', 0)
        lines.append(f"æ»äº¤ææ¬¡æ°: {total_trades}")
        lines.append(f"èç: {win_rate*100:.2f}%")
        lines.append(f"çäºæ¯: {profit_factor:.3f}")
        lines.append("")
        
        lines.append("=" * 80)
        lines.append(f"çææ¶é´: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _generate_html_report(self, result: Dict[str, Any]) -> str:
        """çæ HTML æ ¼å¼æ¥å"""
        html_parts = []
        
        html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>åæµæ¥å</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 20px; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #007bff; color: white; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .metric { display: inline-block; width: 45%; margin: 10px 2.5%; }
        .positive { color: green; }
        .negative { color: red; }
    </style>
</head>
<body>
    <h1>æè´§éåç­ç¥åæµæ¥å</h1>
""")
        
        # æ¶çææ 
        total_return = result.get('total_return', 0)
        annual_return = result.get('annual_return', 0)
        
        html_parts.append(f"""
    <h2>æ¶çææ </h2>
    <div class="metric">
        <strong>æ»æ¶çç:</strong> <span class="{'positive' if total_return > 0 else 'negative'}">{total_return*100:.2f}%</span>
    </div>
    <div class="metric">
        <strong>å¹´åæ¶çç:</strong> <span class="{'positive' if annual_return > 0 else 'negative'}">{annual_return*100:.2f}%</span>
    </div>
""")
        
        # é£é©ææ 
        volatility = result.get('volatility', 0)
        max_drawdown = result.get('max_drawdown', 0)
        
        html_parts.append(f"""
    <h2>é£é©ææ </h2>
    <div class="metric">
        <strong>å¹´åæ³¢å¨ç:</strong> {volatility*100:.2f}%
    </div>
    <div class="metric">
        <strong>æå¤§åæ¤:</strong> <span class="negative">{max_drawdown*100:.2f}%</span>
    </div>
""")
        
        # é£é©è°æ´æ¶ç
        sharpe = result.get('sharpe_ratio', 0)
        sortino = result.get('sortino_ratio', 0)
        calmar = result.get('calmar_ratio', 0)
        
        html_parts.append(f"""
    <h2>é£é©è°æ´æ¶ç</h2>
    <table>
        <tr>
            <th>ææ </th>
            <th>æ°å¼</th>
        </tr>
        <tr>
            <td>å¤æ®æ¯ç</td>
            <td>{sharpe:.3f}</td>
        </tr>
        <tr>
            <td>ç´¢æè¯ºæ¯ç</td>
            <td>{sortino:.3f}</td>
        </tr>
        <tr>
            <td>å¡çæ¯ç</td>
            <td>{calmar:.3f}</td>
        </tr>
    </table>
""")
        
        # äº¤æç»è®¡
        total_trades = result.get('total_trades', 0)
        win_rate = result.get('win_rate', 0)
        profit_factor = result.get('profit_factor', 0)
        
        html_parts.append(f"""
    <h2>äº¤æç»è®¡</h2>
    <table>
        <tr>
            <th>ææ </th>
            <th>æ°å¼</th>
        </tr>
        <tr>
            <td>æ»äº¤ææ¬¡æ°</td>
            <td>{total_trades}</td>
        </tr>
        <tr>
            <td>èç</td>
            <td>{win_rate*100:.2f}%</td>
        </tr>
        <tr>
            <td>çäºæ¯</td>
            <td>{profit_factor:.3f}</td>
        </tr>
    </table>
    
    <p style="margin-top: 30px; color: #999;">
        çææ¶é´: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </p>
</body>
</html>
""")
        
        return "".join(html_parts)
    
    def _generate_json_report(self, result: Dict[str, Any]) -> str:
        """çæ JSON æ ¼å¼æ¥å"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'initial_capital': result.get('initial_capital', 0),
                'final_equity': result.get('final_equity', 0),
                'total_return': result.get('total_return', 0),
                'annual_return': result.get('annual_return', 0),
            },
            'risk_metrics': {
                'volatility': result.get('volatility', 0),
                'max_drawdown': result.get('max_drawdown', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'sortino_ratio': result.get('sortino_ratio', 0),
                'calmar_ratio': result.get('calmar_ratio', 0),
            },
            'trade_stats': {
                'total_trades': result.get('total_trades', 0),
                'win_rate': result.get('win_rate', 0),
                'profit_factor': result.get('profit_factor', 0),
            },
        }
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)

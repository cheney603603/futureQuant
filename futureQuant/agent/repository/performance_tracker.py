"""
å å­æ§è½è¿½è¸ªæ¨¡å

è´è´£å å­æ§è½ççæ§ãè¡°åæ£æµåé¢è­¦ã
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
import numpy as np

from ...core.logger import get_logger

logger = get_logger('agent.repository.performance_tracker')


class PerformanceTracker:
    """
    å å­æ§è½è¿½è¸ªå¨
    
    è´è´£å å­æ§è½ççæ§ãè¡°åæ£æµåé¢è­¦ã
    """
    
    def __init__(self, db_path: str):
        """
        åå§åæ§è½è¿½è¸ªå¨
        
        Args:
            db_path: æ°æ®åºè·¯å¾
        """
        self.db_path = db_path
        self.logger = logger
    
    def track_monthly(
        self,
        factor_id: str,
        period: str,
        start_date: str,
        end_date: str,
        metrics: Dict[str, float]
    ):
        """
        è®°å½æåº¦æ§è½
        
        Args:
            factor_id: å å­ ID
            period: ç»è®¡å¨æ (å¦ '2026-03')
            start_date: å¼å§æ¥æ
            end_date: ç»ææ¥æ
            metrics: æ§è½ææ å­å¸
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO factor_performance
                (factor_id, period, start_date, end_date, ic_mean, icir, ic_win_rate, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                factor_id,
                period,
                start_date,
                end_date,
                metrics.get('ic_mean', 0),
                metrics.get('icir', 0),
                metrics.get('ic_win_rate', 0),
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Tracked monthly performance for {factor_id} ({period})")
            
        except Exception as e:
            self.logger.error(f"Failed to track monthly performance: {e}")
    
    def detect_decay(
        self,
        factor_id: str,
        window: int = 3
    ) -> bool:
        """
        æ£æµå å­è¡°å
        
        è¡°åå¤æ­ï¼è¿ç»­ window ä¸ªæ IC ä¸é
        
        Args:
            factor_id: å å­ ID
            window: æ£æµçªå£ï¼ææ°ï¼
            
        Returns:
            æ¯å¦æ£æµå°è¡°å
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # è·åæè¿çæ§è½è®°å½
            cursor.execute('''
                SELECT ic_mean FROM factor_performance
                WHERE factor_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (factor_id, window + 1))
            
            rows = cursor.fetchall()
            conn.close()
            
            if len(rows) < window:
                return False
            
            # æ£æ¥æ¯å¦è¿ç»­ä¸é
            ics = [row[0] for row in rows]
            ics.reverse()  # ä»æ©å°æ
            
            declining_count = 0
            for i in range(1, len(ics)):
                if ics[i] < ics[i-1]:
                    declining_count += 1
            
            # è¿ç»­ä¸éå¤æ­
            is_decaying = declining_count >= window - 1
            
            if is_decaying:
                self.logger.warning(f"Factor {factor_id} shows decay trend")
            
            return is_decaying
            
        except Exception as e:
            self.logger.error(f"Failed to detect decay: {e}")
            return False
    
    def get_trend(
        self,
        factor_id: str,
        months: int = 12
    ) -> pd.DataFrame:
        """
        è·åå å­æ§è½è¶å¿
        
        Args:
            factor_id: å å­ ID
            months: æ¥è¯¢ææ°
            
        Returns:
            æ§è½è¶å¿ DataFrame
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT period, ic_mean, icir, ic_win_rate, created_at
                FROM factor_performance
                WHERE factor_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=(factor_id, months))
            conn.close()
            
            if df.empty:
                return pd.DataFrame()
            
            # ååæåºï¼ä»æ©å°æï¼
            df = df.iloc[::-1].reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get trend: {e}")
            return pd.DataFrame()
    
    def generate_warning_report(
        self,
        factor_id: str
    ) -> Dict[str, Any]:
        """
        çæå å­é¢è­¦æ¥å
        
        Args:
            factor_id: å å­ ID
            
        Returns:
            é¢è­¦æ¥åå­å¸
        """
        report = {
            'factor_id': factor_id,
            'timestamp': datetime.now().isoformat(),
            'warnings': [],
            'status': 'normal',
        }
        
        try:
            # æ£æµè¡°å
            if self.detect_decay(factor_id, window=3):
                report['warnings'].append('å å­è¡¨ç°è¿ç»­ä¸éï¼å¯è½å­å¨è¡°å')
                report['status'] = 'warning'
            
            # è·åææ°æ§è½
            trend = self.get_trend(factor_id, months=1)
            if not trend.empty:
                latest = trend.iloc[-1]
                
                # IC è¿ä½
                if latest['ic_mean'] < 0.01:
                    report['warnings'].append(f"IC è¿ä½: {latest['ic_mean']:.4f}")
                    report['status'] = 'warning'
                
                # ICIR è¿ä½
                if latest['icir'] < 0.5:
                    report['warnings'].append(f"ICIR è¿ä½: {latest['icir']:.3f}")
                    report['status'] = 'warning'
                
                # èçè¿ä½
                if latest['ic_win_rate'] < 0.45:
                    report['warnings'].append(f"IC èçè¿ä½: {latest['ic_win_rate']*100:.1f}%")
                    report['status'] = 'warning'
            
            if not report['warnings']:
                report['status'] = 'normal'
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate warning report: {e}")
            report['status'] = 'error'
            report['warnings'].append(str(e))
            return report

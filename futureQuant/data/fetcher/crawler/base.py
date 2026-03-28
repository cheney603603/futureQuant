"""
爬虫基类 - 提供通用爬虫功能
"""

import time
import random
from abc import ABC
from typing import Optional, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ....core.logger import get_logger
from ....core.exceptions import FetchError

logger = get_logger('data.fetcher.crawler')


class BaseCrawler(ABC):
    """爬虫基类"""
    
    # User-Agent列表
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ]
    
    def __init__(self, 
                 delay: float = 5.0,
                 timeout: int = 30,
                 retry: int = 3,
                 use_proxy: bool = False):
        """
        初始化爬虫
        
        Args:
            delay: 请求间隔（秒）
            timeout: 请求超时
            retry: 重试次数
            use_proxy: 是否使用代理
        """
        self.delay = delay
        self.timeout = timeout
        self.retry = retry
        self.use_proxy = use_proxy
        
        self.session = self._create_session()
        self._last_request_time = 0
    
    def _create_session(self) -> requests.Session:
        """创建带重试机制的session"""
        session = requests.Session()
        
        # 重试策略
        retry_strategy = Retry(
            total=self.retry,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_headers(self) -> Dict[str, str]:
        """获取随机User-Agent的请求头"""
        return {
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
    
    def _rate_limit(self):
        """请求频率控制"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            sleep_time = self.delay - elapsed + random.uniform(0, 1)
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """
        发送GET请求
        
        Args:
            url: 请求URL
            **kwargs: 额外参数
            
        Returns:
            Response对象
        """
        self._rate_limit()
        
        headers = kwargs.pop('headers', self._get_headers())
        timeout = kwargs.pop('timeout', self.timeout)
        
        try:
            response = self.session.get(
                url,
                headers=headers,
                timeout=timeout,
                **kwargs
            )
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {url}, error: {e}")
            raise FetchError(f"Request failed: {e}")
    
    def post(self, url: str, **kwargs) -> requests.Response:
        """
        发送POST请求
        
        Args:
            url: 请求URL
            **kwargs: 额外参数
            
        Returns:
            Response对象
        """
        self._rate_limit()
        
        headers = kwargs.pop('headers', self._get_headers())
        timeout = kwargs.pop('timeout', self.timeout)
        
        try:
            response = self.session.post(
                url,
                headers=headers,
                timeout=timeout,
                **kwargs
            )
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {url}, error: {e}")
            raise FetchError(f"Request failed: {e}")
    
    def close(self):
        """关闭session"""
        if self.session:
            self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

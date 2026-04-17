"""
Alpha101 Generator - 将数学公式翻译为可执行 pandas 代码

使用 LLM 将 Alpha101 的数学表达式翻译为 pandas 代码，
并在样本数据上验证其正确性。
"""

from typing import Any, Dict, Optional

import pandas as pd

from .alpha101_pool import Alpha101Pool
from ..tools import CodeExecutionTool
from ...core.llm_client import LLMClient
from ...core.logger import get_logger

logger = get_logger("agent.factor_mining.alpha101_generator")


class Alpha101Generator:
    """
    Alpha101 代码生成器
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm = llm_client or LLMClient()
        self._pool = Alpha101Pool()
        self._code_exec = CodeExecutionTool()

    def generate(self, alpha_name: str, sample_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """
        生成单个 Alpha 因子的 pandas 代码并验证

        Returns:
            {
                "name": alpha_name,
                "code": "...",
                "validated": True/False,
                "error": "...",
                "sample_output": ...,
            }
        """
        formula = self._pool.get_formula(alpha_name)
        if not formula:
            logger.warning(f"Alpha101 formula not found: {alpha_name}")
            return None

        helper_doc = self._pool.get_helper_doc()

        prompt = (
            "你是一个量化因子工程师。请将以下 Alpha101 数学公式翻译为一段可执行的 Python/pandas 代码。\n\n"
            f"因子名称: {alpha_name}\n"
            f"数学公式: {formula}\n\n"
            "要求:\n"
            "1. 输入 DataFrame 包含列: open, high, low, close, volume\n"
            "2. 输出必须是一个名为 `result` 的 pandas Series，长度与输入相同\n"
            "3. 只允许使用 pandas, numpy\n"
            "4. 不要输出任何解释，只输出 Python 代码\n\n"
            f"辅助函数参考:\n{helper_doc}\n\n"
            "示例代码结构:\n"
            "```python\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "\n"
            "def compute_alpha(df):\n"
            "    open = df['open']\n"
            "    high = df['high']\n"
            "    low = df['low']\n"
            "    close = df['close']\n"
            "    volume = df['volume']\n"
            "    # 计算逻辑...\n"
            "    return result\n"
            "\n"
            "result = compute_alpha(df)\n"
            "```"
        )

        try:
            resp = self._llm.chat([{"role": "user", "content": prompt}], temperature=0.1)
            code = resp.content or ""
            # 清洗 markdown
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            code = code.strip()
        except Exception as exc:
            logger.error(f"LLM generation failed for {alpha_name}: {exc}")
            return {"name": alpha_name, "code": "", "validated": False, "error": str(exc)}

        # 验证代码
        validated, error, sample_output = self._validate_code(code, sample_data)

        return {
            "name": alpha_name,
            "code": code,
            "validated": validated,
            "error": error,
            "sample_output": sample_output,
        }

    def _validate_code(
        self,
        code: str,
        sample_data: Optional[pd.DataFrame] = None,
    ):
        df = sample_data
        if df is None or df.empty:
            df = self._make_sample_data()

        exec_code = (
            "import pandas as pd\n"
            "import numpy as np\n"
            f"df = pd.DataFrame({df.to_dict('list')})\n"
            f"{code}\n"
        )

        result = self._code_exec.execute(code=exec_code, timeout=15)
        if not result.success:
            return False, result.error, None

        data = result.data
        # 检查 result 是否存在且是序列
        if data is None:
            return False, "Code executed but result is None", None

        # 尝试判断 result 是否是合理的 Series/数组
        if isinstance(data, list):
            if len(data) != len(df):
                return False, f"Result length mismatch: {len(data)} vs {len(df)}", None
            return True, "", data
        if isinstance(data, dict):
            # 可能是 Series.to_dict()
            if len(data) != len(df):
                return False, f"Result length mismatch: {len(data)} vs {len(df)}", None
            return True, "", data
        return True, "", data

    @staticmethod
    def _make_sample_data() -> pd.DataFrame:
        import numpy as np
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n))
        return pd.DataFrame({
            "open": close * (1 + np.random.randn(n) * 0.01),
            "high": close * (1 + abs(np.random.randn(n)) * 0.02),
            "low": close * (1 - abs(np.random.randn(n)) * 0.02),
            "close": close,
            "volume": np.random.randint(1000, 10000, n),
        })

"""
玻璃期货（FG）日频预测分析 - 修复版

修复：
1. 更稳健的数据处理
2. 修正波动率计算
3. 更合理的区间预测

Author: futureQuant Team
Date: 2026-04-19
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from futureQuant.core.logger import get_logger
from futureQuant.data.fetcher.akshare_fetcher import AKShareFetcher

logger = get_logger('fg_prediction')


def fetch_data():
    """获取玻璃期货数据"""
    logger.info("=" * 60)
    logger.info("Step 1: 获取玻璃期货数据")
    logger.info("=" * 60)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    logger.info(f"获取 FG 价格数据 ({start_date} ~ {end_date})...")
    
    try:
        fetcher = AKShareFetcher()
        price_df = fetcher.fetch_daily('FG', start_date, end_date)
        
        if price_df.empty:
            logger.error("价格数据为空")
            return None
        
        # 数据清洗
        price_df = price_df.dropna(subset=['close'])
        price_df = price_df.sort_values('date').reset_index(drop=True)
        
        # 过滤异常值（价格变化超过20%可能是数据错误）
        price_df['ret'] = price_df['close'].pct_change()
        price_df = price_df[(price_df['ret'].abs() < 0.15) | price_df['ret'].isna()]
        price_df = price_df.drop(columns=['ret']).reset_index(drop=True)
        
        logger.info(f"价格数据: {len(price_df)} 条记录")
        logger.info(f"日期范围: {price_df['date'].min()} ~ {price_df['date'].max()}")
        logger.info(f"最新收盘价: {price_df['close'].iloc[-1]:.2f}")
        
        return price_df
        
    except Exception as e:
        logger.error(f"价格数据获取失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_factors(price_df):
    """计算技术因子"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 2: 计算技术因子")
    logger.info("=" * 60)
    
    factors = {}
    close = price_df['close']
    high = price_df['high'] if 'high' in price_df.columns else close
    low = price_df['low'] if 'low' in price_df.columns else close
    
    # 动量因子
    for period in [5, 10, 20]:
        mom = (close - close.shift(period)) / close.shift(period) * 100
        factors[f'momentum_{period}'] = mom
    
    # RSI
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    factors['rsi_14'] = calc_rsi(close, 14)
    
    # ATR
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift(1)),
        'lc': abs(low - close.shift(1))
    }).max(axis=1)
    factors['atr_14'] = tr.rolling(14).mean()
    
    # 波动率
    factors['volatility_10'] = close.pct_change().rolling(10).std() * np.sqrt(252)
    factors['volatility_20'] = close.pct_change().rolling(20).std() * np.sqrt(252)
    
    # 成交量比
    if 'volume' in price_df.columns:
        vol = price_df['volume']
        factors['volume_ratio_5'] = vol / vol.rolling(5).mean()
    
    logger.info(f"因子计算完成: {len(factors)} 个因子")
    for name, series in factors.items():
        valid = series.dropna()
        if len(valid) > 0:
            logger.info(f"  {name}: 最新值 {valid.iloc[-1]:.2f}")
    
    return factors


def analyze_and_predict(price_df, factors):
    """分析并预测"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 3: 分析与预测")
    logger.info("=" * 60)
    
    latest_close = price_df['close'].iloc[-1]
    prev_close = price_df['close'].iloc[-2] if len(price_df) > 1 else latest_close
    
    # 计算日涨跌幅
    daily_change = (latest_close - prev_close) / prev_close * 100
    
    # 20日区间
    price_20 = price_df['close'].iloc[-20:] if len(price_df) >= 20 else price_df['close']
    high_20d = price_20.max()
    low_20d = price_20.min()
    
    # 价格位置
    price_range = high_20d - low_20d
    price_position = (latest_close - low_20d) / price_range * 100 if price_range > 0 else 50
    
    # RSI
    rsi = factors['rsi_14'].dropna().iloc[-1] if len(factors['rsi_14'].dropna()) > 0 else 50
    
    # 波动率
    daily_vol = price_df['close'].pct_change().dropna().std()
    weekly_vol = daily_vol * np.sqrt(5)
    
    # ATR
    atr = factors['atr_14'].dropna().iloc[-1] if len(factors['atr_14'].dropna()) > 0 else latest_close * 0.02
    
    # 动量
    mom_5 = factors['momentum_5'].dropna().iloc[-1] if len(factors['momentum_5'].dropna()) > 0 else 0
    mom_10 = factors['momentum_10'].dropna().iloc[-1] if len(factors['momentum_10'].dropna()) > 0 else 0
    
    logger.info(f"最新收盘价: {latest_close:.2f}")
    logger.info(f"日涨跌幅: {daily_change:.2f}%")
    logger.info(f"20日高点: {high_20d:.2f}")
    logger.info(f"20日低点: {low_20d:.2f}")
    logger.info(f"价格位置: {price_position:.1f}%")
    logger.info(f"RSI(14): {rsi:.1f}")
    logger.info(f"日波动率: {daily_vol*100:.2f}%")
    logger.info(f"周波动率: {weekly_vol*100:.2f}%")
    logger.info(f"ATR(14): {atr:.2f}")
    
    # 预测区间
    weekly_range_points = atr * 2  # 2倍ATR作为周波动
    range_low = latest_close - weekly_range_points
    range_high = latest_close + weekly_range_points
    
    # 概率分析
    bullish_prob = 0.30
    neutral_prob = 0.45
    bearish_prob = 0.25
    
    # 根据技术指标调整
    if rsi > 70:
        bullish_prob -= 0.10
        bearish_prob += 0.10
    elif rsi < 30:
        bullish_prob += 0.10
        bearish_prob -= 0.10
    
    if mom_5 > 2:
        bullish_prob += 0.10
        neutral_prob -= 0.10
    elif mom_5 < -2:
        bearish_prob += 0.10
        neutral_prob -= 0.10
    
    if price_position > 80:
        bearish_prob += 0.05
        bullish_prob -= 0.05
    elif price_position < 20:
        bullish_prob += 0.05
        bearish_prob -= 0.05
    
    # 确保概率在合理范围
    probs = [max(0.1, bullish_prob), max(0.2, neutral_prob), max(0.1, bearish_prob)]
    total = sum(probs)
    bullish_prob, neutral_prob, bearish_prob = [p/total for p in probs]
    
    analysis = {
        'latest_close': latest_close,
        'daily_change': daily_change,
        'high_20d': high_20d,
        'low_20d': low_20d,
        'price_position': price_position,
        'rsi': rsi,
        'daily_vol': daily_vol,
        'weekly_vol': weekly_vol,
        'atr': atr,
        'momentum_5': mom_5,
        'momentum_10': mom_10,
        'range_low': range_low,
        'range_high': range_high,
        'bullish_prob': bullish_prob,
        'neutral_prob': neutral_prob,
        'bearish_prob': bearish_prob,
    }
    
    return analysis


def generate_report(analysis):
    """生成报告"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 4: 生成预测报告")
    logger.info("=" * 60)
    
    report_dir = project_root / 'docs' / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / f'FG_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
    
    # 下周日期
    today = datetime.now()
    days_until_monday = (7 - today.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    next_monday = today + timedelta(days=days_until_monday)
    next_friday = next_monday + timedelta(days=4)
    week_str = f"{next_monday.strftime('%m/%d')} - {next_friday.strftime('%m/%d')}"
    
    # 判断趋势
    if analysis['bullish_prob'] > analysis['bearish_prob'] + 0.15:
        trend = "偏多震荡上行"
        trend_emoji = "📈"
    elif analysis['bearish_prob'] > analysis['bullish_prob'] + 0.15:
        trend = "偏空震荡下行"
        trend_emoji = "📉"
    else:
        trend = "区间震荡"
        trend_emoji = "📊"
    
    # RSI信号
    rsi = analysis['rsi']
    if rsi > 70:
        rsi_signal = "超买，注意回调风险"
    elif rsi < 30:
        rsi_signal = "超卖，存在反弹机会"
    else:
        rsi_signal = "中性"
    
    lines = [
        "# 玻璃期货（FG）周度预测报告",
        "",
        f"**报告日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**预测周期**: 下周 ({week_str})",
        f"**品种**: 玻璃期货 (FG) - 郑商所",
        "",
        "---",
        "",
        "## 一、当前市场状态",
        "",
        "### 价格概况",
        "",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 最新收盘价 | **{analysis['latest_close']:.2f}** 元/吨 |",
        f"| 日涨跌幅 | {analysis['daily_change']:+.2f}% |",
        f"| 20日最高 | {analysis['high_20d']:.2f} |",
        f"| 20日最低 | {analysis['low_20d']:.2f} |",
        f"| 价格位置 | {analysis['price_position']:.0f}% (相对20日区间) |",
        "",
        "### 技术指标",
        "",
        f"| 指标 | 数值 | 信号 |",
        f"|------|------|------|",
        f"| RSI(14) | {analysis['rsi']:.1f} | {rsi_signal} |",
        f"| 动量(5日) | {analysis['momentum_5']:+.2f}% | {'偏多' if analysis['momentum_5'] > 0 else '偏空'} |",
        f"| 动量(10日) | {analysis['momentum_10']:+.2f}% | {'偏多' if analysis['momentum_10'] > 0 else '偏空'} |",
        f"| 日波动率 | {analysis['daily_vol']*100:.2f}% | |",
        f"| ATR(14) | {analysis['atr']:.2f} | |",
        "",
        "---",
        "",
        "## 二、下周价格预测",
        "",
        "### 预测区间",
        "",
        f"基于ATR（平均真实波幅）和历史波动率，预测下周玻璃期货价格运行区间：",
        "",
        f"| 区间类型 | 下限 | 上限 | 幅度 |",
        f"|----------|------|------|------|",
        f"| **主要区间** | **{analysis['range_low']:.0f}** | **{analysis['range_high']:.0f}** | ±{analysis['atr']*2:.0f}点 |",
        f"| 保守区间 | {analysis['latest_close'] - analysis['atr']:.0f} | {analysis['latest_close'] + analysis['atr']:.0f} | ±{analysis['atr']:.0f}点 |",
        "",
        f"**周波动率预估**: {analysis['weekly_vol']*100:.1f}%",
        "",
        "---",
        "",
        "## 三、多情景分析",
        "",
        f"### 情景A: 乐观上涨 (概率 {analysis['bullish_prob']*100:.0f}%)",
        "",
        f"- **触发条件**: 突破 {analysis['high_20d']:.0f}，成交量放大",
        f"- **上行目标**: {analysis['range_high']:.0f}",
        f"- **阻力位**: {analysis['high_20d']:.0f} → {analysis['range_high']:.0f}",
        "- **适合操作**: 突破做多，止损设于突破点下方",
        "",
        f"### 情景B: 区间震荡 (概率 {analysis['neutral_prob']*100:.0f}%)",
        "",
        f"- **预期走势**: 价格在 {analysis['range_low']:.0f} ~ {analysis['range_high']:.0f} 区间震荡",
        f"- **支撑位**: {analysis['low_20d']:.0f}",
        f"- **阻力位**: {analysis['high_20d']:.0f}",
        "- **适合操作**: 高抛低吸，严格止损止盈",
        "",
        f"### 情景C: 回调下跌 (概率 {analysis['bearish_prob']*100:.0f}%)",
        "",
        f"- **触发条件**: 跌破 {analysis['low_20d']:.0f}，成交量放大",
        f"- **下行目标**: {analysis['range_low']:.0f}",
        f"- **支撑位**: {analysis['low_20d']:.0f} → {analysis['range_low']:.0f}",
        "- **适合操作**: 突破做空，止损设于突破点上方",
        "",
        "---",
        "",
        "## 四、综合判断",
        "",
        f"**主基调**: {trend_emoji} **{trend}**",
        "",
        f"- 当前价格: **{analysis['latest_close']:.2f}**",
        f"- 价格位置: 20日区间的 **{analysis['price_position']:.0f}%** 位置",
        f"- 预测下周运行区间: **{analysis['range_low']:.0f} ~ {analysis['range_high']:.0f}**",
        f"- 最可能情景: {'区间震荡' if analysis['neutral_prob'] > max(analysis['bullish_prob'], analysis['bearish_prob']) else ('偏多' if analysis['bullish_prob'] > analysis['bearish_prob'] else '偏空')}",
        "",
        "---",
        "",
        "## 五、关键价位速查",
        "",
        "| 类型 | 价位 | 说明 |",
        "|------|------|------|",
        f"| 当前价 | **{analysis['latest_close']:.0f}** | |",
        f"| 上方阻力 | {analysis['high_20d']:.0f} | 20日高点 |",
        f"| 上方目标 | {analysis['range_high']:.0f} | 区间上沿 |",
        f"| 下方支撑 | {analysis['low_20d']:.0f} | 20日低点 |",
        f"| 下方目标 | {analysis['range_low']:.0f} | 区间下沿 |",
        "",
        "---",
        "",
        "## 六、风险提示",
        "",
        "1. 本报告基于历史数据和技术分析，不构成投资建议",
        "2. 期货市场波动较大，请严格控制仓位和止损",
        "3. 建议结合基本面信息和市场情绪综合判断",
        "4. 实际走势可能与预测存在较大偏差",
        "",
        "---",
        "",
        f"*报告由 futureQuant 自动生成 @ {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
    ]
    
    report_content = "\n".join(lines)
    report_path.write_text(report_content, encoding='utf-8')
    
    logger.info(f"报告已生成: {report_path}")
    print(f"\n{'='*60}")
    print("玻璃期货（FG）下周预测摘要")
    print('='*60)
    print(f"当前价格: {analysis['latest_close']:.2f}")
    print(f"预测区间: {analysis['range_low']:.0f} ~ {analysis['range_high']:.0f}")
    print(f"趋势判断: {trend}")
    print(f"乐观概率: {analysis['bullish_prob']*100:.0f}%")
    print(f"震荡概率: {analysis['neutral_prob']*100:.0f}%")
    print(f"悲观概率: {analysis['bearish_prob']*100:.0f}%")
    print('='*60)
    print(f"\n报告完整路径: {report_path}")
    
    return report_path


def main():
    """主流程"""
    logger.info("=" * 60)
    logger.info("玻璃期货（FG）日频预测分析")
    logger.info("=" * 60)
    
    # 获取数据
    price_df = fetch_data()
    if price_df is None:
        return None
    
    # 计算因子
    factors = compute_factors(price_df)
    
    # 分析预测
    analysis = analyze_and_predict(price_df, factors)
    
    # 生成报告
    report_path = generate_report(analysis)
    
    return report_path


if __name__ == '__main__':
    main()

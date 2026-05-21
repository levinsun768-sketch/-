"""
Polars 预处理大管线配置中心 (Builder Config)
仅用于控制特征工厂的数据源摄入、时空过滤和标准化分组的字段池。

最小示例：
- START_DATE="2021-07-01"
- END_DATE="2026-03-25"
- OUT_DIR="/data/lbsun/std_tensor_dataset"
- INCLUDE_FLAGS=["is_st"]
- EXCLUDE_FLAGS=["is_new", "is_delisted", "is_suspended"]
"""

class PolarsBuilderConfig:
    # =========================================================================
    # 取数时间范围控制 (会自动根据 ROLLING_WINDOW 提前额外读取源端数据)
    # =========================================================================
    START_DATE = "2021-07-01"  # 数据起始日期；示例：2021-07-01
    END_DATE   = "2026-03-25"  # 数据结束日期；示例：2026-03-25
    
    # =========================================================================
    # 生成的终极标准 3D 张量输出保存的根目录
    # =========================================================================
    OUT_DIR = "/data/lbsun/std_tensor_dataset"  # tensor 数据集输出根目录；示例：/data/lbsun/std_tensor_dataset
    
    # =========================================================================
    # 底层 Parquet 原本所在的源头文件夹 (全扫描)
    # =========================================================================
    # 存放分钟级 Pivot 宽表的根目录（我们会在其内部自动寻找 open 等因子名拼接路径）
    MINUTE_DATA_DIR = "/data/file-systems/lvl2_min_bak"  # 分钟级特征 parquet 根目录
    
    # 存放日级筛选条件 Flat 窄表的特定路劲
    DAILY_META_DIR  = "/data/release_dataset/dwd_v5_dd"  # 日频元数据 parquet 根目录
    
    # =========================================================================
    # 模型欲提取的特征集定义 (严格对应分钟级大文件夹下的各个子因子名)
    # =========================================================================
    # 1. 涉及到绝对价格标准化的特征 （使用基于当日开盘价的 price_open_standardize）
    PRICE_FIELDS = [
        "high",                # 最高价
        "low",                 # 最低价
        "close",               # 收盘价
    ]
    
    # 2. 涉及到体量流类对数标准化的特征 （使用 log_flow_standardize）
    FLOW_FIELDS = [
   # 2. 不复权交易特征（14个）
    "total_amount",        # 成交额
    "total_count",         # 成交笔数
    "avg_amount",          # 每笔成交额
    "buy_amount",          # 主买成交额
    "buy_count",           # 主买成交笔数
    "sell_amount",         # 主卖成交额
    "sell_count",          # 主卖成交笔数
    "exlarge_buy_count",   # 主买超大单成交笔数
    "large_buy_count",     # 主买大单成交笔数
    # "mid_buy_count",       # 主买中单成交笔数 # 缺失
    "exlarge_sell_count",  # 主卖超大单成交笔数
    "large_sell_count",    # 主卖大单成交笔数
    "mid_sell_count",      # 主卖中单成交笔数
    # "buy_order_amount",    # 挂单额 # 缺失
    "buy_order_count",     # 挂单笔数
    "buy_cancel_amount",   # 撤单额
    "buy_cancel_count",    # 撤单笔数

    # 3. 需复权交易特征（14个）
    "total_volume",        # 成交量
    "buy_volume",          # 主买成交量
    "sell_volume",         # 主卖成交量
    "exlarge_buy_volume",  # 主买超大单成交量
    "large_buy_volume",    # 主买大单成交量
    "mid_buy_volume",      # 主买中单成交量
    "exlarge_sell_volume", # 主卖超大单成交量
    "large_sell_volume",   # 主卖大单成交量
    "mid_sell_volume",     # 主卖中单成交量
    "buy_cancel_volume",   # 撤单量
    "buy_order_volume"     # 挂单量
    ]

    #是否要复权交易特征
    POST_PROCESS = True  # 是否对分钟成交量类特征乘以复权因子；示例：True
    #需要复权的交易特征
    MINUTE_COLS_POST = [
    "total_volume",        # 成交量
    "buy_volume",          # 主买成交量
    "sell_volume",         # 主卖成交量
    "exlarge_buy_volume",  # 主买超大单成交量
    "large_buy_volume",    # 主买大单成交量
    "mid_buy_volume",      # 主买中单成交量
    "exlarge_sell_volume", # 主卖超大单成交量
    "large_sell_volume",   # 主卖大单成交量
    "mid_sell_volume",     # 主卖中单成交量
    "buy_cancel_volume",   # 撤单量
    "buy_order_volume"     # 挂单量
    ]
    # =========================================================================
    # 底层算子与清洗边界设定
    # =========================================================================
    # True 时，流量类 rolling 统计跨越 regime 边界；False 时按 regime 分段
    CROSS_REGIME = True  # 是否让流量 rolling 跨 regime；示例：True

    ROLLING_WINDOW_DAYS = 20        # 流类标准化 rolling 窗口天数；示例：20
    MINUTES_PER_DAY = 240           # 每个交易日的分钟数；示例：240
    
    EPSILON = 1e-4                  # 流量分母保护下限；示例：1e-4
    FLOW_CLIP_UPPER = 1.0           # 流量标准化后的上限；示例：1.0
    CLIP_UPPER_ABS = 5.0            # 所有标准化特征的绝对截断上限；示例：5.0
    
    # 仅保留同时命中以下标签的个股/交易日；留空表示不过滤
    # 例如只保留 ST 股票时，可设为 ['is_st']
    INCLUDE_FLAGS = ['is_st']  # 必须命中的日频标签；示例：['is_st']；留空表示不过滤

    # 过滤掉带有以下股票池标签的个股（作用于日频过滤端）
    EXCLUDE_FLAGS = [
        'is_new', 'is_delisted',  'is_suspended'
                    #  ,'up_one_line', 'down_one_line'
                     ]  # 需要剔除的日频标签；示例：['is_new', 'is_delisted', 'is_suspended']

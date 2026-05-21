"""
Polars 分钟特征构建脚本 V4。

处理流程：
1. 读取日频元数据，生成有效样本索引与 `regime_id`，并挂载日频参考价和复权因子。
2. 按年度、按股票分块读取分钟级 parquet，构造完整的 237 分钟交易骨架。
3. 在分块内完成缺失清洗、日内填充、可选复权与价格/流量标准化。
4. 写入 memmap 后统一扫描 NaN，输出清洗后的 tensor、meta 和特征配置。

当前版本特性：
- flow 类字段按“当日是否存在任一原始有效值”决定是否补 0。
- 当 `POST_PROCESS=True` 时，对 `MINUTE_COLS_POST` 在填充后、标准化前乘以 `adj_factor`。
- 最终统一记录 NaN 剔除明细，便于定位样本损失原因。
"""
import os
import sys
import glob
import gc
import csv
import numpy as np
import polars as pl
import datetime as dt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from builder_config import PolarsBuilderConfig as cfg

# V4 主流程：
# 1. 读取日频元数据，生成有效样本索引与 regime_id。
# 2. 按年度、按股票分块读取分钟特征，并构造完整分钟骨架。
# 3. 在块内完成缺失处理、可选复权与标准化。
# 4. 统一扫描 NaN，输出 clean tensor、meta 与特征配置。

def get_237_minutes():
    mins = []
    # 上午交易时段：09:31:00 - 11:30:00，共 120 分钟
    cur = dt.datetime(2000, 1, 1, 9, 31, 0)
    for _ in range(120):
        mins.append(cur.strftime('%H:%M:%S'))
        cur += dt.timedelta(minutes=1)
    # 下午交易时段：13:01:00 - 14:57:00，共 117 分钟
    cur = dt.datetime(2000, 1, 1, 13, 1, 0)
    for _ in range(117):
        mins.append(cur.strftime('%H:%M:%S'))
        cur += dt.timedelta(minutes=1)
    return mins


def flag_is_true_expr(flag_name: str) -> pl.Expr:
    return (pl.col(flag_name) == 1) | (pl.col(flag_name) == True)

def build_global_meta():
    print("-> [1/3] 读取日频元数据并生成样本索引...")
    safe_start_dt = dt.datetime.strptime(cfg.START_DATE, "%Y-%m-%d") - dt.timedelta(days=cfg.ROLLING_WINDOW_DAYS * 3)
    safe_str = safe_start_dt.strftime('%Y-%m-%d')
    
    df = pl.scan_parquet(os.path.join(cfg.DAILY_META_DIR, "*.parquet")).collect()
    
    if "instrument" in df.columns: df = df.rename({"instrument": "stock"})
    if "datetime" in df.columns and "date" not in df.columns: df = df.rename({"datetime": "date"})
        
    if df["date"].dtype in [pl.Int64, pl.Int32]:
        df = df.with_columns(pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d"))
    df = df.with_columns(pl.col("date").cast(pl.Date).cast(pl.Utf8))

    include_flags = [flag for flag in getattr(cfg, "INCLUDE_FLAGS", []) if flag in df.columns]
    missing_include_flags = [flag for flag in getattr(cfg, "INCLUDE_FLAGS", []) if flag not in df.columns]
    exclude_flags = [flag for flag in getattr(cfg, "EXCLUDE_FLAGS", []) if flag not in set(include_flags)]
    missing_exclude_flags = [flag for flag in exclude_flags if flag not in df.columns]

    if include_flags:
        include_expr = pl.lit(True)
        for flag in include_flags:
            include_expr = include_expr & flag_is_true_expr(flag)
    else:
        include_expr = pl.lit(True)

    invalid_expr = ~include_expr
    for flag in exclude_flags:
        if flag in df.columns:
            invalid_expr = invalid_expr | flag_is_true_expr(flag)

    if include_flags:
        print(f"   INCLUDE_FLAGS 生效: {include_flags}")
    if missing_include_flags:
        print(f"   INCLUDE_FLAGS 缺失列，已跳过: {missing_include_flags}")
    if exclude_flags:
        print(f"   EXCLUDE_FLAGS 生效: {[flag for flag in exclude_flags if flag in df.columns]}")
    if missing_exclude_flags:
        print(f"   EXCLUDE_FLAGS 缺失列，已跳过: {missing_exclude_flags}")

    df = df.with_columns(invalid_expr.alias("is_invalid"))
    df = df.sort(["stock", "date"])
    df = df.with_columns(pl.col("is_invalid").cast(pl.Int32).cum_sum().over("stock").alias("regime_id"))
    
    # 读取日频参考价格，用于分钟价格标准化和 price_pos 计算。
    # 若日频表存在 factor，则：
    # 1. open / limit_up / limit_down 先做除权；
    # 2. 同时保留 adj_factor，供分钟成交量类字段在后处理中复权。
    daily_ref_cols = ["stock", "date"]
    daily_ref_exprs = []
    factor_denom = None
    if "factor" in df.columns:
        factor_denom = (
            pl.when(
                pl.col("factor").is_null() |
                pl.col("factor").is_nan() |
                pl.col("factor").is_infinite() |
                (pl.col("factor") == 0)
            )
            .then(None)
            .otherwise(pl.col("factor"))
        )
        daily_ref_exprs.append(factor_denom.alias("adj_factor"))
    if "open" in df.columns:
        expr = pl.col("open") / factor_denom if factor_denom is not None else pl.col("open")
        daily_ref_exprs.append(expr.alias("raw_open"))
    if "limit_up" in df.columns:
        expr = pl.col("limit_up") / factor_denom if factor_denom is not None else pl.col("limit_up")
        daily_ref_exprs.append(expr.alias("raw_limit_up"))
    if "limit_down" in df.columns:
        expr = pl.col("limit_down") / factor_denom if factor_denom is not None else pl.col("limit_down")
        daily_ref_exprs.append(expr.alias("raw_limit_down"))

    if daily_ref_exprs:
        daily_ref = df.filter(~pl.col("is_invalid")).select(daily_ref_cols + daily_ref_exprs)
        daily_ref = daily_ref.filter((pl.col("date") >= safe_str) & (pl.col("date") <= cfg.END_DATE))
    else:
        daily_ref = None
    
    valid_daily_all = df.filter(~pl.col("is_invalid")).select(["date", "stock", "regime_id"])
    valid_daily_all = valid_daily_all.filter((pl.col("date") >= safe_str) & (pl.col("date") <= cfg.END_DATE))
    
    valid_targets = valid_daily_all.filter((pl.col("date") >= cfg.START_DATE) & (pl.col("date") <= cfg.END_DATE))
    valid_targets = valid_targets.sort(["stock", "date"]).with_columns(pl.int_range(0, pl.len()).alias("sample_id"))
    
    valid_daily_ext = valid_daily_all.join(valid_targets.select(["date", "stock", "sample_id"]), on=["date", "stock"], how="left")
    
    return valid_daily_ext, valid_targets, daily_ref

def process_single_yearly_chunk(y_start, y_end, valid_daily_ext, valid_targets, daily_ref, mmap_path, final_feats):
    """
    处理单个年度区间的数据块。

    执行顺序：
    1. 为当前股票块构造完整的日内分钟骨架。
    2. 逐个特征读取 parquet，并仅裁剪当前股票块涉及的列。
    3. 在块内完成缺失值处理、价格标准化和 flow 滚动归一化。
    4. 将结果写入 memmap，避免一次性持有全量张量。
    """
    safe_start = (dt.datetime.strptime(y_start, "%Y-%m-%d") - dt.timedelta(days=cfg.ROLLING_WINDOW_DAYS * 3)).strftime('%Y-%m-%d')
    all_features = list(dict.fromkeys(cfg.PRICE_FIELDS + cfg.FLOW_FIELDS))
    has_close = "close" in all_features
    flow_feats = [f for f in cfg.FLOW_FIELDS if f in all_features]
    
    chunk_ext = valid_daily_ext.filter((pl.col("date") >= safe_start) & (pl.col("date") <= y_end))
    if chunk_ext.height == 0: return 0
    
    # 当前年度区间内涉及的股票列表
    unique_stocks = chunk_ext.select("stock").unique()["stock"].to_list()
    grid_df = pl.DataFrame({"time_seq": get_237_minutes()})
    
    N_chunk_samples = 0
    fp = np.memmap(mmap_path, dtype='float32', mode='r+', shape=(valid_targets.height, 237, len(final_feats)))
    
    # 每次仅处理一个股票块，控制内存峰值
    CHUNK_SIZE = 500
    
    print(f"   [{y_start[:4]}] 开始年度分块处理，股票数: {len(unique_stocks)}")
    for i in tqdm(range(0, len(unique_stocks), CHUNK_SIZE), desc=f"   [{y_start[:4]}] 写入年度分块"):
        batch_stocks = unique_stocks[i : i+CHUNK_SIZE]
        
        # 1. 为当前股票块构造完整的日内分钟骨架
        spine_batch_lf = chunk_ext.lazy().filter(
            pl.col("stock").is_in(batch_stocks)
        ).join(grid_df.lazy(), how="cross").with_columns(
            (pl.col("date") + " " + pl.col("time_seq")).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("datetime")
        ).drop("time_seq").sort(["stock", "datetime"]) 
        
        lf = spine_batch_lf
        
        # 2. 逐个特征读取分钟数据，并仅保留当前股票块对应的列
        t_start = dt.datetime.strptime(f"{safe_start} 00:00:00", "%Y-%m-%d %H:%M:%S")
        t_end   = dt.datetime.strptime(f"{y_end} 23:59:59", "%Y-%m-%d %H:%M:%S")
        
        for feat in all_features:
            feat_dir = os.path.join(cfg.MINUTE_DATA_DIR, feat)
            if not os.path.exists(feat_dir):
                lf = lf.with_columns(pl.lit(float('nan')).cast(pl.Float32).alias(feat))
                continue
                
            pq_files = glob.glob(os.path.join(feat_dir, "*.parquet"))
            single_lfs = []
            
            for pq_f in pq_files:
                try:
                    # 先读取 schema，确认当前文件中真实存在的股票列
                    cols_in_file = pl.read_parquet_schema(pq_f).keys()
                    valid_cols = [s for s in batch_stocks if s in cols_in_file]
                    # print(f"   [{y_start[:4]}] 正在读取文件: {pq_f}, 发现有效股票: {len(valid_cols)} 只")
                    if not valid_cols:
                        continue
                        
                    # 列裁剪读取，避免无关股票列参与解压和反透视
                    feat_lf = pl.scan_parquet(pq_f).select(["datetime"] + valid_cols)
                    feat_lf = feat_lf.filter(
                        (pl.col("datetime").cast(pl.Datetime) >= t_start) & 
                        (pl.col("datetime").cast(pl.Datetime) <= t_end)
                    )
                    # 宽表转长表，统一为 [datetime, stock, feat] 结构
                    feat_lf = feat_lf.unpivot(index="datetime", variable_name="stock", value_name=feat).drop_nulls(feat)
                    single_lfs.append(feat_lf)
                except Exception as e:
                    print(f"\n[跳过异常文件] {pq_f} -> {e}")
                    continue
                    
            if single_lfs:
                feat_df = pl.concat(single_lfs).unique(subset=["datetime", "stock"], keep="last")
                lf = lf.join(feat_df, on=["datetime", "stock"], how="left")
            else:
                lf = lf.with_columns(pl.lit(float('nan')).cast(pl.Float32).alias(feat))
                
        # 3. 挂载日频参考信息，并在块内完成标准化与缺失处理
        if daily_ref is not None:
            ref_chunk = daily_ref.filter((pl.col("date") >= safe_start) & (pl.col("date") <= y_end)).filter(pl.col("stock").is_in(batch_stocks))
            lf = lf.join(ref_chunk.lazy(), on=["stock", "date"], how="left")
            
        lf_cols = lf.collect_schema().names()
        has_raw_open     = "raw_open" in lf_cols
        has_raw_lim_up   = "raw_limit_up" in lf_cols
        has_raw_lim_down = "raw_limit_down" in lf_cols

        p_feats = [f for f in cfg.PRICE_FIELDS if f in lf_cols]
        flow_feats_actual = [f for f in flow_feats if f in lf_cols]
        minute_feats_actual = p_feats + flow_feats_actual

        lf = lf.sort(["stock", "datetime"])

        cleaned_exprs = []
        valid_count_exprs = []
        for feat in minute_feats_actual:
            invalid_expr = pl.col(feat).is_null() | pl.col(feat).is_infinite() | pl.col(feat).is_nan()
            if feat in p_feats:
                invalid_expr = invalid_expr | (pl.col(feat) == 0)
            cleaned_exprs.append(
                pl.when(invalid_expr).then(None).otherwise(pl.col(feat)).cast(pl.Float32).alias(feat)
            )
        lf = lf.with_columns(cleaned_exprs)

        for feat in minute_feats_actual:
            valid_count_exprs.append(
                pl.col(feat).is_not_null().cast(pl.Int32).sum().over(["stock", "date"]).alias(f"__{feat}_valid_count")
            )
        lf = lf.with_columns(valid_count_exprs)

        if flow_feats_actual:
            lf = lf.with_columns(
                pl.any_horizontal([pl.col(f"__{feat}_valid_count") > 0 for feat in flow_feats_actual]).alias("__raw_flow_day_has_value")
            )

        fill_exprs = []
        for feat in p_feats:
            valid_count_col = f"__{feat}_valid_count"
            fill_exprs.append(
                pl.when(pl.col(valid_count_col) > 0)
                .then(pl.col(feat).forward_fill().backward_fill().over(["stock", "date"]))
                .otherwise(None)
                .alias(feat)
            )
        for feat in flow_feats_actual:
            valid_count_col = f"__{feat}_valid_count"
            fill_exprs.append(
                pl.when(pl.col("__raw_flow_day_has_value"))
                .then(pl.col(feat).fill_null(0.0))
                .otherwise(None)
                .alias(feat)
            )
        lf = lf.with_columns(fill_exprs)

        if getattr(cfg, "POST_PROCESS", False):
            post_cols = [c for c in getattr(cfg, "MINUTE_COLS_POST", []) if c in flow_feats_actual]
            has_adj_factor = "adj_factor" in lf.collect_schema().names()
            if post_cols and has_adj_factor:
                lf = lf.with_columns([
                    (
                        pl.when(
                            pl.col(col).is_null() |
                            pl.col("adj_factor").is_null() |
                            pl.col("adj_factor").is_nan() |
                            pl.col("adj_factor").is_infinite() |
                            (pl.col("adj_factor") == 0)
                        )
                        .then(pl.col(col))
                        .otherwise(pl.col(col) * pl.col("adj_factor"))
                    ).alias(col)
                    for col in post_cols
                ])

        if flow_feats_actual:
            lf = lf.with_columns(
                pl.col("__raw_flow_day_has_value").alias("__valid_flow_day")
            )

        std_exprs = []
        if has_close and has_raw_lim_down and has_raw_lim_up and "price_pos" in final_feats:
            expr = (
                (pl.col("close") - pl.col("raw_limit_down")) / 
                (pl.col("raw_limit_up") - pl.col("raw_limit_down") + 1e-8)
            ).clip(lower_bound=-cfg.CLIP_UPPER_ABS, upper_bound=cfg.CLIP_UPPER_ABS).fill_null(float('nan'))
            std_exprs.append(expr.alias("price_pos"))
            
        for pf in cfg.PRICE_FIELDS:
            if pf in final_feats and pf in lf_cols:
                if has_raw_open:
                    expr = (pl.col(pf) / pl.col("raw_open")) - 1.0
                else:
                    expr = pl.lit(float('nan')).cast(pl.Float32)
                expr = pl.when(expr.is_infinite() | expr.is_nan()).then(None).otherwise(expr)
                std_exprs.append(expr.alias(pf))
                
        daily_agg_base = lf.filter(pl.col("__valid_flow_day")) if flow_feats_actual else lf
        daily_aggs = daily_agg_base.group_by(["stock", "date", "regime_id"]).agg([pl.col(f).sum().alias(f"{f}_sum") for f in flow_feats_actual]).sort(["stock", "date"])
        
        daily_aggs = daily_aggs.with_columns([
            pl.col(f"{ff}_sum").rolling_mean(cfg.ROLLING_WINDOW_DAYS, min_periods=cfg.ROLLING_WINDOW_DAYS).shift(1).over(["stock", "regime_id"]).alias(f"{ff}_roll")
            for ff in flow_feats_actual
        ])
        
        lf = lf.join(daily_aggs.select(["stock", "date", "regime_id"] + [f"{ff}_roll" for ff in flow_feats_actual]), on=["stock", "date", "regime_id"], how="left")
        
        for ff in flow_feats_actual:
            roll_safe = pl.col(f"{ff}_roll").clip(lower_bound=cfg.EPSILON)
            expr = pl.col(ff) / roll_safe
            expr = pl.when(expr.is_infinite() | expr.is_nan()).then(None).otherwise(expr)
            expr = expr.clip(upper_bound=cfg.FLOW_CLIP_UPPER)
            std_exprs.append(expr.alias(ff))
            
        lf = lf.with_columns(std_exprs)
        lf = lf.with_columns([
            pl.col(f).fill_null(float('nan')).clip(lower_bound=-cfg.CLIP_UPPER_ABS, upper_bound=cfg.CLIP_UPPER_ABS).alias(f) 
            for f in final_feats
        ])
        
        # 4. 实体化结果并写入 memmap
        lf = lf.filter((pl.col("date") >= y_start) & (pl.col("date") <= y_end))
        lf = lf.sort(["stock", "datetime"])
        
        df_batch = lf.select(["stock", "date", "datetime", "sample_id"] + final_feats).collect()
        
        N_batch_samples = df_batch.height // 237
        if N_batch_samples == 0:
            continue
            
        N_chunk_samples += N_batch_samples
        batch_sample_ids = df_batch.select(["stock", "date", "sample_id"]).unique(subset=["stock", "date"], keep="first", maintain_order=True)["sample_id"].to_numpy()
        
        arr_3d = df_batch.select(final_feats).to_numpy().reshape((N_batch_samples, 237, len(final_feats)))
        fp[batch_sample_ids, :, :] = arr_3d[:]
        fp.flush()
        
        del df_batch, arr_3d
        gc.collect()

    del fp
    return N_chunk_samples


def scan_and_log_nan_samples(fp, meta_df, final_feats, job_dir, chunk_size=4096):
    """
    扫描最终 tensor 中因 NaN 将被删除的样本，并输出明细日志与特征汇总。

    返回：
    - valid_mask: 可保留样本的布尔掩码
    - detail_path: 明细日志路径
    - summary_path: 汇总日志路径
    - invalid_count: 被删除样本数
    """
    detail_path = os.path.join(job_dir, "dropped_nan_samples.csv")
    summary_path = os.path.join(job_dir, "dropped_nan_feature_summary.csv")

    n_samples = fp.shape[0]
    valid_mask = np.ones(n_samples, dtype=bool)
    feature_missing_sample_count = np.zeros(len(final_feats), dtype=np.int64)
    feature_missing_cell_count = np.zeros(len(final_feats), dtype=np.int64)
    invalid_count = 0

    with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["stock", "date", "missing_features", "missing_feature_count", "missing_cell_count"])

        for start in tqdm(range(0, n_samples, chunk_size), desc="   扫描 NaN 样本"):
            end = min(start + chunk_size, n_samples)
            chunk = np.asarray(fp[start:end])
            missing = np.isnan(chunk)
            invalid_local = missing.any(axis=(1, 2))
            valid_mask[start:end] = ~invalid_local

            if not invalid_local.any():
                continue

            invalid_count += int(invalid_local.sum())
            meta_chunk = meta_df.iloc[start:end].reset_index(drop=True)
            feat_missing = missing.any(axis=1)
            feat_missing_count = feat_missing.sum(axis=1)
            cell_missing_count = missing.sum(axis=(1, 2))

            feature_missing_sample_count += feat_missing[invalid_local].sum(axis=0)
            feature_missing_cell_count += missing[invalid_local].sum(axis=(0, 1))

            invalid_idx = np.where(invalid_local)[0]
            for local_idx in invalid_idx:
                row = meta_chunk.iloc[local_idx]
                missing_features = [final_feats[j] for j, flag in enumerate(feat_missing[local_idx]) if flag]
                writer.writerow([
                    row["stock"],
                    row["date"],
                    "|".join(missing_features),
                    int(feat_missing_count[local_idx]),
                    int(cell_missing_count[local_idx]),
                ])

    summary_df = pl.DataFrame({
        "feature": final_feats,
        "dropped_sample_count": feature_missing_sample_count.tolist(),
        "missing_cell_count": feature_missing_cell_count.tolist(),
    }).sort("dropped_sample_count", descending=True)
    summary_df.write_csv(summary_path)

    return valid_mask, detail_path, summary_path, invalid_count

def run_single_process_polars():
    print("========================================")
    print("Polars 分钟特征构建引擎 V4")
    print("========================================")
    
    valid_daily_ext, valid_targets, daily_ref = build_global_meta()
    
    all_features = list(dict.fromkeys(cfg.PRICE_FIELDS + cfg.FLOW_FIELDS))
    has_close = "close" in all_features
    has_lim_down = daily_ref is not None and "raw_limit_down" in daily_ref.columns
    has_lim_up   = daily_ref is not None and "raw_limit_up"   in daily_ref.columns
    
    final_feats = cfg.PRICE_FIELDS + cfg.FLOW_FIELDS + (["price_pos"] if has_close and has_lim_up and has_lim_down else [])
    
    N_total_samples = valid_targets.height
    print(f"-> [2/3] 有效目标样本数: {N_total_samples}")
    
    job_dir = os.path.join(cfg.OUT_DIR, f"tensor_dataset_v4_{cfg.START_DATE}_{cfg.END_DATE}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(job_dir, exist_ok=True)
    mmap_path = os.path.join(job_dir, "raw_tensor_cache.npy")
    clean_path = os.path.join(job_dir, "clean_tensor.npy")
    
    fp = np.memmap(mmap_path, dtype='float32', mode='w+', shape=(N_total_samples, 237, len(final_feats)))
    fp[:] = np.nan
    fp.flush()
    del fp
    
    y_chunks, cur = [], dt.datetime.strptime(cfg.START_DATE, "%Y-%m-%d")
    while cur.year <= dt.datetime.strptime(cfg.END_DATE, "%Y-%m-%d").year:
        y_c_start = max(cur, dt.datetime(cur.year, 1, 1)).strftime('%Y-%m-%d')
        y_c_end = min(dt.datetime.strptime(cfg.END_DATE, "%Y-%m-%d"), dt.datetime(cur.year, 12, 31)).strftime('%Y-%m-%d')
        if y_c_start <= y_c_end:
            y_chunks.append((y_c_start, y_c_end))
        cur = dt.datetime(cur.year + 1, 1, 1)
        
    for (y_start, y_end) in y_chunks:
        n = process_single_yearly_chunk(y_start, y_end, valid_daily_ext, valid_targets, daily_ref, mmap_path, final_feats)
        print(f"   [{y_start[:4]}] 完成，写入样本数: {n}")
        
    print("\n-> [3/3] 扫描 NaN 并导出清洗结果...")
    
    fp = np.memmap(mmap_path, dtype='float32', mode='r+', shape=(N_total_samples, 237, len(final_feats)))
    meta_df = valid_targets.sort(["stock", "date"]).select(["stock", "date"]).to_pandas()
    valid_mask, nan_detail_path, nan_summary_path, invalid_count = scan_and_log_nan_samples(
        fp=fp,
        meta_df=meta_df,
        final_feats=final_feats,
        job_dir=job_dir,
    )
    clean_meta = meta_df[valid_mask]
    
    clean_tensors = fp[valid_mask]
    np.save(clean_path, clean_tensors)
    del clean_tensors; gc.collect()
    
    clean_meta.to_csv(os.path.join(job_dir, "tensor_meta.csv"), index=False)
    
    price_indices = [i for i, c in enumerate(final_feats) if c in ["high","low","close","price_pos"] or "Price" in c]
    trade_indices = [i for i in range(len(final_feats)) if i not in price_indices]
    
    with open(os.path.join(job_dir, "feature_config.yaml"), "w", encoding="utf-8") as f:
        f.write("# V4 Pipeline Config\n")
        f.write(f"F_DIM: {len(final_feats)}\nPRICE_IDX: {price_indices}\nTRADE_IDX: {trade_indices}\n")
        f.write("FREQ: lvl2_min_v4\n")
        f.write("\n# 明细:\n")
        for idx, col in enumerate(final_feats):
            f.write(f"#   - [{idx:02d}] {col.ljust(20)} : {'Price' if idx in price_indices else 'Flow'}\n")
            
    del fp
    os.remove(mmap_path)
    print(f"\n构建完成，有效切片数: {len(clean_meta)}")
    print(f"NaN 剔除样本数: {invalid_count}")
    print(f"NaN 样本明细: {nan_detail_path}")
    print(f"NaN 特征汇总: {nan_summary_path}")
    print(f"输出文件: {clean_path}")

if __name__ == "__main__":
    run_single_process_polars()

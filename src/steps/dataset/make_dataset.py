import os
import duckdb
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

def export_last_n_days_call_histories(
    in_path: str,
    out_dir: str,
    days: int = 90,
    tz: str = "Asia/Ho_Chi_Minh",
    threads: int = 4,
) -> str:
    """
    For each run:
      end_date   = current date in timezone `tz`
      start_date = end_date - (days - 1) days (inclusive range with exactly `days` days)
    Select columns:
      phone_member (from member_id), phone, report (copied from phone), spam (=0),
      type, time, in_contact, duration
    then save to parquet in out_dir.

    Returns:
        out_path (saved parquet file path)
    """
    # 1) Compute dates in the provided timezone
    end_d = datetime.now(ZoneInfo(tz)).date()
    start_d = end_d - timedelta(days=days - 1)

    start_date = start_d.isoformat()
    end_date = end_d.isoformat()

    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(
        out_dir,
        f"call_histories_last_{days}d_{start_date}_to_{end_date}.parquet"
    )

    in_sql  = in_path.replace("'", "''")
    out_sql = out_path.replace("'", "''")

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(threads)};")

    con.execute(f"""
    COPY (
      SELECT
        CAST(member_id AS VARCHAR) AS phone_member,
        CAST(phone     AS VARCHAR) AS phone,
        CAST(phone     AS VARCHAR) AS report,
        0::INTEGER                 AS spam,
        "type",
        "time",
        in_contact,
        duration
      FROM read_parquet('{in_sql}')
      WHERE DATE(TRY_CAST("time" AS TIMESTAMP))
        BETWEEN DATE '{start_date}' AND DATE '{end_date}'
    )
    TO '{out_sql}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    con.close()
    print(f"Exported last {days} days: {start_date} -> {end_date}")
    print("Wrote:", out_path)
    return out_path


def export_last_n_days_report(
    in_path: str,
    out_dir: str,
    days: int = 90,
    tz: str = "Asia/Ho_Chi_Minh",
    threads: int = 4,
) -> str:
    """
    Export data for the last N days with these rules:
      - SELECT * (keep all columns, no rename)
      - Keep the output filename identical to input (basename)
      - Filter by preferred time column: report_time -> time
      - Use inclusive date window: [end - (days-1), end] in timezone `tz`

    Returns: out_path
    """
    # 1) Compute date window in the provided timezone
    end_d = datetime.now(ZoneInfo(tz)).date()
    start_d = end_d - timedelta(days=days - 1)
    start_date = start_d.isoformat()
    end_date = end_d.isoformat()

    os.makedirs(out_dir, exist_ok=True)

    # 2) Keep the same filename as the input
    out_path = os.path.join(out_dir, os.path.basename(in_path))

    in_sql = in_path.replace("'", "''")
    out_sql = out_path.replace("'", "''")

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(threads)};")

    # 3) Detect the time column used for filtering
    schema_cols = con.execute(f"SELECT * FROM read_parquet('{in_sql}') LIMIT 0").df().columns
    cols_lower = {c.lower(): c for c in schema_cols}

    if "report_time" in cols_lower:
        time_col = cols_lower["report_time"]
    elif "time" in cols_lower:
        time_col = cols_lower["time"]
    else:
        time_col = None

    # 4) COPY while preserving schema
    if time_col:
        where_clause = f"""
        WHERE DATE(TRY_CAST("{time_col}" AS TIMESTAMP))
          BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        """
    else:
        # No time/report_time column found, so date filtering is skipped
        where_clause = ""

    con.execute(f"""
    COPY (
      SELECT *
      FROM read_parquet('{in_sql}')
      {where_clause}
    )
    TO '{out_sql}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    con.close()
    print(f"Exported last {days} days: {start_date} -> {end_date}")
    print(out_path)
    return out_path

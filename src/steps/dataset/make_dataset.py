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
    Mỗi lần chạy:
      end_date   = ngày hiện tại theo tz
      start_date = end_date - (days-1) ngày  (đảm bảo đủ đúng 'days' ngày, inclusive)
    Lấy các cột:
      phone_member (từ member_id), phone, report (copy từ phone), spam (=0), type, time, in_contact, duration
    rồi lưu ra parquet trong out_dir.

    Returns:
        out_path (đường dẫn file parquet đã lưu)
    """
    # 1) Tính ngày theo timezone
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
    print(f"✅ Exported last {days} days: {start_date} -> {end_date}")
    print("✅ Wrote:", out_path)
    return out_path


def export_last_n_days_report(
    in_path: str,
    out_dir: str,
    days: int = 90,
    tz: str = "Asia/Ho_Chi_Minh",
    threads: int = 4,
) -> str:
    """
    Export dữ liệu last N days nhưng:
      - SELECT * (giữ nguyên toàn bộ cột, không rename)
      - Output giữ nguyên tên file như input (basename)
      - Lọc theo cột thời gian ưu tiên: report_time -> time
      - Lọc date inclusive: [end - (days-1), end] theo timezone tz

    Returns: out_path
    """
    # 1) Tính khoảng ngày theo timezone
    end_d = datetime.now(ZoneInfo(tz)).date()
    start_d = end_d - timedelta(days=days - 1)
    start_date = start_d.isoformat()
    end_date = end_d.isoformat()

    os.makedirs(out_dir, exist_ok=True)

    # 2) Giữ nguyên tên file như input
    out_path = os.path.join(out_dir, os.path.basename(in_path))

    in_sql = in_path.replace("'", "''")
    out_sql = out_path.replace("'", "''")

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(threads)};")

    # 3) Dò cột thời gian để lọc
    schema_cols = con.execute(f"SELECT * FROM read_parquet('{in_sql}') LIMIT 0").df().columns
    cols_lower = {c.lower(): c for c in schema_cols}

    if "report_time" in cols_lower:
        time_col = cols_lower["report_time"]
    elif "time" in cols_lower:
        time_col = cols_lower["time"]
    else:
        time_col = None

    # 4) COPY giữ nguyên schema
    if time_col:
        where_clause = f"""
        WHERE DATE(TRY_CAST("{time_col}" AS TIMESTAMP))
          BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        """
    else:
        # Không có cột time/report_time => không lọc được theo ngày
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
    print(f"✅ Exported last {days} days: {start_date} -> {end_date}")
    print(out_path)
    return out_path

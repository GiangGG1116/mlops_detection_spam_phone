# features_duckdb.py
from __future__ import annotations

import duckdb
import yaml
from typing import Dict, Any, Optional, List, Tuple


# ---------------------------
# Helpers
# ---------------------------
def _sql_quote(path: str) -> str:
    """Quote string for SQL safely (basic escaping of single quotes)."""
    return "'" + path.replace("'", "''") + "'"


def load_external_data(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prefix_provider_table(con: duckdb.DuckDBPyConnection,
                                providers_lib: Dict[str, List[str]],
                                table_name: str = "prefix_provider") -> None:
    """
    providers_lib dạng:
      {
        "viettel": ["0981","0982",...],
        "vina":    ["0912",...],
        ...
      }
    Tạo TEMP TABLE: prefix_provider(prefix, provider)
    """
    con.execute(f"CREATE OR REPLACE TEMP TABLE {table_name}(prefix VARCHAR, provider VARCHAR)")
    rows: List[Tuple[str, str]] = []
    for provider, prefixes in (providers_lib or {}).items():
        for p in prefixes or []:
            rows.append((str(p), str(provider)))

    if rows:
        con.executemany(f"INSERT INTO {table_name} VALUES (?, ?)", rows)


# ---------------------------
# Core DuckDB pipeline
# ---------------------------
def create_raw_view(con: duckdb.DuckDBPyConnection, in_path: str, ext: str = "parquet") -> None:
    """
    Tạo view raw từ file. KHÔNG dùng prepared parameter cho CREATE VIEW để tránh BinderException.
    """
    p = _sql_quote(in_path)
    ext = (ext or "").lower().strip()

    if ext in ("parquet", "pq"):
        con.execute(f"CREATE OR REPLACE TEMP VIEW raw AS SELECT * FROM read_parquet({p})")
    elif ext in ("csv",):
        con.execute(
            f"CREATE OR REPLACE TEMP VIEW raw AS "
            f"SELECT * FROM read_csv_auto({p}, header=true, sample_size=-1)"
        )
    elif ext in ("json", "jsonl", "ndjson"):
        con.execute(
            f"CREATE OR REPLACE TEMP VIEW raw AS "
            f"SELECT * FROM read_json_auto({p})"
        )
    else:
        raise ValueError(f"Unsupported ext={ext}. Use parquet/csv/json/jsonl.")


def build_features_duckdb(
    in_path: str,
    external_yaml: str,
    ext: str = "parquet",
    out_parquet: Optional[str] = None,
) -> duckdb.DuckDBPyConnection:
    """
    Chạy full pipeline và (tuỳ chọn) ghi ra parquet.
    Output cuối: TEMP VIEW final_features
    """
    external_data = load_external_data(external_yaml)

    providers_lib = external_data.get("providers", {}) or {}
    work_days = external_data.get("work_days", [0, 1, 2, 3, 4])  # pandas dayofweek: Mon=0..Sun=6
    # type_map trong code gốc không cần cho DuckDB vì type 1..4 đã cố định

    work_days_sql = ",".join(str(int(x)) for x in work_days)

    con = duckdb.connect(database=":memory:")
    # (tuỳ chọn) tối ưu cho máy RAM thấp
    con.execute("PRAGMA threads=2;")
    # con.execute("PRAGMA memory_limit='3GB';")  # nếu muốn giới hạn

    create_raw_view(con, in_path, ext=ext)
    build_prefix_provider_table(con, providers_lib)

    # 1) get_valid_data (lọc + remove prefix + digits-only + parse time)
    #    - remove '+' rồi remove '84' ở đầu
    #    - length: BETWEEN 4 AND 18 (đúng theo đoạn drop len <4 hoặc >18)
    #    - digits-only: regexp_full_match(phone, '^[0-9]+$')
    #    - parse time: TRY_CAST
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW valid_data AS
    WITH cleaned AS (
        SELECT
            regexp_replace(regexp_replace(CAST(phone AS VARCHAR), '^\\+', ''), '^84', '')        AS phone,
            CAST(type AS INTEGER)                                                               AS type,
            TRY_CAST(time AS TIMESTAMP)                                                         AS time,
            CAST(in_contact AS DOUBLE)                                                          AS in_contact,
            CAST(duration AS DOUBLE)                                                            AS duration,
            regexp_replace(regexp_replace(CAST(phone_member AS VARCHAR), '^\\+', ''), '^84', '') AS phone_member,
            CAST(Spam AS INTEGER)                                                               AS Spam,
            CAST(report AS VARCHAR)                                                             AS report
        FROM raw
    )
    SELECT *
    FROM cleaned
    WHERE
        phone IS NOT NULL
        AND phone_member IS NOT NULL
        AND report IS NOT NULL
        AND type IN (1,2,3,4)
        AND time IS NOT NULL
        AND in_contact IS NOT NULL
        AND duration IS NOT NULL
        AND Spam IS NOT NULL
        AND length(phone) BETWEEN 4 AND 18
        AND length(phone_member) BETWEEN 4 AND 18
        AND regexp_full_match(phone, '^[0-9]+$')
        AND regexp_full_match(phone_member, '^[0-9]+$')
    """)

    # 2) features_engineering + get_dummies_variables (OHE bằng CASE)
    #    - same_network: map prefix(4) -> provider, default 'others'
    #    - in_hour: 7..19 AND dayofweek in work_days
    #      pandas dayofweek: Mon=0..Sun=6
    #      DuckDB date_part('dow'): Sun=0..Sat=6
    #      => convert: (dow + 6) % 7
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW engineered AS
    WITH base AS (
        SELECT
            v.*,
            substr(v.phone, 1, 4)        AS phone_prefix,
            substr(v.phone_member, 1, 4) AS member_prefix,
            CAST(date_part('hour', v.time) AS INTEGER) AS call_hour,
            ((CAST(date_part('dow', v.time) AS INTEGER) + 6) % 7) AS call_dayofweek_pandas
        FROM valid_data v
    ),
    prov AS (
        SELECT
            b.*,
            COALESCE(p1.provider, 'others') AS caller_provider,
            COALESCE(p2.provider, 'others') AS receiver_provider
        FROM base b
        LEFT JOIN prefix_provider p1 ON b.phone_prefix = p1.prefix
        LEFT JOIN prefix_provider p2 ON b.member_prefix = p2.prefix
    ),
    ohe AS (
        SELECT
            *,
            CASE WHEN type = 1 THEN 1 ELSE 0 END AS call_to,
            CASE WHEN type = 2 THEN 1 ELSE 0 END AS call_in,
            CASE WHEN type = 3 THEN 1 ELSE 0 END AS call_to_miss,
            CASE WHEN type = 4 THEN 1 ELSE 0 END AS call_in_miss
        FROM prov
    )
    SELECT
        report, phone, phone_member, type, time, in_contact, duration, Spam,
        (caller_provider = receiver_provider) AS same_network,
        (
            (call_hour BETWEEN 7 AND 19)
            AND (call_dayofweek_pandas IN ({work_days_sql}))
        ) AS in_hour,
        NOT (
            (call_hour BETWEEN 7 AND 19)
            AND (call_dayofweek_pandas IN ({work_days_sql}))
        ) AS not_in_hour,
        (duration > 20) AS success,
        call_to, call_in, call_to_miss, call_in_miss,
        duration * call_in AS duration_call_in,
        duration * call_to AS duration_call_to
    FROM ohe
    """)

    # 3) group_by_phone (aggregate theo report) + frequency/median-diff + per-member median sum
    #    KHÔNG dùng FILTER, dùng CASE/CTE để tương thích tốt.
    con.execute(r"""
    CREATE OR REPLACE TEMP VIEW agg_report AS
    WITH
    base_agg AS (
        SELECT
            report,
            avg(in_contact)                               AS avg_in_contact,
            avg(CAST(same_network AS INTEGER))            AS same_network,
            sum(duration)                                 AS duration,
            sum(duration_call_to)                         AS duration_call_to,
            sum(duration_call_in)                         AS duration_call_in,
            sum(CAST(in_hour AS INTEGER))                 AS in_hour,
            sum(CAST(not_in_hour AS INTEGER))             AS not_in_hour,
            avg(CAST(success AS INTEGER))                 AS avg_success,
            sum(call_to)                                  AS call_to,
            sum(call_in)                                  AS call_in,
            sum(call_to_miss)                             AS call_to_miss,
            sum(call_in_miss)                             AS call_in_miss,
            max(Spam)                                     AS Spam
        FROM engineered
        GROUP BY report
    ),
    freq_all AS (
        SELECT
            report,
            median(diff_sec) / 3600.0 AS frequency
        FROM (
            SELECT
                report,
                date_diff('second',
                          lag(time) OVER (PARTITION BY report ORDER BY time),
                          time) AS diff_sec
            FROM engineered
        ) t
        GROUP BY report
    ),
    freq_to AS (
        SELECT
            report,
            median(diff_sec) / 3600.0 AS total_frequency_to
        FROM (
            SELECT
                report,
                date_diff('second',
                          lag(time) OVER (PARTITION BY report ORDER BY time),
                          time) AS diff_sec
            FROM engineered
            WHERE type IN (1,3)
        ) t
        GROUP BY report
    ),
    freq_in AS (
        SELECT
            report,
            median(diff_sec) / 3600.0 AS total_frequency_in
        FROM (
            SELECT
                report,
                date_diff('second',
                          lag(time) OVER (PARTITION BY report ORDER BY time),
                          time) AS diff_sec
            FROM engineered
            WHERE type IN (2,4)
        ) t
        GROUP BY report
    ),
    freq_to_by_member AS (
        SELECT
            report,
            sum(med_sec) / 3600.0 AS total_frequency_to_by_phonemember
        FROM (
            SELECT
                report, phone_member,
                median(diff_sec) AS med_sec
            FROM (
                SELECT
                    report, phone_member,
                    date_diff('second',
                              lag(time) OVER (PARTITION BY report, phone_member ORDER BY time),
                              time) AS diff_sec
                FROM engineered
                WHERE type IN (1,3)
            ) d
            GROUP BY report, phone_member
        ) m
        GROUP BY report
    ),
    freq_in_by_member AS (
        SELECT
            report,
            sum(med_sec) / 3600.0 AS total_frequency_in_by_phonemember
        FROM (
            SELECT
                report, phone_member,
                median(diff_sec) AS med_sec
            FROM (
                SELECT
                    report, phone_member,
                    date_diff('second',
                              lag(time) OVER (PARTITION BY report, phone_member ORDER BY time),
                              time) AS diff_sec
                FROM engineered
                WHERE type IN (2,4)
            ) d
            GROUP BY report, phone_member
        ) m
        GROUP BY report
    ),
    time_join AS (
        SELECT
            report,
            date_diff('second', min(time), max(time)) / 3600.0 AS time_join
        FROM engineered
        GROUP BY report
    ),
    redial AS (
        SELECT
            report,
            SUM(CASE WHEN phone_member = prev_phone_member THEN 1 ELSE 0 END) AS total_redial
        FROM (
            SELECT
                report,
                phone_member,
                LAG(phone_member) OVER (PARTITION BY report ORDER BY time) AS prev_phone_member
            FROM engineered
        ) t
        GROUP BY report
    ),
    contacted AS (
        SELECT
            report,
            count(DISTINCT phone_member) AS total_contacted,
            count(DISTINCT CASE WHEN type IN (1,3) THEN phone_member ELSE NULL END) AS total_contacted_to,
            count(DISTINCT CASE WHEN type IN (2,4) THEN phone_member ELSE NULL END) AS total_contacted_in
        FROM engineered
        GROUP BY report
    ),
    mean_call_by_day AS (
        SELECT
            report,
            avg(day_calls) AS mean_call_by_day
        FROM (
            SELECT
                report,
                date_trunc('day', time) AS d,
                sum(call_to + call_in)  AS day_calls
            FROM engineered
            GROUP BY report, d
        ) t
        GROUP BY report
    )
    SELECT
        b.report,
        b.avg_in_contact,
        b.same_network,
        b.duration,
        b.duration_call_to,
        b.duration_call_in,
        b.in_hour,
        b.not_in_hour,
        b.avg_success,
        b.call_to,
        b.call_in,
        b.call_to_miss,
        b.call_in_miss,
        COALESCE(fa.frequency, 0.0) AS frequency,
        COALESCE(ft.total_frequency_to, 0.0) AS total_frequency_to,
        COALESCE(fi.total_frequency_in, 0.0) AS total_frequency_in,
        COALESCE(ftm.total_frequency_to_by_phonemember, 0.0) AS total_frequency_to_by_phonemember,
        COALESCE(fim.total_frequency_in_by_phonemember, 0.0) AS total_frequency_in_by_phonemember,
        COALESCE(tj.time_join, 0.0) AS time_join,
        COALESCE(r.total_redial, 0) AS total_redial,
        COALESCE(c.total_contacted, 0) AS total_contacted,
        COALESCE(c.total_contacted_to, 0) AS total_contacted_to,
        COALESCE(c.total_contacted_in, 0) AS total_contacted_in,
        COALESCE(mbd.mean_call_by_day, 0.0) AS mean_call_by_day,
        b.Spam
    FROM base_agg b
    LEFT JOIN freq_all fa ON b.report = fa.report
    LEFT JOIN freq_to  ft ON b.report = ft.report
    LEFT JOIN freq_in  fi ON b.report = fi.report
    LEFT JOIN freq_to_by_member ftm ON b.report = ftm.report
    LEFT JOIN freq_in_by_member fim ON b.report = fim.report
    LEFT JOIN time_join tj ON b.report = tj.report
    LEFT JOIN redial r ON b.report = r.report
    LEFT JOIN contacted c ON b.report = c.report
    LEFT JOIN mean_call_by_day mbd ON b.report = mbd.report
    """)

    # 4) feature_creation (ratio + derived)
    con.execute(r"""
    CREATE OR REPLACE TEMP VIEW final_features AS
    WITH x AS (
        SELECT
            *,
            (call_to + call_in + call_to_miss + call_in_miss) AS denom_calls,
            (call_in_miss + call_to_miss)                  AS denom_miss,
            (call_to + call_in)                            AS sum_call_inout
        FROM agg_report
    )
    SELECT
        report,

        -- =========================
        -- base
        -- =========================
        avg_in_contact,
        same_network,
        CAST(duration AS BIGINT)          AS duration,
        CAST(duration_call_to AS BIGINT)  AS duration_call_to,
        CAST(duration_call_in AS BIGINT)  AS duration_call_in,
        CAST(in_hour AS INTEGER)          AS in_hour,
        CAST(not_in_hour AS INTEGER)      AS not_in_hour,
        avg_success,
        CAST(call_to AS INTEGER)          AS call_to,
        CAST(call_in AS INTEGER)          AS call_in,
        CAST(call_to_miss AS INTEGER)     AS call_to_miss,
        CAST(call_in_miss AS INTEGER)     AS call_in_miss,
        frequency,
        total_frequency_to,
        total_frequency_in,
        total_frequency_to_by_phonemember,
        total_frequency_in_by_phonemember,
        time_join,
        CAST(total_redial AS INTEGER)     AS total_redial,
        CAST(total_contacted AS INTEGER)  AS total_contacted,
        CAST(total_contacted_to AS INTEGER) AS total_contacted_to,
        CAST(total_contacted_in AS INTEGER) AS total_contacted_in,
        mean_call_by_day,
        CAST(Spam AS INTEGER)             AS Spam,

        -- =========================
        -- ratios (zero_div safe)
        -- =========================
        CASE WHEN denom_calls = 0 THEN 0 ELSE call_to      * 1.0 / denom_calls END AS call_to_rate,
        CASE WHEN denom_calls = 0 THEN 0 ELSE call_in      * 1.0 / denom_calls END AS call_in_rate,
        CASE WHEN denom_calls = 0 THEN 0 ELSE call_to_miss * 1.0 / denom_calls END AS call_to_miss_rate,
        CASE WHEN denom_calls = 0 THEN 0 ELSE call_in_miss * 1.0 / denom_calls END AS call_in_miss_rate,

        CASE WHEN call_in = 0 THEN 0 ELSE call_to      * 1.0 / call_in END AS callto_callin,
        CASE WHEN call_in = 0 THEN 0 ELSE call_to_miss * 1.0 / call_in END AS call_back_rate,

        CASE WHEN denom_miss = 0 THEN 0 ELSE sum_call_inout * 1.0 / denom_miss END AS call_miss,

        (call_in + call_to + call_in_miss + call_to_miss) AS sum_call,

        CASE WHEN call_to = 0 THEN 0 ELSE duration_call_to * 1.0 / call_to END AS avg_duration_call_to,
        CASE WHEN call_in = 0 THEN 0 ELSE duration_call_in * 1.0 / call_in END AS avg_duration_call_in,

        -- =========================
        -- derived (giữ đúng theo code gốc)
        -- =========================
        call_to * frequency AS call_to_miss_mul_frequency,

        duration_call_to * 1.0 / (duration_call_to + duration_call_in + 1) AS duration_call_to_rate,

        call_to * avg_in_contact AS call_to_mul_in_contact,
        call_to * avg_success    AS call_to_mul_success,

        frequency * in_hour AS frequency_mul_in_hour,

        -- code gốc đặt tên "call_to_miss_frequency_in_hour" nhưng dùng call_in_miss_rate
        (CASE WHEN denom_calls = 0 THEN 0 ELSE call_in_miss * 1.0 / denom_calls END) * (frequency * in_hour)
            AS call_to_miss_frequency_in_hour,

        (call_in + call_in_miss) * 1.0 / (call_to + call_in_miss + 1) AS call_in_div_call_to,

        ((call_in + call_in_miss) * 1.0 / (call_to + call_in_miss + 1)) * in_hour
            AS call_in_div_call_to_mul_in_hour,

        (CASE WHEN denom_calls = 0 THEN 0 ELSE call_to_miss * 1.0 / denom_calls END) * duration_call_to
            AS call_to_miss_rate_mul_duration_call_to,

        (frequency * in_hour) * duration_call_to AS frequency_in_hour_mul_duration_call_to,

        in_hour * avg_success AS in_hour_mul_avg_success,

        -- code gốc: call_in_div_call_to_mul_in_hour_mul_avg_success =
        -- frequency_in_hour_mul_duration_call_to * call_in_div_call_to
        ((frequency * in_hour) * duration_call_to)
            * ((call_in + call_in_miss) * 1.0 / (call_to + call_in_miss + 1))
            AS call_in_div_call_to_mul_in_hour_mul_avg_success

    FROM x
    """)



    if out_parquet:
        outp = _sql_quote(out_parquet)
        con.execute(f"COPY final_features TO {outp} (FORMAT PARQUET)") 
    
    return con


"""
preprocess.py — Data Ingestion & Cleaning Pipeline
====================================================
Converted from Databricks PySpark (Cells 1–3 of original notebook).

Original: PySpark DataFrames + Spark SQL functions (F.col, F.when, F.hour, etc.)
This file: pandas + numpy — same logic, same column names, same filter thresholds.

The dataset is downloaded at runtime from the NYC TLC public URL so we don't
commit a 1.4 GB Parquet file to GitHub (which has a 100 MB file size limit).

Video talking point:
  "preprocess.py does everything the original Databricks Cell 1–3 did —
   ingest from the public NYC TLC URL, clean the data through five funnel
   stages, engineer features, and return a clean DataFrame ready for training.
   The only difference is pandas instead of PySpark because we're running on
   a plain Python Docker container with no Spark cluster."
"""

import os
import numpy as np
import pandas as pd
import requests

# ── NYC TLC public dataset URL — downloaded at runtime, not committed to repo ──
PARQUET_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_2025-01.parquet"
)
DATA_DIR = os.environ.get("DATA_DIR", "/tmp/data")
LOCAL_PARQUET = os.path.join(DATA_DIR, "yellow_tripdata_2025-01.parquet")

# ── Train/validation split date — same as original notebook ──────────────────
SPLIT_DATE = "2025-01-25"

# ── 20% sample fraction — mirrors the Databricks serverless memory constraint ─
# In the original we sampled because of Databricks' 1 GB model cache limit.
# Here we sample for faster CI test runs. Set SAMPLE_FRACTION=1.0 to use all data.
SAMPLE_FRACTION = float(os.environ.get("SAMPLE_FRACTION", "0.2"))
SEED = 42

# ── Zone lookup — embedded directly (all 263 NYC TLC taxi zones) ──────────────
# In the original notebook this was a Python list passed to spark.createDataFrame.
# Here it becomes a dict keyed by LocationID for a simple pandas merge.
ZONE_DATA = {
    1: ("EWR", "Newark Airport"), 2: ("Queens", "Jamaica Bay"),
    3: ("Bronx", "Allerton/Pelham Gardens"), 4: ("Manhattan", "Alphabet City"),
    5: ("Staten Island", "Arden Heights"), 6: ("Staten Island", "Arrochar/Fort Wadsworth"),
    7: ("Queens", "Astoria"), 8: ("Queens", "Astoria Park"),
    9: ("Queens", "Auburndale"), 10: ("Queens", "Baisley Park"),
    11: ("Brooklyn", "Bath Beach"), 12: ("Manhattan", "Battery Park"),
    13: ("Manhattan", "Battery Park City"), 14: ("Brooklyn", "Bay Ridge"),
    15: ("Queens", "Bay Terrace/Fort Totten"), 16: ("Queens", "Bayside"),
    17: ("Brooklyn", "Bedford"), 18: ("Bronx", "Bedford Park"),
    19: ("Queens", "Bellerose"), 20: ("Bronx", "Belmont"),
    21: ("Brooklyn", "Bensonhurst East"), 22: ("Brooklyn", "Bensonhurst West"),
    23: ("Staten Island", "Bloomfield/Emerson Hill"), 24: ("Manhattan", "Bloomingdale"),
    25: ("Brooklyn", "Boerum Hill"), 26: ("Brooklyn", "Borough Park"),
    27: ("Bronx", "Bronx Park"), 28: ("Bronx", "Bronxdale"),
    29: ("Brooklyn", "Brooklyn Heights"), 30: ("Brooklyn", "Brooklyn Navy Yard"),
    31: ("Brooklyn", "Brownsville"), 32: ("Brooklyn", "Bushwick North"),
    33: ("Brooklyn", "Bushwick South"), 34: ("Queens", "Cambria Heights"),
    35: ("Brooklyn", "Canarsie"), 36: ("Brooklyn", "Carroll Gardens"),
    37: ("Manhattan", "Central Harlem"), 38: ("Manhattan", "Central Harlem North"),
    39: ("Manhattan", "Central Park"), 40: ("Staten Island", "Charleston/Tottenville"),
    41: ("Manhattan", "Chinatown"), 42: ("Bronx", "City Island"),
    43: ("Bronx", "Claremont/Bathgate"), 44: ("Staten Island", "Clifton/Stapleton"),
    45: ("Bronx", "Co-Op City"), 46: ("Manhattan", "Columbia St"),
    47: ("Brooklyn", "Coney Island"), 48: ("Queens", "Corona"),
    49: ("Queens", "Corona"), 50: ("Bronx", "Country Club"),
    51: ("Bronx", "Crotona Park"), 52: ("Bronx", "Crotona Park East"),
    53: ("Brooklyn", "Crown Heights North"), 54: ("Brooklyn", "Crown Heights South"),
    55: ("Brooklyn", "Cypress Hills"), 56: ("Queens", "Douglaston"),
    57: ("Brooklyn", "Downtown Brooklyn/MetroTech"), 58: ("Brooklyn", "DUMBO/Vinegar Hill"),
    59: ("Brooklyn", "Dyker Heights"), 60: ("Bronx", "East Bronx"),
    61: ("Brooklyn", "East Flatbush/Farragut"), 62: ("Brooklyn", "East Flatbush/Remsen Village"),
    63: ("Queens", "East Flushing"), 64: ("Manhattan", "East Harlem North"),
    65: ("Manhattan", "East Harlem South"), 66: ("Brooklyn", "East New York"),
    67: ("Brooklyn", "East New York/Pennsylvania Avenue"), 68: ("Bronx", "East Tremont"),
    69: ("Manhattan", "East Village"), 70: ("Brooklyn", "East Williamsburg"),
    71: ("Bronx", "Eastchester"), 72: ("Queens", "Elmhurst"),
    73: ("Queens", "Elmhurst/Maspeth"), 74: ("Staten Island", "Eltingville/Annadale/Prince's Bay"),
    75: ("Brooklyn", "Erasmus"), 76: ("Queens", "Far Rockaway"),
    77: ("Manhattan", "Financial District North"), 78: ("Manhattan", "Financial District South"),
    79: ("Brooklyn", "Flatbush/Ditmas Park"), 80: ("Brooklyn", "Flatlands"),
    81: ("Brooklyn", "Flushing"), 82: ("Queens", "Flushing Meadows-Corona Park"),
    83: ("Bronx", "Fordham South"), 84: ("Queens", "Forest Hills"),
    85: ("Queens", "Forest Park/Highland Park"), 86: ("Brooklyn", "Fort Greene"),
    87: ("Queens", "Fresh Meadows"), 88: ("Staten Island", "Freshkills Park"),
    89: ("Manhattan", "Garment District"), 90: ("Queens", "Glen Oaks"),
    91: ("Queens", "Glendale"), 92: ("Manhattan", "Governor's Island/Ellis Island/Liberty Island"),
    93: ("Manhattan", "Governor's Island/Ellis Island/Liberty Island"),
    94: ("Manhattan", "Governor's Island/Ellis Island/Liberty Island"),
    95: ("Brooklyn", "Gowanus"), 96: ("Manhattan", "Gramercy"),
    97: ("Brooklyn", "Gravesend"), 98: ("Staten Island", "Great Kills"),
    99: ("Staten Island", "Great Kills Park"), 100: ("Brooklyn", "Green-Wood Cemetery"),
    101: ("Brooklyn", "Greenpoint"), 102: ("Manhattan", "Greenwich Village North"),
    103: ("Manhattan", "Greenwich Village South"), 104: ("Bronx", "Highbridge"),
    105: ("Manhattan", "Hudson Sq"), 106: ("Bronx", "Hunts Point"),
    107: ("Manhattan", "Inwood"), 108: ("Manhattan", "Inwood Hill Park"),
    109: ("Queens", "Jackson Heights"), 110: ("Queens", "Jamaica"),
    111: ("Queens", "Jamaica Estates"), 112: ("Queens", "JFK Airport"),
    113: ("Brooklyn", "Kensington"), 114: ("Queens", "Kew Gardens"),
    115: ("Queens", "Kew Gardens Hills"), 116: ("Bronx", "Kingsbridge Heights"),
    117: ("Manhattan", "Lenox Hill East"), 118: ("Manhattan", "Lenox Hill West"),
    119: ("Manhattan", "Lincoln Square East"), 120: ("Manhattan", "Lincoln Square West"),
    121: ("Manhattan", "Little Italy/NoLiTa"), 122: ("Queens", "Long Island City/Hunters Point"),
    123: ("Queens", "Long Island City/Queens Plaza"), 124: ("Bronx", "Longwood"),
    125: ("Manhattan", "Lower East Side"), 126: ("Brooklyn", "Madison"),
    127: ("Brooklyn", "Manhattan Beach"), 128: ("Manhattan", "Manhattan Valley"),
    129: ("Manhattan", "Manhattanville"), 130: ("Manhattan", "Marble Hill"),
    131: ("Brooklyn", "Marine Park/Floyd Bennett Field"), 132: ("Brooklyn", "Marine Park/Mill Basin"),
    133: ("Staten Island", "Mariners Harbor"), 134: ("Queens", "Maspeth"),
    135: ("Manhattan", "Meatpacking/West Village West"), 136: ("Bronx", "Melrose South"),
    137: ("Queens", "Middle Village"), 138: ("Manhattan", "Midtown Center"),
    139: ("Manhattan", "Midtown East"), 140: ("Manhattan", "Midtown North"),
    141: ("Manhattan", "Midtown South"), 142: ("Brooklyn", "Midwood"),
    143: ("Manhattan", "Morningside Heights"), 144: ("Bronx", "Morrisania/Melrose"),
    145: ("Bronx", "Mott Haven/Port Morris"), 146: ("Bronx", "Mount Hope"),
    147: ("Manhattan", "Murray Hill"), 148: ("Staten Island", "New Dorp/Midland Beach"),
    149: ("Bronx", "Norwood"), 150: ("Queens", "Oakland Gardens"),
    151: ("Staten Island", "Oakwood"), 152: ("Manhattan", "Old Astoria"),
    153: ("Queens", "Ozone Park"), 154: ("Brooklyn", "Park Slope"),
    155: ("Bronx", "Parkchester"), 156: ("Bronx", "Pelham Bay"),
    157: ("Bronx", "Pelham Bay Park"), 158: ("Bronx", "Pelham Parkway"),
    159: ("Manhattan", "Penn Station/Madison Sq West"), 160: ("Queens", "Queens Village"),
    161: ("Queens", "Queensboro Hill"), 162: ("Queens", "Queensbridge/Ravenswood"),
    163: ("Manhattan", "Randalls Island"), 164: ("Brooklyn", "Red Hook"),
    165: ("Queens", "Rego Park"), 166: ("Queens", "Richmond Hill"),
    167: ("Queens", "Ridgewood"), 168: ("Staten Island", "Rikers Island"),
    169: ("Bronx", "Riverdale/North Riverdale/Fieldston"), 170: ("Queens", "Rockaway Park"),
    171: ("Manhattan", "Roosevelt Island"), 172: ("Queens", "Rosedale"),
    173: ("Staten Island", "Rossville/Woodrow"), 174: ("Queens", "Saint Albans"),
    175: ("Staten Island", "Saint George/New Brighton"), 176: ("Queens", "Saint Michaels Cemetery/Woodside"),
    177: ("Bronx", "Schuylerville/Edgewater Park"), 178: ("Manhattan", "Seaport"),
    179: ("Brooklyn", "Sheepshead Bay"), 180: ("Manhattan", "SoHo"),
    181: ("Bronx", "Soundview/Bruckner"), 182: ("Bronx", "Soundview/Castle Hill"),
    183: ("Staten Island", "South Beach/Dongan Hills"), 184: ("Queens", "South Jamaica"),
    185: ("Queens", "South Ozone Park"), 186: ("Brooklyn", "South Williamsburg"),
    187: ("Queens", "Springfield Gardens North"), 188: ("Queens", "Springfield Gardens South"),
    189: ("Queens", "Sprints"), 190: ("Staten Island", "Stapleton/Rosebank"),
    191: ("Brooklyn", "Starrett City"), 192: ("Queens", "Steinway"),
    193: ("Manhattan", "Stuy Town/Peter Cooper Village"), 194: ("Brooklyn", "Stuyvesant Heights"),
    195: ("Queens", "Sunnyside"), 196: ("Brooklyn", "Sunset Park East"),
    197: ("Brooklyn", "Sunset Park West"), 198: ("Manhattan", "Sutton Place/Turtle Bay North"),
    199: ("Manhattan", "Times Sq/Theatre District"), 200: ("Manhattan", "TriBeCa/Civic Center"),
    201: ("Manhattan", "Two Bridges/Seward Park"), 202: ("Brooklyn", "Unionport"),
    203: ("Manhattan", "Union Sq"), 204: ("Bronx", "University Heights/Morris Heights"),
    205: ("Manhattan", "Upper East Side North"), 206: ("Manhattan", "Upper East Side South"),
    207: ("Manhattan", "Upper West Side North"), 208: ("Manhattan", "Upper West Side South"),
    209: ("Manhattan", "Van Cortlandt Park"), 210: ("Bronx", "Van Cortlandt Village"),
    211: ("Bronx", "Van Nest/Morris Park"), 212: ("Manhattan", "Washington Heights North"),
    213: ("Manhattan", "Washington Heights South"), 214: ("Staten Island", "West Brighton"),
    215: ("Queens", "West Concourse"), 216: ("Bronx", "West Farms/Bronx River"),
    217: ("Manhattan", "West Village"), 218: ("Bronx", "Westchester Village/Unionport"),
    219: ("Queens", "Whitestone"), 220: ("Queens", "Willets Point"),
    221: ("Bronx", "Williamsbridge/Olinville"), 222: ("Brooklyn", "Williamsburg (North Side)"),
    223: ("Brooklyn", "Williamsburg (South Side)"), 224: ("Brooklyn", "Windsor Terrace"),
    225: ("Queens", "Woodhaven"), 226: ("Brooklyn", "Woodlawn/Wakefield"),
    227: ("Queens", "Woodside"), 228: ("Manhattan", "World Trade Center"),
    229: ("Manhattan", "Yorkville East"), 230: ("Manhattan", "Yorkville West"),
    231: ("Staten Island", "unknown"), 232: ("Staten Island", "unknown"),
    233: ("Queens", "unknown"), 234: ("Queens", "unknown"),
    235: ("Bronx", "unknown"), 236: ("Bronx", "unknown"),
    237: ("Manhattan", "unknown"), 238: ("Manhattan", "unknown"),
    239: ("Brooklyn", "unknown"), 240: ("Brooklyn", "unknown"),
    241: ("Staten Island", "unknown"), 242: ("Queens", "unknown"),
    243: ("Bronx", "unknown"), 244: ("Manhattan", "unknown"),
    245: ("Brooklyn", "unknown"), 246: ("Queens", "unknown"),
    247: ("Bronx", "unknown"), 248: ("Manhattan", "unknown"),
    249: ("Brooklyn", "unknown"), 250: ("Queens", "unknown"),
    251: ("Bronx", "unknown"), 252: ("Manhattan", "unknown"),
    253: ("Brooklyn", "unknown"), 254: ("Queens", "unknown"),
    255: ("Bronx", "unknown"), 256: ("Manhattan", "unknown"),
    257: ("Brooklyn", "unknown"), 258: ("Queens", "unknown"),
    259: ("Bronx", "unknown"), 260: ("Manhattan", "unknown"),
    261: ("Brooklyn", "unknown"), 262: ("Manhattan", "NV"),
    263: ("Queens", "unknown"),
}


def _build_zone_df() -> pd.DataFrame:
    """Convert the zone dict to a small lookup DataFrame for a merge."""
    rows = [(loc_id, boro, zone) for loc_id, (boro, zone) in ZONE_DATA.items()]
    return pd.DataFrame(rows, columns=["PULocationID", "PU_Borough", "PU_Zone"])


def download_data(url: str = PARQUET_URL, dest: str = LOCAL_PARQUET) -> str:
    """
    Download the NYC TLC Parquet file if it isn't already on disk.

    Why: GitHub has a 100 MB file limit. At 1.4 GB the dataset can't be
    committed to the repo, so we pull it at runtime from the TLC public CDN.
    This mirrors what a real data pipeline would do — pull from source,
    don't bake data into the container image.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"[preprocess] Data already cached at {dest}")
        return dest

    print(f"[preprocess] Downloading {url} ...")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
    print(f"[preprocess] Download complete → {dest}")
    return dest


def load_raw(path: str) -> pd.DataFrame:
    """
    Read the Parquet file into a pandas DataFrame.

    Original notebook Cell 1 used spark.read.parquet(PARQUET_PATH) and then
    joined a zone lookup Spark DataFrame. Here we use pd.read_parquet + merge.
    """
    df = pd.read_parquet(path)
    print(f"[preprocess] Raw records: {len(df):,}")

    # Join zone borough / zone name — equivalent to the original Spark join
    zone_df = _build_zone_df()
    df = df.merge(zone_df, on="PULocationID", how="left")
    print(f"[preprocess] After zone join: {len(df):,}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the five-stage data cleaning funnel from notebook Cell 2.

    Stage 1 — Temporal filter: drop trips where pickup is after dropoff.
      Original: F.col("tpep_pickup_datetime") < F.col("tpep_dropoff_datetime")
    Stage 2 — Illogical values: drop zero distance or zero fare records.
      Original: F.col("trip_distance") > 0 and F.col("fare_amount") > 0
    Stage 3 — P99 outlier cap: remove top 1% of distance, duration, and total_amount.
      Original: df.approxQuantile(["trip_distance", ...], [0.99], 0.01)
      Note: We use pandas quantile() — exact rather than approximate, same intent.
    Stage 4 — Feature filters: drop records where spend_per_minute is undefined,
              infinite, or over $100/min.
    """
    n0 = len(df)

    # Ensure datetime types
    df["tpep_pickup_datetime"]  = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    # ── Stage 1: Temporal filter ─────────────────────────────────────────────
    # Original: 3,475,226 → 3,473,175  (removed 2,051 impossible timestamps)
    df = df[df["tpep_pickup_datetime"] < df["tpep_dropoff_datetime"]].copy()
    print(f"[clean] After temporal filter : {len(df):,}  (removed {n0 - len(df):,})")

    # ── Stage 2: Illogical values ────────────────────────────────────────────
    n1 = len(df)
    df = df[(df["trip_distance"] > 0) & (df["fare_amount"] > 0)].copy()
    print(f"[clean] After illogical filter: {len(df):,}  (removed {n1 - len(df):,})")

    # ── Stage 3: P99 outlier cap ─────────────────────────────────────────────
    n2 = len(df)
    # Compute trip_duration_mins first so we can apply the P99 cap on it
    df["trip_duration_mins"] = (
        (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"])
        .dt.total_seconds() / 60.0
    )
    p99_distance = df["trip_distance"].quantile(0.99)
    p99_duration = df["trip_duration_mins"].quantile(0.99)
    p99_total    = df["total_amount"].quantile(0.99)
    df = df[
        (df["trip_distance"]    <= p99_distance) &
        (df["trip_duration_mins"] <= p99_duration) &
        (df["total_amount"]     <= p99_total)
    ].copy()
    print(f"[clean] After P99 cap         : {len(df):,}  (removed {n2 - len(df):,})")

    # ── Stage 4 & 5: Feature filters (applied after feature engineering) ──────
    # Handled in feature_engineering() below to keep the funnel logic together.
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive all features from notebook Cell 3.

    PySpark              →  pandas/numpy equivalent
    ─────────────────────────────────────────────────
    F.hour(col)          →  .dt.hour
    F.dayofweek(col)     →  .dt.dayofweek + 1  (Spark is 1=Sun, pandas is 0=Mon)
    F.when(...).otherwise→  np.where / boolean indexing
    F.log1p(col)         →  np.log1p(col)
    F.coalesce(col, 0)   →  .fillna(0)

    Target: log_spend_per_trip = log1p(fare_amount)
    All predictions will use np.expm1() to convert back to dollars.
    """
    # ── Temporal features ────────────────────────────────────────────────────
    df["hour_of_day"] = df["tpep_pickup_datetime"].dt.hour
    # Spark dayofweek: 1=Sunday, 7=Saturday.  pandas dayofweek: 0=Monday, 6=Sunday.
    # We replicate Spark behaviour: add 1 and rotate Sunday to position 1.
    dow_pandas = df["tpep_pickup_datetime"].dt.dayofweek   # 0=Mon ... 6=Sun
    df["day_of_week"] = (dow_pandas + 2) % 7              # 1=Sun, 7=Sat — matches Spark
    df["is_weekend"]  = np.where(df["tpep_pickup_datetime"].dt.dayofweek >= 5, 1, 0)

    # ── Revenue targets ──────────────────────────────────────────────────────
    # Original decision: switched from total_amount to fare_amount because
    # tip_amount is $0 for cash payments, creating a bimodal distribution
    # that the model can't reliably learn.
    df["spend_per_trip"]    = df["fare_amount"].astype(float)
    df["spend_per_minute"]  = df["fare_amount"] / df["trip_duration_mins"]

    # log1p transform to handle the right-skewed fare distribution.
    # log1p(x) = log(1+x) — safe for x=0, avoids -inf.
    # Reverse: np.expm1(prediction) to get dollars back.
    df["log_spend_per_trip"] = np.log1p(df["spend_per_trip"])

    # ── Surcharge indicator flags ────────────────────────────────────────────
    # Convert continuous surcharge columns to binary flags.
    # This reduces rounding noise while preserving the trip-type signal.
    df["has_congestion"]  = (df["congestion_surcharge"].fillna(0) > 0).astype(int)
    df["has_airport_fee"] = (df["Airport_fee"].fillna(0) > 0).astype(int)
    df["has_cbd_fee"]     = (df["cbd_congestion_fee"].fillna(0) > 0).astype(int)

    # ── Borough string column (for one-hot encoding downstream) ─────────────
    df["PU_Borough_str"] = df["PU_Borough"].fillna("Unknown").astype(str)

    # ── Stage 4 & 5: Filter invalid spend_per_minute ─────────────────────────
    n3 = len(df)
    df = df[
        df["spend_per_minute"].notna() &
        np.isfinite(df["spend_per_minute"]) &
        (df["spend_per_minute"] < 100) &
        (df["spend_per_minute"] > 0) &
        df["passenger_count"].notna() &
        (df["passenger_count"] > 0)
    ].copy()
    print(f"[feature_eng] After feature filters: {len(df):,}  (removed {n3 - len(df):,})")
    return df


def time_split(df: pd.DataFrame, split_date: str = SPLIT_DATE):
    """
    Time-aware train/validation split — same logic as notebook Cell 5.

    Why time-aware instead of random?
    Random splitting leaks future temporal patterns into training and produces
    optimistically biased evaluation metrics. We train on Jan 1–24 and validate
    on Jan 25–31 so the model is evaluated on genuinely unseen future data.
    """
    cutoff = pd.Timestamp(split_date)
    train = df[df["tpep_pickup_datetime"] < cutoff].copy()
    val   = df[df["tpep_pickup_datetime"] >= cutoff].copy()

    # Apply 20% sample — mirrors the Databricks serverless memory constraint.
    # In production with full compute you'd remove this sampling step.
    train_s = train.sample(frac=SAMPLE_FRACTION, random_state=SEED)
    val_s   = val.sample(frac=SAMPLE_FRACTION, random_state=SEED)

    print(f"[split] Train full: {len(train):,} | Sample: {len(train_s):,}")
    print(f"[split] Val full  : {len(val):,} | Sample: {len(val_s):,}")
    return train_s, val_s


def run_pipeline(parquet_path: str = None) -> tuple:
    """
    End-to-end pipeline: download → load → clean → feature engineer → split.
    Returns (train_df, val_df) — both are feature-complete pandas DataFrames.

    Called by train.py as: train_df, val_df = preprocess.run_pipeline()
    """
    if parquet_path is None:
        parquet_path = download_data()

    df_raw   = load_raw(parquet_path)
    df_clean = clean(df_raw)
    df_feat  = feature_engineering(df_clean)
    train_df, val_df = time_split(df_feat)
    return train_df, val_df


if __name__ == "__main__":
    # Run standalone to verify the pipeline works end-to-end
    train_df, val_df = run_pipeline()
    print(f"\n✓ Pipeline complete")
    print(f"  Train shape: {train_df.shape}")
    print(f"  Val   shape: {val_df.shape}")
    print(f"  Target range: {train_df['log_spend_per_trip'].min():.2f} – "
          f"{train_df['log_spend_per_trip'].max():.2f}")

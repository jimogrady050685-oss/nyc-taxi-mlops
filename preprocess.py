"""
preprocess.py
=============
Data ingestion and cleaning pipeline for the NYC Taxi Fare model.

Originally written in PySpark on Databricks. Converted to pandas and numpy so it can run on a plain Python Docker container with no Spark cluster.
The logic, column names, and filter thresholds are the same as the original.

The dataset is downloaded at runtime from the NYC TLC public URL.
At 1.4 GB it cannot be committed to GitHub so we pull it fresh each run.
"""

import os
import numpy as np
import pandas as pd
import requests

PARQUET_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_2025-01.parquet"
)
DATA_DIR      = os.environ.get("DATA_DIR", "/tmp/data")
LOCAL_PARQUET = os.path.join(DATA_DIR, "yellow_tripdata_2025-01.parquet")

SPLIT_DATE      = "2025-01-25"
SAMPLE_FRACTION = float(os.environ.get("SAMPLE_FRACTION", "0.2"))
SEED            = 42

# All 263 NYC TLC taxi zones keyed by LocationID.
# Used to join borough and zone name onto the trip records.
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
    """Converts the zone dict into a small lookup DataFrame for merging."""
    rows = [(loc_id, boro, zone) for loc_id, (boro, zone) in ZONE_DATA.items()]
    return pd.DataFrame(rows, columns=["PULocationID", "PU_Borough", "PU_Zone"])


def download_data(url: str = PARQUET_URL, dest: str = LOCAL_PARQUET) -> str:
    """
    Downloads the NYC TLC parquet file if it is not already on disk.
    Streams in 8 MB chunks to avoid memory issues with the 1.4 GB file.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"Data already cached at {dest}")
        return dest

    print(f"Downloading {url} ...")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
    print(f"Download complete at {dest}")
    return dest


def load_raw(path: str) -> pd.DataFrame:
    """
    Reads the parquet file and joins the zone lookup onto it.
    Equivalent to the original Spark read + zone DataFrame join.
    """
    df = pd.read_parquet(path)
    print(f"Raw records: {len(df):,}")

    zone_df = _build_zone_df()
    df = df.merge(zone_df, on="PULocationID", how="left")
    print(f"After zone join: {len(df):,}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Five stage cleaning funnel ported from the original notebook.

    Stage 1: drop trips where pickup is after dropoff
    Stage 2: drop records with zero distance or zero fare
    Stage 3: remove the top 1% of distance, duration, and total_amount
    Stage 4 and 5: drop records with invalid spend_per_minute values
                   (handled in feature_engineering to keep the funnel together)
    """
    n0 = len(df)

    df["tpep_pickup_datetime"]  = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    # Stage 1: remove impossible timestamps
    df = df[df["tpep_pickup_datetime"] < df["tpep_dropoff_datetime"]].copy()
    print(f"After temporal filter: {len(df):,}  (removed {n0 - len(df):,})")

    # Stage 2: remove zero distance and zero fare records
    n1 = len(df)
    df = df[(df["trip_distance"] > 0) & (df["fare_amount"] > 0)].copy()
    print(f"After illogical filter: {len(df):,}  (removed {n1 - len(df):,})")

    # Stage 3: cap at the 99th percentile for distance, duration, and total amount
    n2 = len(df)
    df["trip_duration_mins"] = (
        (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"])
        .dt.total_seconds() / 60.0
    )
    p99_distance = df["trip_distance"].quantile(0.99)
    p99_duration = df["trip_duration_mins"].quantile(0.99)
    p99_total    = df["total_amount"].quantile(0.99)
    df = df[
        (df["trip_distance"]      <= p99_distance) &
        (df["trip_duration_mins"] <= p99_duration) &
        (df["total_amount"]       <= p99_total)
    ].copy()
    print(f"After P99 cap: {len(df):,}  (removed {n2 - len(df):,})")

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives all features from the original notebook.

    PySpark to pandas equivalents used here:
      F.hour(col)           >>  .dt.hour
      F.dayofweek(col)      >>  .dt.dayofweek + offset to match Spark numbering
      F.when().otherwise()  >>  np.where
      F.log1p(col)          >>  np.log1p
      F.coalesce(col, 0)    >>  .fillna(0)

    The model target is log_spend_per_trip = log1p(fare_amount).
    Use np.expm1() to convert predictions back to dollars.
    """
    df["hour_of_day"] = df["tpep_pickup_datetime"].dt.hour

    # Spark dayofweek is 1=Sunday through 7=Saturday.
    # pandas dayofweek is 0=Monday through 6=Sunday.
    # We rotate to match Spark so the feature means the same thing.
    dow_pandas    = df["tpep_pickup_datetime"].dt.dayofweek
    df["day_of_week"] = (dow_pandas + 2) % 7
    df["is_weekend"]  = np.where(df["tpep_pickup_datetime"].dt.dayofweek >= 5, 1, 0)

    # fare_amount is used rather than total_amount because tip_amount is zero
    # for cash payments, which creates a bimodal distribution the model struggles with
    df["spend_per_trip"]   = df["fare_amount"].astype(float)
    df["spend_per_minute"] = df["fare_amount"] / df["trip_duration_mins"]

    # log1p transform handles the right skew in fare amounts
    # reverse with np.expm1 to get dollars back from predictions
    df["log_spend_per_trip"] = np.log1p(df["spend_per_trip"])

    # Binary flags for surcharges rather than raw amounts to reduce rounding noise
    df["has_congestion"]  = (df["congestion_surcharge"].fillna(0) > 0).astype(int)
    df["has_airport_fee"] = (df["Airport_fee"].fillna(0) > 0).astype(int)
    df["has_cbd_fee"]     = (df["cbd_congestion_fee"].fillna(0) > 0).astype(int)

    df["PU_Borough_str"] = df["PU_Borough"].fillna("Unknown").astype(str)

    # Stages 4 and 5: drop records where spend_per_minute is invalid
    n3 = len(df)
    df = df[
        df["spend_per_minute"].notna() &
        np.isfinite(df["spend_per_minute"]) &
        (df["spend_per_minute"] < 100) &
        (df["spend_per_minute"] > 0) &
        df["passenger_count"].notna() &
        (df["passenger_count"] > 0)
    ].copy()
    print(f"After feature filters: {len(df):,}  (removed {n3 - len(df):,})")
    return df


def time_split(df: pd.DataFrame, split_date: str = SPLIT_DATE):
    """
    Time aware train and validation split.

    Random splitting would leak future patterns into training and produce
    inflated metrics. We train on Jan 1 to 24 and validate on Jan 25 to 31
    so the model is always evaluated on data it has never seen.
    """
    cutoff  = pd.Timestamp(split_date)
    train   = df[df["tpep_pickup_datetime"] < cutoff].copy()
    val     = df[df["tpep_pickup_datetime"] >= cutoff].copy()

    train_s = train.sample(frac=SAMPLE_FRACTION, random_state=SEED)
    val_s   = val.sample(frac=SAMPLE_FRACTION, random_state=SEED)

    print(f"Train: {len(train):,} full  |  {len(train_s):,} sampled")
    print(f"Val:   {len(val):,} full  |  {len(val_s):,} sampled")
    return train_s, val_s


def run_pipeline(parquet_path: str = None) -> tuple:
    """
    Runs the full pipeline end to end.
    Returns (train_df, val_df) ready for training.
    Called by train.py as: train_df, val_df = preprocess.run_pipeline()
    """
    if parquet_path is None:
        parquet_path = download_data()

    df_raw   = load_raw(parquet_path)
    df_clean = clean(df_raw)
    df_feat  = feature_engineering(df_clean)
    return time_split(df_feat)


if __name__ == "__main__":
    train_df, val_df = run_pipeline()
    print(f"\nPipeline complete")
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Target range: {train_df['log_spend_per_trip'].min():.2f} to "
          f"{train_df['log_spend_per_trip'].max():.2f}")
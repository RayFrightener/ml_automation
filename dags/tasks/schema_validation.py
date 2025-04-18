#!/usr/bin/env python3
"""
dags/tasks/schema_validation.py

Validate the post‑preprocessed DataFrame against a Pandera schema,
then snapshot the column dtypes to S3 for versioning.
"""

import json
import time
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from utils.storage import upload       # your retrying S3 helper
from utils.config import S3_BUCKET     # from the new config.py

# Define the expected schema for your processed data
homeowner_schema = DataFrameSchema({
    "eey":           Column(float, Check.ge(0)),
    "il_total":      Column(float, Check.ge(0)),
    "pure_premium":  Column(float, Check.ge(0)),
    # … add all other columns you expect here …
})

def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raises a SchemaError if the DataFrame doesn't match.
    Returns the validated DataFrame otherwise.
    """
    return homeowner_schema.validate(df)

def snapshot_schema(df: pd.DataFrame) -> None:
    """
    Take a snapshot of the DataFrame's dtypes, write to /tmp,
    and upload to S3 under 'schema_snapshots/' with a timestamp.
    """
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    schema_map = df.dtypes.apply(lambda dt: dt.name).to_dict()
    local_path = f"/tmp/schema_{ts}.json"

    with open(local_path, "w") as f:
        json.dump(schema_map, f)

    s3_key = f"schema_snapshots/schema_{ts}.json"
    upload(local_path, s3_key)

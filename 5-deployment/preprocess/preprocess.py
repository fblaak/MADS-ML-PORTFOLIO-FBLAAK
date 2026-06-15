from pathlib import Path

import pandas as pd
from loguru import logger


def preprocess_names(input_file: Path, output_file: Path) -> None:
    logger.info(f"Reading raw data from {input_file}")

    df = pd.read_csv(input_file)

    df["name"] = df["name"].astype(str).str.strip()
    df["name_lower"] = df["name"].str.lower()
    df["name_length"] = df["name"].str.len()
    df["first_letter"] = df["name"].str[0].str.upper()

    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).str.upper()

    df = df.dropna(subset=["name"])
    df = df[df["name_length"] > 1]
    df = df.reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing processed data to {output_file}")
    df.to_parquet(output_file, index=False)


if __name__ == "__main__":
    input_file = Path("data/raw/baby_names.csv")
    output_file = Path("data/processed/baby_names.parquet")

    preprocess_names(input_file, output_file)
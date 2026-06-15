from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from loguru import logger


DATA_FILE = Path("data/processed/baby_names.parquet")
OUTPUT_DIR = Path("data/output")
OUTPUT_FILE = OUTPUT_DIR / "name_length_distribution.png"


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        st.error(f"Processed data file not found: {DATA_FILE}")
        st.stop()

    return pd.read_parquet(DATA_FILE)


def create_name_length_plot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))

    df["name_length"].value_counts().sort_index().plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Distribution of baby name lengths")
    ax.set_xlabel("Name length")
    ax.set_ylabel("Number of names")

    return fig


def save_plot(fig) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    logger.info(f"Saved plot to {OUTPUT_FILE}")


st.title("Baby Names Analysis")

st.write(
    "This Streamlit app loads the processed baby names dataset, "
    "runs a simple analysis and visualizes the results."
)

if st.button("Run model / analysis"):
    df = load_data()

    st.subheader("Preview of processed data")
    st.dataframe(df.head())

    st.subheader("Basic statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Number of rows", len(df))
    col2.metric("Unique names", df["name"].nunique())
    col3.metric("Average name length", round(df["name_length"].mean(), 2))

    st.subheader("Name length distribution")

    fig = create_name_length_plot(df)
    save_plot(fig)

    st.pyplot(fig)

    st.success(f"Analysis completed. Plot saved to {OUTPUT_FILE}")
import requests
from pathlib import Path
from loguru import logger


def download(url, datafile: Path):
    datadir = datafile.parent

    if not datadir.exists():
        logger.info(f"Creating directory {datadir}")
        datadir.mkdir(parents=True)

    if not datafile.exists():
        logger.info(f"Downloading {url} to {datafile}")

        response = requests.get(url)
        response.raise_for_status()

        with datafile.open("wb") as f:
            f.write(response.content)

    else:
        logger.info(f"File {datafile} already exists, skipping download")


url = "https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv"

datadir = Path("data/raw")
datafile = datadir / "baby_names.csv"

download(url, datafile)
"""Load a snapshot and create a meadow dataset.

Import population data from Gapminder. Very little processing is done.

More details at https://www.gapminder.org/data/documentation/gd003/.
"""

import pandas as pd
from owid.catalog import Table
from structlog import get_logger

from etl.helpers import PathFinder, create_dataset
from etl.snapshot import Snapshot

# Initialize logger.
log = get_logger()

# Get paths and naming conventions for current step.
paths = PathFinder(__file__)


def run(dest_dir: str) -> None:
    log.info("population.start")

    #
    # Load inputs.
    #
    # Retrieve snapshot.
    snap: Snapshot = paths.load_dependency("population.xlsx")

    # Load data from snapshot.
    df = pd.read_excel(
        snap.path,
        sheet_name="data-for-countries-etc-by-year",
        usecols=["name", "time", "Population"],
    ).rename(columns={"name": "country", "time": "year", "Population": "population"})
    df = df.set_index(["country", "year"], verify_integrity=True)
    #
    # Process data.
    #
    # Create a new table and ensure all columns are snake-case.
    tb = Table(df, short_name=paths.short_name, underscore=True)

    #
    # Save outputs.
    #
    # Create a new meadow dataset with the same metadata as the snapshot.
    ds_meadow = create_dataset(dest_dir, tables=[tb], default_metadata=snap.metadata)

    # Save changes in the new garden dataset.
    ds_meadow.save()

    log.info("population.end")

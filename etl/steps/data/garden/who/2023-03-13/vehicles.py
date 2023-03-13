"""Load a meadow dataset and create a garden dataset."""

import pandas as pd
from owid.catalog import Dataset, Table
from structlog import get_logger

from etl.data_helpers import geo
from etl.helpers import PathFinder, create_dataset

log = get_logger()

# Get paths and naming conventions for current step.
paths = PathFinder(__file__)


def run(dest_dir: str) -> None:
    log.info("vehicles.start")

    # Load population.
    population_table: Table = paths.load_dependency("key_indicators")["population"]
    pop = population_table["population"]

    # Load registered vehicles.
    short_name = "dataset_5676_global_health_observatory__world_health_organization__2022_08"
    who_gh: Table = paths.load_dependency(short_name)[short_name]

    # Use the same index as popoulation
    registered_vehicles = who_gh["indicator__number_of_registered_vehicles"].dropna()
    registered_vehicles = (
        registered_vehicles.reset_index()
        .rename(columns={"entity_name": "country"})
        .set_index(["country", "year"])["indicator__number_of_registered_vehicles"]
    )

    df = pd.DataFrame({"registered_vehicles_per_person": (registered_vehicles / pop).dropna()})

    tb_garden = Table(df, short_name="vehicles")

    #
    # Save outputs.
    #
    # Create a new garden dataset with the same metadata as the meadow dataset.
    ds_garden = create_dataset(dest_dir, tables=[tb_garden])

    # Save changes in the new garden dataset.
    ds_garden.save()

    log.info("vehicles.end")

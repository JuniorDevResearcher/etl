#
#  __init__.py
#  owid/latest/population_density
#

"""
Adapted from Ed's population density script:

https://github.com/owid/notebooks/blob/main/EdouardMathieu/omm_population_density/script.py
"""

import pandas as pd

from owid.catalog import Dataset, DatasetMeta, Table
from owid import catalog

from etl.paths import DATA_DIR


KEY_INDICATORS = DATA_DIR / "garden/owid/latest/key_indicators"


def load_population() -> Table:
    return Dataset(KEY_INDICATORS)["population"].reset_index()


def load_land_area() -> Table:
    d = Dataset(
        DATA_DIR
        / "meadow/open_numbers/latest/open_numbers__world_development_indicators"
    )
    table = d["ag_lnd_totl_k2"]

    table = table.reset_index()

    # harmonize country names
    # TODO: fix this
    reference_dataset = catalog.Dataset(DATA_DIR / "reference")
    countries_regions = reference_dataset["countries_regions"]

    geo_codes = table.geo.str.upper()

    table["country"] = geo_codes.map(countries_regions["name"])

    # TODO: what about world? should we map it first to countries and then aggregate or use
    # `world` land area from worldbank?
    table = table.dropna(subset=["country"])

    table = table.rename(
        columns={
            "time": "year",
            "ag_lnd_totl_k2": "land_area",
        }
    )[["country", "year", "land_area"]]

    return table


def make_table() -> Table:
    population = load_population()
    land_area = load_land_area()

    population = population.loc[population.country == "Singapore"]
    land_area = land_area.loc[land_area.country == "Singapore"]

    # take the latest measurement of land area
    land_area = (
        land_area.sort_values("year").groupby(["country"], as_index=False).last()
    )

    # TODO: is merge gonna keep Table type? if not, fix just given country column
    df = pd.merge(
        land_area[["country", "land_area"]],
        population[["country", "population", "year"]],
        on="country",
        validate="one_to_many",
    )

    df = (
        df.assign(
            population_density=(df.population / df.land_area)
            .round(3)
            .rename("population_density")
        )
        .drop(columns=["population", "land_area"])
        .sort_values(["country", "year"])
    )

    df.metadata.short_name = "population-density"
    df.metadata.description = "Population density (World Bank, Gapminder, HYDE & UN)"
    return df


def run(dest_dir: str) -> None:
    ds = Dataset.create_empty(dest_dir)
    ds.metadata = DatasetMeta(
        namespace="owid",
        short_name="population_density",
        description="Population density (World Bank, Gapminder, HYDE & UN)",
    )
    ds.save()

    t = make_table()
    ds.add(t)

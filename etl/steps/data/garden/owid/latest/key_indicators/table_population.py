#
#  table_population.py
#  key_indicators
#

"""
Adapted from Ed's importers script:

https://github.com/owid/importers/blob/master/population/etl.py
"""

from pathlib import Path
import json
from typing import cast, List

import pandas as pd

from owid.catalog import Dataset, Table
from etl.paths import DATA_DIR

UNWPP = DATA_DIR / "meadow/wpp/2019/standard_projections"
GAPMINDER = DATA_DIR / "meadow/gapminder/2019-12-10/population"
HYDE = DATA_DIR / "meadow/hyde/2017/baseline"
REFERENCE = DATA_DIR / "reference"

COUNTRY_MAPPING = Path(__file__).with_suffix(".mapping.csv")

DIR_PATH = Path(__file__).parent


def make_table() -> Table:
    t = (
        make_combined()
        .pipe(rename_entities)
        .pipe(select_source)
        .pipe(calculate_aggregates)
        .pipe(prepare_dataset)
    )
    _validate(t)
    t.metadata.short_name = "population"
    return t


def _compare(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    assert a.index.names == b.index.names
    index_names = a.index.names
    a = a.reset_index()
    b = b.reset_index()

    a_keys = set(a[index_names].itertuples(index=False, name=None))
    b_keys = set(b[index_names].itertuples(index=False, name=None))

    if a_keys - b_keys:
        raise AssertionError(f"{a_keys - b_keys}")
    if b_keys - a_keys:
        raise AssertionError(f"{b_keys - a_keys}")

    return a.compare(b)


def _validate(t: Table) -> None:
    """Is the new population data identical to a version from importers?"""
    COLS_MAP = {
        "Entity": "country",
        "Year": "year",
        "Population (historical estimates and future projections)": "population",
        "Share of world population": "world_pop_share",
    }
    # transform it into new shape
    old_df = (
        pd.read_csv(DIR_PATH / "output" / "Population (Gapminder, HYDE & UN).csv")
        .rename(columns=COLS_MAP)[COLS_MAP.values()]
        .set_index(["country", "year"])
    )

    # TODO: South America is not yet ready, fix this after https://github.com/owid/etl/issues/92
    # gets resolved
    old_df = old_df.drop(["South America"])

    # we have newer Hyde dataset which contains more history for some countries
    # TODO: we are missing `South America`
    t = t[t.index.isin(old_df.index)]

    # TODO: continent population is significantly different (e.g. Europe)
    # TODO: validate comparison
    _compare(pd.DataFrame(t), old_df)


def rename_entities(df: pd.DataFrame) -> pd.DataFrame:
    mapping = (
        pd.read_csv(COUNTRY_MAPPING)
        .drop_duplicates()
        .rename(
            columns={
                "Country": "country",
                "Our World In Data Name": "owid_country",
            }
        )
    )
    df = df.merge(mapping, left_on="country", right_on="country", how="left")

    missing = df[pd.isnull(df["owid_country"])]
    if len(missing) > 0:
        missing = "\n".join(missing.country.unique())
        raise Exception(f"Missing entities in mapping:\n{missing}")

    df = df.drop(columns=["country"]).rename(columns={"owid_country": "country"})

    df = df.loc[-(df.country == "DROPENTITY")]
    return df


def _assert_unique(df: pd.DataFrame, subset: List[str]) -> None:
    """Make sure dataframe have only one row per subset"""
    # NOTE: this could be moved to helpers
    df_deduped = df.drop_duplicates(subset=subset)
    if df.shape != df_deduped.shape:
        diff = df[~df.index.isin(df_deduped.index)]
        raise AssertionError(f"Duplicate rows:\n {diff}")


def select_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rows are selected according to the following logic: "unwpp" > "gapminder" > "hyde"
    """
    df = df.loc[df.population > 0]

    # If a country has UN data, then remove all non-UN data after 1949
    has_un_data = set(df.loc[df.source == "unwpp", "country"])
    df = df.loc[
        -((df.country.isin(has_un_data)) & (df.year >= 1950) & (df.source != "unwpp"))
    ]

    # If a country has Gapminder data, then remove all non-Gapminder data between 1800 and 1949
    has_gapminder_data = set(df.loc[df.source == "gapminder", "country"])
    df = df.loc[
        -(
            (df.country.isin(has_gapminder_data))
            & (df.year >= 1800)
            & (df.year <= 1949)
            & (df.source != "gapminder")
        )
    ]

    # Test if all countries have only one row per year
    _assert_unique(df, subset=["country", "year"])

    return df.drop(columns=["source"])


def calculate_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate our own totals according to OWID continent definitions.
    """
    df = cast(
        pd.DataFrame,
        df[
            -df.country.isin(
                [
                    "North America",
                    "South America",
                    "Europe",
                    "Africa",
                    "Asia",
                    "Oceania",
                    "World",
                ]
            )
        ],
    )

    countries = Dataset(REFERENCE)["countries_regions"]

    # TODO: Latin America has been recently removed from the continent definitions in `importers`
    countries = countries[countries.name != "Latin America"]

    continent_rows = []
    for code, row in countries.iterrows():
        if pd.isnull(row.members):
            continue

        members = json.loads(row.members)
        for member in members:
            # TODO: Latin America has been recently removed from the continent definitions in `importers`
            if member == "OWID_LAM":
                continue

            # use ['name'] instead of .name, since .name references name of the object,
            # not the actual value
            continent_rows.append(
                {"country": countries.loc[member]["name"], "continent": row["name"]}
            )

    continent_list = pd.DataFrame.from_records(continent_rows)

    continents = (
        df.merge(continent_list, on="country")
        .groupby(["continent", "year"], as_index=False)
        .sum()
        .rename(columns={"continent": "country"})
    )

    world = (
        df[["year", "population"]]
        .groupby("year")
        .sum()
        .reset_index()
        .assign(country="World")
    )

    return pd.concat([df, continents, world], ignore_index=True)


def prepare_dataset(df: pd.DataFrame) -> Table:
    df = cast(pd.DataFrame, df[df.population > 0].copy())
    df["population"] = df.population.astype("int64")
    df.year = df.year.astype(int)

    # Add a metric "% of world population"
    world_pop = df[df.country == "World"][["year", "population"]].rename(
        columns={"population": "world_pop"}
    )
    df = df.merge(world_pop, on="year", how="left")
    df["world_pop_share"] = (df["population"].div(df.world_pop)).round(4)

    df = df.drop(columns="world_pop").sort_values(["country", "year"])

    t = Table(df.set_index(["country", "year"]))
    t.population.title = "Total population (Gapminder, HYDE & UN)"
    t.world_pop_share.title = "Share of World Population"
    return t


def load_unwpp() -> pd.DataFrame:
    df = Dataset(UNWPP)["total_population"]
    df = df.reset_index().rename(
        columns={
            "location": "country",
            "population_total": "population",
        }
    )
    df = (
        df[df.variant == "Medium"]
        .drop(columns="variant")
        .assign(
            source="unwpp", population=lambda df: df.population.mul(1000).astype(int)
        )[["country", "year", "population", "source"]]
    )
    return cast(pd.DataFrame, df)


def make_combined() -> pd.DataFrame:
    unwpp = load_unwpp()

    gapminder = Dataset(GAPMINDER)["population"]
    gapminder["source"] = "gapminder"
    gapminder.reset_index(inplace=True)

    hyde = Dataset(HYDE)["population"]
    hyde["source"] = "hyde"
    hyde.reset_index(inplace=True)

    return pd.DataFrame(pd.concat([gapminder, hyde, unwpp], ignore_index=True))

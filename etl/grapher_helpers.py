import pandas as pd
from owid import catalog
from collections.abc import Iterable
import yaml
import slugify
from pathlib import Path

from etl.paths import DATA_DIR

from typing import Optional, Dict, Literal
from pydantic import BaseModel


class DatasetModel(BaseModel):
    source: str
    short_name: str
    namespace: str


class DimensionModel(BaseModel):
    pass


class VariableModel(BaseModel):
    description: str
    unit: str
    short_unit: Optional[str]


# TODO: prune unnecessary stuff
class Annotation(BaseModel):
    dataset: Optional[DatasetModel]
    dimensions: Optional[Dict[str, Optional[DimensionModel]]]
    variables: Dict[str, VariableModel]

    @property
    def dimension_names(self):
        return list(self.dimensions.keys())

    @property
    def variable_names(self):
        return list(self.variables.keys())

    @classmethod
    def load_from_yaml(cls, path):
        # Load variable descriptions and units from the annotations.yml file and
        # store them as column metadata
        with open(path) as istream:
            annotations = yaml.safe_load(istream)
        return cls.parse_obj(annotations)

    def create_dataset(self) -> catalog.Dataset:
        # TODO: we already have this dataset path... can we reuse it?
        dataset = catalog.Dataset(DATA_DIR / self.dataset.source)
        dataset.metadata.short_name = self.dataset.short_name
        dataset.metadata.namespace = self.dataset.namespace
        return dataset


def as_table(df: pd.DataFrame, table: catalog.Table) -> catalog.Table:
    """Convert dataframe into Table and add metadata from other table if available."""
    t = catalog.Table(df, metadata=table.metadata)
    for col in set(df.columns) & set(table.columns):
        # TODO: setter on metadata would be nicer
        t[col]._fields[t[col].checked_name] = table[col].metadata
    return t


def annotate_table_from_yaml(table: catalog.Table, path: Path, missing_col:Literal["raise", "ignore"]='raise') -> catalog.Table:
    """Load variable descriptions and units from the annotations.yml file and
    store them as column metadata."""
    annot = Annotation.load_from_yaml(path)

    for column in annot.variable_names:
        v = annot.variables[column]
        if column not in table:
            if missing_col == 'raise':
                raise Exception(f"Column {column} not in table")
            elif missing_col != 'ignore':
                raise ValueError(f"Unknown missing_col value: {missing_col}")
        else:
            # overwrite metadata
            for k, v in dict(v).items():
                setattr(table[column].metadata, k, v)

    return table


def yield_table(table: catalog.Table) -> Iterable[catalog.Table]:
    """We have 5 dimensions but graphers data model can only handle 2 (year and entityId). This means
    we have to iterate all combinations of the remaining 3 dimensions and create a new variable for
    every combination that cuts out only the data points for a specific combination of these 3 dimensions
    Grapher can only handle 2 dimensions (year and entityId)"""
    # Validation
    if 'year' not in table.primary_key:
        raise Exception(f"Table is missing `year` primary key")
    if 'entity_id' not in table.primary_key:
        raise Exception(f"Table is missing `entity_id` primary key")

    dim_names = [k for k in table.primary_key if k not in ('year', 'entity_id')]

    for dims, table_to_yield in table.groupby(dim_names, as_index=False):
        print(" - ".join(dims))

        # Now iterate over every column in the original dataset and export the
        # subset of data that we prepared above
        for column in table_to_yield.columns:

            # Add column and dimensions as short_name
            table_to_yield.metadata.short_name = slugify.slugify(
                "-".join([column] + list(dims))
            )

            # Safety check to see if the metadata is still intact
            assert (
                table_to_yield[column].metadata.unit is not None
            ), "Unit should not be None here!"

            yield table_to_yield.reset_index(dim_names)[[column]]

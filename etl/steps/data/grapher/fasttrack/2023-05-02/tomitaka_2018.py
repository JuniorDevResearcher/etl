import pandas as pd
from owid import catalog

from etl.helpers import PathFinder, create_dataset
from etl.snapshot import Snapshot

P = PathFinder(__file__)


def run(dest_dir: str) -> None:
    # load snapshot
    data = pd.read_csv(Snapshot("fasttrack/2023-05-02/tomitaka_2018.csv").path)

    # create empty dataframe and table
    tb = catalog.Table(data, short_name=P.short_name)

    # add table, update metadata from *.meta.yml and save
    ds = create_dataset(dest_dir, tables=[tb])
    ds.save()

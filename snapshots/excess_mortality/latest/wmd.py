"""Script to create a snapshot of dataset 'World Mortality Dataset'."""

from datetime import date
from pathlib import Path

import click

from etl.snapshot import Snapshot

# Version for current snapshot dataset.
SNAPSHOT_VERSION = Path(__file__).parent.name


@click.command()
@click.option(
    "--upload/--skip-upload",
    default=True,
    type=bool,
    help="Upload dataset to Snapshot",
)
def main(upload: bool) -> None:
    # Create a new snapshot.
    snap = Snapshot(f"excess_mortality/{SNAPSHOT_VERSION}/wmd.csv")

    # Add date_accessed
    snap.metadata.source.date_accessed = date.today()
    snap.metadata.save()

    # Download data from source.
    snap.download_from_source()

    # Add file to DVC and upload to S3.
    snap.dvc_add(upload=upload)


if __name__ == "__main__":
    main()

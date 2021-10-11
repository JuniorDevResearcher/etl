#
#  config.py
#

from os import environ
from typing import cast

from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = "owid-catalog"
S3_REGION_NAME = "nyc3"
S3_ENDPOINT_URL = "https://nyc3.digitaloceanspaces.com"
S3_HOST = "nyc3.digitaloceanspaces.com"
S3_ACCESS_KEY = environ.get("OWID_ACCESS_KEY")
S3_SECRET_KEY = environ.get("OWID_SECRET_KEY")

DEBUG = environ.get("DEBUG") == "True"
GRAPHER_USER_ID = cast(int, environ.get("GRAPHER_USER_ID", -1))

# Copyright 2022 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import click


@click.group()
def cli():
    pass

from . import training
from . import playing
from . import plotting
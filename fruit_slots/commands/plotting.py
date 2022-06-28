import random
import os
import logging
import sys
import itertools
import warnings
import functools

import pathlib

import click
import pandas as pd

from fruit_slots import utils
from . import cli



def get_dataframe(tensorboard_log_path):
    utils.prevent_tensorflow_spam()
    import tensorflow.compat.v1.logging
    tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)
    from tensorflow.python.summary.summary_iterator import summary_iterator

    events = tuple(summary_iterator(str(tensorboard_log_path)))
    d = {}
    columns = {'step'}
    for event in events:
        try:
            step_d = d[event.step]
        except KeyError:
            step_d = d[event.step] = {'step': event.step}
        for value in event.summary.value:
            columns.add(value.tag)
            step_d[value.tag] = value.simple_value
    records = sorted(d.values(), key=lambda step_d: step_d['step'])
    df = pd.DataFrame.from_records(records, columns=columns, index='step')
    return df


all_columns = (
    'rollout/mean_cumulative_visible_apple_reward',
    'rollout/mean_cumulative_invisible_apple_reward',
    'rollout/mean_cumulative_banana_reward',
    'rollout/mean_cumulative_lemon_reward',
)


@cli.command()
@click.argument('tensorboard_log_path', type=str, default='')
def make_chart(tensorboard_log_path):
    import plotly.graph_objects as go
    if tensorboard_log_path == '':
        recent_tensorboard_log_folder = max(utils.log_path.iterdir(),
                                            key=lambda path: path.stat().st_mtime)
        (tensorboard_log_path,) = recent_tensorboard_log_folder.iterdir()
    else:
        tensorboard_log_path = pathlib.Path(tensorboard_log_path)
    print(f'Making a chart for {tensorboard_log_path}')
    df = get_dataframe(tensorboard_log_path)
    columns = tuple(column for column in all_columns if column in df.columns)
    process_column_name = lambda name: (
        name.replace('rollout/mean_cumulative_', '').replace('_', ' ').capitalize()
    )
    axis_template = {'title_font': {'size': 23,}, 'tickfont': {'size': 20,}}
    figure = go.Figure(
        data=[
            go.Scatter(x=df.index, y=df[column], name=process_column_name(column))
            for column in columns
        ],
        layout=go.Layout(
            title=('Points gained over generations of training PPO on Fruit Slots, single '
                   'neural network'),
            title_font_size=28,
            xaxis={**axis_template, 'title': 'Generation'},
            yaxis={**axis_template, 'title': 'Points'},
            legend={'xanchor': 'right', 'x': 1, 'yanchor': 'bottom',
                    'y': 0, 'font': {'size': 30}}
        )
    )
    figure.show()



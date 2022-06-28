import random
import os
import logging
import sys
import itertools
import warnings
import functools
warnings.filterwarnings('ignore')

import pathlib

# Avoid TensorFlow spam:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logging.getLogger('tensorflow').addFilter(
    lambda record: 'Tracing is expensive and the excessive' not in record.msg
)

import click
import pandas as pd



def get_dataframe(tensorboard_log_path):
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

@click.group()
def post_fruitos():
    pass


@post_fruitos.command()
@click.argument('tensorboard_log_path',
                type=click.Path(exists=True, dir_okay=False, resolve_path=True))
def make_csv(tensorboard_log_path):
    df = get_dataframe(tensorboard_log_path)
    df.to_csv(sys.stdout)


@post_fruitos.command()
@click.argument('csv_path', type=click.Path(exists=True, dir_okay=False, resolve_path=True))
def make_chart(csv_path):
    import plotly.graph_objects as go

    df = pd.read_csv(csv_path)[:15000]
    columns = (
        # These are ordered by the phases of the experiment
        # 'rollout/mean_cumulative_reward',
        'rollout/mean_cumulative_visible_apple_reward',
        'rollout/mean_cumulative_invisible_apple_reward',
        'rollout/mean_cumulative_banana_reward',
        'rollout/mean_cumulative_lemon_reward',
    )
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
    figure.update_xaxes
    figure.show('browser')


if __name__ == '__main__':
    post_fruitos()


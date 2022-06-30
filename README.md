# Fruit Slots: An experiment in implicit communication between learning agents

* [Paper](http://r.rachum.com/fruit-slots-paper)
* [Video](http://r.rachum.com/fruit-slots-workshop-video)
* [Slides](http://r.rachum.com/fruit-slots-slides)

## Paper abstract

Recent research in multi-agent reinforcement learning (MARL) has shown success in reproducing social behavior. One caveat that underlies these efforts is that many of them use pro-social intrinsic rewards to coerce the agents into behaving socially, and without this the agents would not learn to behave in such a prosocial manner. We postulate that designing an evaluation domain that encourages implicit communication between agents could lead to achieving emergent reciprocity without pro-social intrinsic rewards.

In this paper we describe a series of experiments using implicit communication that we hope could be
stepping stones for achieving emergent reciprocity, and by extension, emergent social behavior.

## Installing Fruit Slots

```shell
python3 -m venv "${HOME}/fruit_slots_venv"
source "${HOME}/fruit_slots_venv/bin/activate"
pip3 install fruit_slots
```

## Running Fruit Slots


Train an agent and save its policy to your home folder:

```shell
python3 -m fruit_slots train
```

Show Fruit Slots gameplay on the agent you just trained:

```shell
python3 -m fruit_slots play
```

Make a plot showing the learning curves for the agent you just trained:

```shell
python3 -m fruit_slots plot
```

## Documentation

Show list of commands:

```shell
python -m fruit_slots --help
```

Show arguments and options for a specific command:

```shell
python -m fruit_slots train --help
```
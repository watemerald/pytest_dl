# How to Trust Your Deep Learning Code

This is a reimplementation of the code accompanying the blog post [How to Trust Your Deep Learning Code](https://krokotsch.eu/cleancode/2020/08/11/Unit-Tests-for-Deep-Learning.html).

## Motivation

The blog post provided a great resource for understanding how to write reusable unit tests for Deep learning models. But, it served very little practical use since none of my ML projects use python's built-in `unittest` library. In an attempt to better understand how these tests work and make them more reusable on my future projects, I decided to reimplement them using `pytest`.

## Usage

This project uses `poetry` for dependency management. Install it from [https://python-poetry.org/](https://python-poetry.org/), then

```shell
poetry install
pytest
```

to generate the report.

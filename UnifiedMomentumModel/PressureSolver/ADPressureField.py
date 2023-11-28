from pathlib import Path
from typing import Tuple

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from scipy.interpolate import RegularGridInterpolator

from ..Utilities.FixedPointIteration import adaptivefixedpointiteration
from .poisson_pressure_nonlinear import NonLinearPoissonCenterline

CACHE_FN = Path(__file__).parent.parent.parent / "p_NL.csv"


def generate_pressure_table(
    ddp: float = 0.1,
    xmax: float = 10.0,
    max_iter=3,
    tolerance=0.00001,
    relaxations=[0, 0.1, 0.2],
) -> Tuple[ArrayLike, ...]:
    """
    Generate interpolation table by solving nonlinear PDE

    ddp (float) : dp spacing for interpolation table
    xmax (float) : max x-value for interpolation table

    Returns tuple of arrays:
        dps (np.ndarray): dp values for interpolation
        xs (np.ndarray): x values for interpolation
        ps (np.ndarray): pressure values for interpolation
    """

    dps = np.arange(0, 1 + ddp, ddp)
    model = adaptivefixedpointiteration(
        max_iter=max_iter, tolerance=tolerance, relaxations=relaxations
    )(NonLinearPoissonCenterline)()

    # Generate nonlinear pressure table
    ps = []
    for dp in dps:
        xs, p = model(dp)
        ps.append(p[(xs < xmax) & (xs > 0)])

    ps = np.array(ps)
    xs = xs[(xs < xmax) & (xs > 0)]

    # clip non-linear pressure to below zero.
    ps = np.minimum(ps, 0)

    return dps, xs, ps


def make_interpolator(dps, xs, ps) -> RegularGridInterpolator:
    interpolator = RegularGridInterpolator(
        [dps, xs], ps, bounds_error=False, fill_value=0
    )
    return interpolator


def save_cache(
    dps: ArrayLike, xs: ArrayLike, ps: ArrayLike, cache_fn: Path = CACHE_FN
) -> None:
    schema = ["dp"] + [f"{x:.2f}" for x in xs]
    data = np.hstack((np.round(dps[:, np.newaxis], 2), ps))
    df = pl.DataFrame(data, schema=schema)

    df.write_csv(cache_fn)


def load_cache(cache_fn: Path = CACHE_FN):
    df = pl.read_csv(cache_fn)

    dps = df.to_numpy()[:, 0]
    xs = np.array(df.columns[1:], dtype=float)
    ps = df.to_numpy()[:, 1:]

    interpolator = make_interpolator(dps, xs, ps)
    return interpolator

from functools import partial
from typing import Callable, Any, Mapping

import flax
import flax.linen as nn
from flax.core import FrozenDict
from flax.core.scope import CollectionFilter, PRNGSequenceFilter
from flax.linen.transforms import Target, lift_transform
from flax.typing import InOutScanAxis


normal_init = nn.initializers.truncated_normal(stddev=0.02)


def _flax_scan(
    body_fn: Callable[..., Any],
    length: int,
    variable_broadcast: CollectionFilter = False,
    variable_carry: CollectionFilter = False,
    variable_axes: Mapping[CollectionFilter, InOutScanAxis] = {True: 0},
    split_rngs: Mapping[PRNGSequenceFilter, bool] = {True: True},
    unroll: int = 1,
) -> Callable[..., Any]:
    scan_fn = partial(
        flax.core.lift.scan,
        variable_broadcast=variable_broadcast,
        variable_carry=variable_carry,
        variable_axes=variable_axes,
        split_rngs=split_rngs,
        unroll=unroll,
    )

    def wrapper(scope, carry):
        return body_fn(scope, carry), None

    fn = lambda scope, c: scan_fn(wrapper, length=length)(scope, c)[0]

    return fn


def flax_scan(
    target: Target,
    length: int,
    variable_broadcast: CollectionFilter = False,
    variable_carry: CollectionFilter = False,
    variable_axes: Mapping[CollectionFilter, InOutScanAxis] = FrozenDict({True: 0}),
    split_rngs: Mapping[PRNGSequenceFilter, bool] = FrozenDict({True: True}),
    unroll: int = 1,
) -> Target:
    return lift_transform(
        _flax_scan,
        target,
        length=length,
        variable_broadcast=variable_broadcast,
        variable_carry=variable_carry,
        variable_axes=variable_axes,
        split_rngs=split_rngs,
        unroll=unroll,
    )

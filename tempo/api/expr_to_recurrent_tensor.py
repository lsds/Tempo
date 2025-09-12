import functools
from collections.abc import Mapping

from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.core import index_expr as ie
from tempo.core.dtype import dtypes

empty_dict: Mapping[ie.Symbol, int] = {}


def translate_ie_to_rt(
    e: ie.IndexExpr, symbols: Mapping[ie.Symbol, int] = empty_dict
) -> RecurrentTensor:
    # TODO: just call into symbolic tensor translate_ie_to_st once max and min are implemented
    # Similar to here, use a map
    if isinstance(e, ie.ConstBool):
        return RecurrentTensor.const(e.const, dtype=dtypes.bool_)
    if isinstance(e, ie.ConstInt):
        return RecurrentTensor.const(e.const, dtype=dtypes.implied(e.const))
    if isinstance(e, ie.Symbol):
        return RecurrentTensor.from_symbol(e)
    if isinstance(e, ie.Slice):
        # TODO when we have dynamic arange, use this instead
        # start = translate_ie_to_rt(e.start)
        # stop = translate_ie_to_rt(e.stop)

        start: int = e.start.evaluate(symbols)
        stop: int = e.stop.evaluate(symbols)
        return RecurrentTensor.arange(start=start, stop=stop, dtype=dtypes.implied(stop))
    if isinstance(e, ie.IndexSequence):
        # TODO either this or throw an error
        members = [translate_ie_to_rt(m, symbols) for m in e.members]
        return RecurrentTensor.stack(*members)
    if isinstance(e, (ie.IntBinaryExpr, ie.BooleanBinaryExpr, ie.FloatBinaryExpr)):
        bin_map = {
            ie.Add: RecurrentTensor.add,
            ie.Sub: RecurrentTensor.sub,
            ie.Modulos: RecurrentTensor.mod,
            ie.FloorDivision: RecurrentTensor.floor_div,
            ie.Mul: RecurrentTensor.mul,
            ie.Pow: RecurrentTensor.pow_,
            # Float
            ie.TrueDivision: RecurrentTensor.div,
            # Bool
            ie.And: RecurrentTensor.logical_and,
            ie.Or: RecurrentTensor.logical_or,
            ie.Equal: RecurrentTensor.equals,
            ie.GreaterThan: RecurrentTensor.greater_than,
            ie.GreaterThanOrEqual: RecurrentTensor.greater_than_or_equal,
            ie.LessThan: RecurrentTensor.less_than,
            ie.LessThanOrEqual: RecurrentTensor.less_than_or_equal,
        }
        return bin_map[type(e)](
            translate_ie_to_rt(e.left_operand, symbols),
            translate_ie_to_rt(e.right_operand, symbols),
        )
    if isinstance(e, (ie.IntUnaryExpr, ie.BooleanUnaryExpr)):
        un_map = {
            ie.Neg: RecurrentTensor.neg,
            ie.Not: RecurrentTensor.not_,
        }
        return un_map[type(e)](translate_ie_to_rt(e.operand, symbols))
    if isinstance(e, ie.NAryExpr):
        children = [translate_ie_to_rt(c, symbols) for c in e.children]
        nary_map = {
            ie.Max: RecurrentTensor.max,
            ie.Min: RecurrentTensor.min,
        }
        # NOTE: must get only the max or min values, not the indexes
        return functools.reduce(lambda x, y: nary_map[type(e)](x, y)[0], children)  # type: ignore
    raise ValueError(f"Unsupported index expression: {e}")


# Design Notes

The way this works is that the tensor store always returns tensors with all the dimensions,
and requires tensor "sets" to also have all dimensions.

Thus some computation like:
    y[b,t] = x[b, t:min(t+3,T)].reshape((min(t+3,T) - t, -1))

Will create a runtime tensor with shape [B, 1, min(t+3,T) - t, prod(inner_dims)]

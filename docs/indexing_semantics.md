# Indexing Semantics



## Torch Indexing Semantics
With a torch tensor x of shape (A,B,C,D), if we first index it, then negate the tensor we get:
|  Index  | Shape after Index |
| :-----: | :---------------: |
|    0    |      (B,C,D)      |
|  0, 1   |       (C,D)       |
|  :, 1   |      (A,C,D)      |
| :, 5:10 |     (A,5,C,D)     |
|   ...   |     (A,B,C,D)     |
| 1,...,2 |       (B,C)       |


## Tempo Indexing Semantics

The abstraction we give users of RecurrentTensors is:
1. You have access to every value of the temporal domain
2. You can think about defining your algorithm for a single batch entry, a single timestep, a single ... (the symbolic basis)

The main difference to Torch is that a non-specified index is assumed to be indexed at the symbolic basis.

With a tpo tensor x of shape (C,D), where the first 2 dimensions (A,B) are symbolic (with basis a, b):

|      Index       | Spatial Shape after Index | Temporal Domain after Index |
| :--------------: | :---------------: | :----------------: |
|        0         |       (C,D)       |        (b,)        |
|       0, 1       |       (C,D)       |        (,)         |
|       :, 1       |       (C,D)       |        (a,)        |
|     :, 5:10      |      (5,C,D)      |        (a,)        |
|       ...        |       (C,D)       |       (a,b)        |
|     1,...,2      |       (C,)        |        (b,)        |
|       a,b        |       (C,D)       |       (a,b,)       |
|      a+1,b       |       (C,D)       |       (a,b,)       |
| min(a+10, A-1),b |       (C,D)       |       (a,b,)       |
|      0:5,b       |      (5,C,D)      |        (b,)        |
|       0:a+1      |       (a+1, C,D)       |        (a,b)         |


Indexing of a symbolic dimension with a const index (5, 0:T, 0:5, T-4), makes that dimension disappear from the temporal domain.
Indexing a symbolic dimension with a slice index (0:T, 0:t, t:T, 0:t+1, 0:5), will create a new spatial dimension in the shape.
Indexing a tensor with a variable slice (0:t, t:T, 0:t+1) will both retain the temporal dimension in the domain and add a dynamic symbolic spatial dimension.
Spatial shapes introduced by indexing are always prepended.

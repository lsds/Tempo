# TODO welford's online algorithm for variance/mean
# variance(samples):
#  M := 0
#  S := 0
#  for k from 1 to N:
#    x := samples[k]
#    oldM := M
#    M := M + (x-M)/k
#    S := S + (x-M)*(x-oldM)
#  return S/(N-1)

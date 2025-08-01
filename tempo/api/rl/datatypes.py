# Define new types based on RecurrentTensor
from typing import NewType

from tempo.api.recurrent_tensor import RecurrentTensor

ActionsRecurrentTensor = NewType("ActionsRecurrentTensor", RecurrentTensor)
ValueRecurrentTensor = NewType("ValueRecurrentTensor", RecurrentTensor)
ObservationsRecurrentTensor = NewType("ObservationsRecurrentTensor", RecurrentTensor)
RewardsRecurrentTensor = NewType("RewardsRecurrentTensor", RecurrentTensor)
TruncationsRecurrentTensor = NewType("TruncationsRecurrentTensor", RecurrentTensor)
TerminationsRecurrentTensor = NewType("TerminationsRecurrentTensor", RecurrentTensor)
InfoRecurrentTensor = NewType("InfoRecurrentTensor", RecurrentTensor)

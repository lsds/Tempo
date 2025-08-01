from enum import Enum


class StmtType(Enum):
    EXECUTE = "E"
    DEALLOCATE = "D"
    OFFLOAD = "O"
    FETCH = "F"

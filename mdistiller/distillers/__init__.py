from ._base import Vanilla
from .KD import KD
from .MLLD_Ours_backup import MLLD_Ours
from .KD_Mixed2 import KD_Mixed2
from .AT import AT
from .OFD import OFD
from .RKD import RKD
from .FitNet import FitNet
from .KDSVD import KDSVD
from .CRD import CRD
from .NST import NST
from .PKT import PKT
from .SP import SP
from .Sonly import Sonly
from .VID import VID
from .ReviewKD import ReviewKD
from .DKD import DKD
from .CRLD_Ours import CRLD_Ours
from .ReviewKD_Ours import ReviewKD_Ours

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "AT": AT,
    "OFD": OFD,
    "RKD": RKD,
    "FITNET": FitNet,
    "KDSVD": KDSVD,
    "CRD": CRD,
    "NST": NST,
    "PKT": PKT,
    "SP": SP,
    "Sonly": Sonly,
    "VID": VID,
    "REVIEWKD": ReviewKD,
    "DKD": DKD,
    "MLLD_Ours": MLLD_Ours_backup,
    "KD_Mixed2": KD_Mixed2,
    "CRLD_Ours": CRLD_Ours,
    "MLLD_Ours": MLLD_Ours,
    "ReviewKD_Ours": ReviewKD_Ours,
}

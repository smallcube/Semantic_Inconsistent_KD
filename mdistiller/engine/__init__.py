from .trainer import BaseTrainer, CRDTrainer, AugTrainer, MLLD_Ours, CRLD_Ours, ReviewKD_Ours

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "ours": AugTrainer,
    "mlld_ours": MLLD_Ours,
    "crld_ours": CRLD_Ours,
    "reviewkd_ours": ReviewKD_Ours,
}

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller
import numpy as np

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, temperature, reduce=True, logit_stand=False):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

#def kd_loss(logits_student, logits_teacher, temperature, reduce=False):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    if reduce:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    else:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature**2
    return loss_kd.view(-1, 1)

def mixup_data(x_weak, x_strong, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x_weak.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    x_weak_mixed = lam * x_weak + (1 - lam) * x_weak[index, :]
    x_strong_mixed = lam * x_strong + (1 - lam) * x_strong[index, :]
    return x_weak_mixed, x_strong_mixed, y[index], lam, index


def get_modulating_factor(feature, feature_mixed, lam, index):
    batch_size = feature.shape[0]
    features_ground_truth = torch.arange(batch_size, dtype=torch.long).view(-1, 1).to(feature.device)
    feature = feature.view(batch_size, -1)
    feature_mixed=feature_mixed.view(batch_size, -1)

    feature = lam*feature + (1-lam)*feature[index]
    feature = feature/feature.norm(dim=1, keepdim=True)
    feature_mixed = feature_mixed/feature_mixed.norm(dim=1, keepdim=True)

    feature_logit = feature @ feature_mixed.t()
    feature_logit = torch.softmax(feature_logit, dim=-1)
    #step 1: modulating factor
    modulating_factor = feature_logit.gather(1, features_ground_truth).view(-1, 1)
    #feature_logit = torch.softmax(feature_logit, dim=0)
    return modulating_factor

class CRLD_Ours(Distiller):
    """Cross-View Consistency Regularisation for Knowledge Distillation (ACMMM2024)"""

    def __init__(self, student, teacher, cfg):
        super(CRLD_Ours, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.CRLD.CE_WEIGHT
        self.wv_loss_weight = cfg.CRLD.WV_WEIGHT
        self.cv_loss_weight = cfg.CRLD.CV_WEIGHT
        self.t = cfg.CRLD.TEMPERATURE
        self.tau_w = cfg.CRLD.TAU_W
        self.tau_s = cfg.CRLD.TAU_S

        self.base_weight = cfg.EXPERIMENT.BASE_WEIGHT
        self.mixed = cfg.EXPERIMENT.MIXED
        self.alpha = cfg.EXPERIMENT.ALPHA
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND
        

    def forward_train(self, image_w, image_s, target, **kwargs):
        #extra step: mixup
        image_w_mixed, image_s_mixed, y_b, lam, index = mixup_data(image_w, image_s, target, alpha=self.alpha)

        logits_student_w, features_student_w = self.student(image_w)
        logits_student_s, features_student_s = self.student(image_s)
        logits_student_w_mixed, features_student_w_mixed = self.student(image_w_mixed)
        if self.mixed:
            logits_student_s_mixed, features_student_s_mixed = self.student(image_s_mixed)
        with torch.no_grad():
            logits_teacher_w, features_teacher_w = self.teacher(image_w)
            logits_teacher_s, features_teacher_s = self.teacher(image_s)
            
            if self.mixed:
                logits_teacher_w_mixed, features_teacher_w_mixed = self.teacher(image_w_mixed) # for vit teacher
                logits_teacher_s_mixed, features_teacher_s_mixed = self.teacher(image_s_mixed) # for vit teacher
            
        pred_teacher_w = F.softmax(logits_teacher_w.detach(), dim=1)
        conf_w, _ = pred_teacher_w.max(dim=1)
        conf_w = conf_w.detach()
        mask_w = conf_w.ge(self.tau_w).bool().view(-1, 1)

        pred_teacher_s = F.softmax(logits_teacher_s.detach(), dim=1)
        conf_s, _ = pred_teacher_s.max(dim=1)
        conf_s = conf_s.detach()
        mask_s = conf_s.ge(self.tau_s).bool().view(-1, 1)
	
        # losses
        if self.mixed:
            #cross-entropy loss
            loss_student_ce_weak = lam*F.cross_entropy(logits_student_w_mixed, target, reduction='none').view(-1, 1) + \
                                    (1-lam)*F.cross_entropy(logits_student_w_mixed, y_b, reduction='none').view(-1, 1)
            
            loss_student_ce_strong = lam*F.cross_entropy(logits_student_s_mixed, target, reduction='none').view(-1, 1) + \
                                    (1-lam)*F.cross_entropy(logits_student_s_mixed, y_b, reduction='none').view(-1, 1)
            
            loss_ce = self.ce_loss_weight * (loss_student_ce_weak+loss_student_ce_strong)
        else:
            loss_ce_weak = self.ce_loss_weight + F.cross_entropy(logits_student_w, target, reduction='none').view(-1, 1)
            loss_ce_strong = self.ce_loss_weight + F.cross_entropy(logits_student_s, target, reduction='none').view(-1, 1)
            loss_ce = self.ce_loss_weight * (loss_ce_weak + loss_ce_strong)

        loss_kd_wv = self.wv_loss_weight * ((kd_loss(logits_student_in=logits_student_w_mixed if self.mixed else logits_student_w, 
                                                     logits_teacher_in=logits_teacher_w_mixed.detach() if self.mixed else logits_teacher_w, 
                                                     temperature=self.t,
                                                     reduce=False,
                                                     logit_stand=self.logit_stand) \
                                           + kd_loss(logits_student_in=logits_student_s_mixed if self.mixed else logits_student_s, 
                                                     logits_teacher_in=logits_teacher_s_mixed.detach() if self.mixed else logits_teacher_s, 
                                                     temperature=self.t,
                                                     reduce=False,
                                                     logit_stand=self.logit_stand)) * mask_w).view(-1, 1)
        
        loss_kd_cv = self.cv_loss_weight * ((kd_loss(logits_student_in=logits_student_s_mixed if self.mixed else logits_student_s, 
                                                     logits_teacher_in=logits_teacher_w_mixed.detach() if self.mixed else logits_teacher_w, 
                                                     temperature=self.t,
                                                     reduce=False,
                                                     logit_stand=self.logit_stand) \
                                           + kd_loss(logits_student_in=logits_student_w_mixed if self.mixed else logits_student_w, 
                                                     logits_teacher_in=logits_teacher_s_mixed.detach() if self.mixed else logits_teacher_s, 
                                                     temperature=self.t,
                                                     reduce=False,
                                                     logit_stand=self.logit_stand)) * mask_w).view(-1, 1)
        
         
        loss_kd = loss_kd_wv + loss_kd_cv
        modulating_factor = self.base_weight - get_modulating_factor(feature=features_student_w['pooled_feat'], 
                                                                    feature_mixed=features_student_w_mixed['pooled_feat'], 
                                                                    lam=lam, index=index)
            
        loss = modulating_factor * (loss_kd + loss_ce)

        return logits_student_w, loss.mean()
from termios import CEOL
from turtle import st
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ._base import Distiller
from .loss import CrossEntropyLabelSmooth

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, temperature, reduce=True, logit_stand=False):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    if reduce:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    else:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature**2
    #print("loss_kd.shape=", loss_kd.shape)
    return loss_kd


def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    #print("consistency_loss.shape=", consistency_loss.shape)
    return consistency_loss.sum()


def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum(1) / batch_size
    return consistency_loss


def mixup_data(x, x_strong, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    x_mixed = lam * x + (1 - lam) * x[index, :]
    x_mixed_strong = lam*x_strong + (1-lam)*x_strong[index, :]
    y_a, y_b = y, y[index]
    return x_mixed, x_mixed_strong, y_b, lam, index


def mixup_data_conf(x, y, lam, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = lam.reshape(-1,1,1,1)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

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
    modulating_factor = feature_logit.gather(1, features_ground_truth)
    return modulating_factor.view(-1, 1)

class MLLD_Ours(Distiller):
    def __init__(self, student, teacher, cfg):
        super(MLLD_Ours, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND
        self.base_weight = cfg.EXPERIMENT.BASE_WEIGHT
        self.mixed = cfg.EXPERIMENT.MIXED
        self.alpha = cfg.EXPERIMENT.ALPHA

        #print("logtdgadgsagdgdgdfg=", self.logit_stand)

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        #extra step: mixup
        image_weak_mixed, image_strong_mixed, y_b, lam, index = mixup_data(image_weak, image_strong, target, alpha=self.alpha)
        logits_student_weak_mixed, features_student_weak_mixed = self.student(image_weak_mixed)

        logits_student_weak, features_student_weak = self.student(image_weak)
        logits_student_strong, features_strong_weak = self.student(image_strong)
        if self.mixed:
            logits_student_strong_mixed, features_strong_weak_mixed = self.student(image_strong_mixed)


        with torch.no_grad():
            logits_teacher_weak, _ = self.teacher(image_weak)
            logits_teacher_strong, _ = self.teacher(image_strong) 
            if self.mixed:
                logits_teacher_weak_mixed, _ = self.teacher(image_weak_mixed)
                logits_teacher_strong_mixed, _ = self.teacher(image_strong_mixed) 
                

        batch_size, class_num = logits_student_strong.shape

        conf_mini = 50 if self.mixed else 50

        pred_teacher_weak = F.softmax(logits_teacher_weak_mixed, dim=1) if self.mixed else F.softmax(logits_teacher_weak.detach(), dim=1)
        confidence, pseudo_labels = pred_teacher_weak.max(dim=1)
        confidence = confidence.detach()
        conf_thresh = np.percentile(
            confidence.cpu().numpy().flatten(), conf_mini
        )
        mask = confidence.le(conf_thresh).bool()

        class_confidence = torch.sum(pred_teacher_weak, dim=0)
        class_confidence = class_confidence.detach()
        class_confidence_thresh = np.percentile(
            class_confidence.cpu().numpy().flatten(), conf_mini
        )
        class_conf_mask = class_confidence.le(class_confidence_thresh).bool()

        #print("class_conf_mask.shape=", class_conf_mask.shape)
        
        # losses
        if self.mixed:
            loss_ce_weak = self.ce_loss_weight * (lam*F.cross_entropy(logits_student_weak_mixed, target, reduction='none') + \
                                             (1-lam)*F.cross_entropy(logits_student_weak_mixed, y_b, reduction='none'))
            loss_ce_strong = self.ce_loss_weight * (lam*F.cross_entropy(logits_student_strong_mixed, target, reduction='none') + \
                                             (1-lam)*F.cross_entropy(logits_student_strong_mixed, y_b, reduction='none'))
        else:
            loss_ce_weak = self.ce_loss_weight * F.cross_entropy(logits_student_weak, target, reduction='none')
            loss_ce_strong = self.ce_loss_weight * F.cross_entropy(logits_student_strong, target, reduction='none')
        
        loss_ce = loss_ce_weak + loss_ce_strong


        loss_kd_weak = self.kd_loss_weight * ((kd_loss(
            logits_student_in=logits_student_weak_mixed if self.mixed else logits_student_weak,
            logits_teacher_in=logits_teacher_weak_mixed if self.mixed else logits_teacher_weak,
            temperature=self.temperature,
            reduce=False,
            logit_stand=self.logit_stand,
        ) * mask)) + self.kd_loss_weight * ((kd_loss(
            logits_student_in=logits_student_weak_mixed if self.mixed else logits_student_weak,
            logits_teacher_in=logits_teacher_weak_mixed if self.mixed else logits_teacher_weak,
            temperature=3.0,
            reduce=False,
            logit_stand=self.logit_stand,
        ) * mask)) + self.kd_loss_weight * ((kd_loss(
            logits_student_in=logits_student_weak_mixed if self.mixed else logits_student_weak,
            logits_teacher_in=logits_teacher_weak_mixed if self.mixed else logits_teacher_weak,
            temperature=5.0,
            reduce=False,
            logit_stand=self.logit_stand,
        ) * mask)) + self.kd_loss_weight * ((kd_loss(
            logits_student_in=logits_student_weak_mixed if self.mixed else logits_student_weak,
            logits_teacher_in=logits_teacher_weak_mixed if self.mixed else logits_teacher_weak,
            temperature=2.0,
            reduce=False,
            logit_stand=self.logit_stand,
        ) * mask)) + self.kd_loss_weight * ((kd_loss(
            logits_student_in=logits_student_weak_mixed if self.mixed else logits_student_weak,
            logits_teacher_in=logits_teacher_weak_mixed if self.mixed else logits_teacher_weak,
            temperature=6.0,
            reduce=False,
            logit_stand=self.logit_stand,
        ) * mask))

        loss_kd_strong = self.kd_loss_weight * kd_loss(
            logits_student_in=logits_student_strong_mixed if self.mixed else logits_student_strong,
            logits_teacher_in=logits_teacher_strong_mixed if self.mixed else logits_teacher_strong,
            temperature=self.temperature,
            reduce=False,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_in=logits_student_strong_mixed if self.mixed else logits_student_strong,
            logits_teacher_in=logits_teacher_strong_mixed if self.mixed else logits_teacher_strong,
            temperature=3.0,
            reduce=False,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_in=logits_student_strong_mixed if self.mixed else logits_student_strong,
            logits_teacher_in=logits_teacher_strong_mixed if self.mixed else logits_teacher_strong,
            temperature=5.0,
            reduce=False,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_in=logits_student_weak_mixed if self.mixed else logits_student_weak,
            logits_teacher_in=logits_teacher_weak_mixed if self.mixed else logits_teacher_weak,
            temperature=2.0,
            reduce=False,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_in=logits_student_weak_mixed if self.mixed else logits_student_weak,
            logits_teacher_in=logits_teacher_weak_mixed if self.mixed else logits_teacher_weak,
            temperature=6.0,
            reduce=False,
            logit_stand=self.logit_stand,
        )

        loss_cc_weak = self.kd_loss_weight * ((cc_loss(
            logits_student=logits_student_weak_mixed if self.mixed else logits_student_weak,
            logits_teacher=logits_teacher_weak_mixed if self.mixed else logits_teacher_weak,
            temperature=self.temperature,
            #reduce=True,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student=logits_student_weak_mixed if self.mixed else logits_student_weak,
            logits_teacher=logits_teacher_weak_mixed if self.mixed else logits_teacher_weak,
            temperature=3.0,
            #reduce=False,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student=logits_student_weak_mixed if self.mixed else logits_student_weak,
            logits_teacher=logits_teacher_weak_mixed if self.mixed else logits_teacher_weak,
            temperature=5.0,
            #reduce=False,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student=logits_student_weak_mixed if self.mixed else logits_student_weak,
            logits_teacher=logits_teacher_weak_mixed if self.mixed else logits_teacher_weak,
            temperature=2.0,
            #reduce=False,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student=logits_student_weak_mixed if self.mixed else logits_student_weak,
            logits_teacher=logits_teacher_weak_mixed if self.mixed else logits_teacher_weak,
            temperature=6.0,
            #reduce=False,
        ) * class_conf_mask).mean())

        #print("loss_cc_weak=",loss_cc_weak.shape)

        loss_cc_strong = self.kd_loss_weight * cc_loss(
            logits_student=logits_student_strong_mixed if self.mixed else logits_student_strong,
            logits_teacher=logits_teacher_strong_mixed if self.mixed else logits_teacher_strong,
            temperature=self.temperature,
            reduce=False,
        ) + self.kd_loss_weight * cc_loss(
            logits_student=logits_student_strong_mixed if self.mixed else logits_student_strong,
            logits_teacher=logits_teacher_strong_mixed if self.mixed else logits_teacher_strong,
            temperature=3.0,
            reduce=False,
        ) + self.kd_loss_weight * cc_loss(
            logits_student=logits_student_strong_mixed if self.mixed else logits_student_strong,
            logits_teacher=logits_teacher_strong_mixed if self.mixed else logits_teacher_strong,
            temperature=5.0,
            reduce=False,
        ) + self.kd_loss_weight * cc_loss(
            logits_student=logits_student_strong_mixed if self.mixed else logits_student_strong,
            logits_teacher=logits_teacher_strong_mixed if self.mixed else logits_teacher_strong,
            temperature=2.0,
            reduce=False,
        ) + self.kd_loss_weight * cc_loss(
            logits_student=logits_student_strong_mixed if self.mixed else logits_student_strong,
            logits_teacher=logits_teacher_strong_mixed if self.mixed else logits_teacher_strong,
            temperature=6.0,
            reduce=False,
        )

        loss_bc_weak = self.kd_loss_weight * ((bc_loss(
            logits_student=logits_student_weak_mixed if self.mixed else logits_student_weak,
            logits_teacher=logits_teacher_weak_mixed if self.mixed else logits_teacher_weak,
            temperature=self.temperature,
            reduce=False,
        ) * mask)) + self.kd_loss_weight * ((bc_loss(
            logits_student=logits_student_weak_mixed if self.mixed else logits_student_weak,
            logits_teacher=logits_teacher_weak_mixed if self.mixed else logits_teacher_weak,
            temperature=3.0,
            reduce=False,
        ) * mask)) + self.kd_loss_weight * ((bc_loss(
            logits_student=logits_student_weak_mixed if self.mixed else logits_student_weak,
            logits_teacher=logits_teacher_weak_mixed if self.mixed else logits_teacher_weak,
            temperature=5.0,
            reduce=False,
        ) * mask)) + self.kd_loss_weight * ((bc_loss(
            logits_student=logits_student_weak_mixed if self.mixed else logits_student_weak,
            logits_teacher=logits_teacher_weak_mixed if self.mixed else logits_teacher_weak,
            temperature=2.0,
            reduce=False,
        ) * mask)) + self.kd_loss_weight * ((bc_loss(
            logits_student=logits_student_weak_mixed if self.mixed else logits_student_weak,
            logits_teacher=logits_teacher_weak_mixed if self.mixed else logits_teacher_weak,
            temperature=6.0,
            reduce=False,
        ) * mask))

        #print("loss_bc_weak=", loss_bc_weak.shape)

        loss_bc_strong = self.kd_loss_weight * ((bc_loss(
            logits_student=logits_student_strong_mixed if self.mixed else logits_student_strong,
            logits_teacher=logits_teacher_strong_mixed if self.mixed else logits_teacher_strong,
            temperature=self.temperature,
            reduce=False,
        ) * mask)) + self.kd_loss_weight * ((bc_loss(
            logits_student=logits_student_strong_mixed if self.mixed else logits_student_strong,
            logits_teacher=logits_teacher_strong_mixed if self.mixed else logits_teacher_strong,
            temperature=3.0,
            reduce=False,
        ) * mask)) + self.kd_loss_weight * ((bc_loss(
            logits_student=logits_student_strong_mixed if self.mixed else logits_student_strong,
            logits_teacher=logits_teacher_strong_mixed if self.mixed else logits_teacher_strong,
            temperature=5.0,
            reduce=False,
        ) * mask)) + self.kd_loss_weight * ((bc_loss(
            logits_student=logits_student_strong_mixed if self.mixed else logits_student_strong,
            logits_teacher=logits_teacher_strong_mixed if self.mixed else logits_teacher_strong,
            temperature=2.0,
            reduce=False,
        ) * mask)) + self.kd_loss_weight * ((bc_loss(
            logits_student=logits_student_strong_mixed if self.mixed else logits_student_strong,
            logits_teacher=logits_teacher_strong_mixed if self.mixed else logits_teacher_strong,
            temperature=6.0,
            reduce=False,
        ) * mask))

        #get modulating_factor
        modulating_factor = self.base_weight- get_modulating_factor(feature=features_student_weak['pooled_feat'], 
                                                                    feature_mixed=features_student_weak_mixed['pooled_feat'], 
                                                                    lam=lam, index=index)
        loss = loss_ce.view(-1, 1) + (loss_kd_weak + loss_kd_strong).view(-1, 1) + (loss_cc_weak + loss_cc_strong) + (loss_bc_weak+loss_bc_strong).view(-1, 1)
        loss = modulating_factor.view(-1, 1) * loss
        
        return logits_student_weak, loss.mean()


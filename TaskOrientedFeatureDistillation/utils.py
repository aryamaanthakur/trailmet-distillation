import torch
import torch.nn as nn
import torch.nn.functional as F

class TOFDLoss(nn.Module):
    def __init__(self, criterion, alpha, beta, device=None):
        super().__init__()
        self.criterion = criterion
        self.alpha = alpha
        self.beta = beta
        self.device = device

    def orthogonal_loss(self, link):
        loss = 0
        for layer in link:
            weight = list(layer.parameters())[0]
            weight_trans = weight.permute(1, 0)
            ones = torch.eye(weight.size(0)).to(self.device)
            ones2 = torch.eye(weight.size(1)).to(self.device)
            loss += torch.dist(torch.mm(weight, weight_trans), ones, p=2)
            loss += torch.dist(torch.mm(weight_trans, weight), ones2, p=2)
        
        return loss

    @staticmethod
    def logit_loss(student_logit, teacher_logit, T):
        log_softmax_outputs = F.log_softmax(student_logit/T, dim=1)
        softmax_targets = F.softmax(teacher_logit/T, dim=1)
        loss = -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()
        return loss

    @staticmethod
    def feature_loss(student_feature, teacher_feature):
        loss = torch.dist(student_feature, teacher_feature, p=2)
        return loss

    @staticmethod
    def task_loss(student_logit, labels, criterion):
        loss = criterion(student_logit, labels)
        return loss

    def forward(self, student_features, teacher_features, student_logits, teacher_logits, labels, temperature, link):
        if self.device:
            loss = torch.FloatTensor([0.]).to(self.device)
        else:
            loss = torch.FloatTensor([0.])

        for index in range(len(student_features)):
            student_feature = link[index](student_features[index])
            loss += self.task_loss(student_logits[index], labels, self.criterion)
            loss += self.alpha * self.feature_loss(student_feature, teacher_features[index])
            loss += self.logit_loss(student_logits[index], teacher_logits[index], temperature)

        loss += self.beta * self.orthogonal_loss(link)
        return loss

def create_link(model, student_feature_size, teacher_feature_size, num_auxiliary_classifiers):

    link = []
    for i in range(num_auxiliary_classifiers):
        link.append(nn.Linear(student_feature_size, teacher_feature_size, bias=False))
    model.link = nn.ModuleList(link)

    return model

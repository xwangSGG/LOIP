import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
import math
from copy import deepcopy

#from train import num_clusters


def CalculateLoss(vladQ, vladP, vladN, margin):
    # if (not isinstance(vladQ,list)) or (not isinstance(vladP,list)) or (not isinstance(vladN,list)):
    #     raise Exception("vladQ, vladP, vladN should be a list with a length of 4!")
    # if len(vladQ)!=4 or len(vladP)!=4 or len(vladN)!=4:
    #     raise Exception("vladQ, vladP, vladN should be a list with a length of 4!")


    batchNum = vladQ.size(0)
    Q = [[0] * 4 for _ in range(batchNum)]
    P = [[0] * 4 for _ in range(batchNum)]
    N = [[0] * 4 for _ in range(batchNum)]
    Loss = torch.zeros(batchNum, 1)


    # CurrentQ = vladQ.deepcopy()
    # CurrentP = vladP.deepcopy()
    # CurrentN = vladN.deepcopy()
    for i in range(4):
        tempQ = vladQ[:,:,i]
        tempP = vladP[:,:,i]
        tempN = vladN[:,:,i]
        # if tempQ.size()!=(batchNum,num_clusters * 512) or tempP.size()!=(batchNum,num_clusters * 512) or tempN.size()!=(batchNum,num_clusters * 512):
        #     raise Exception("The tensor dimension of each element of vladQ, vladP and vladN should be batchNum, num_clusters * 512!")
        for b in range(batchNum):
            # F.normalize(input, p=2, dim=self.dim)
            # CurrentQ = Normal(tempQ[b])
            # CurrentP = Normal(tempP[b])
            # CurrentN = Normal(tempN[b])
            Q[b][i] = F.normalize(tempQ[b], dim=0)
            P[b][i] = F.normalize(tempP[b], dim=0)
            N[b][i] = F.normalize(tempN[b], dim=0)

            # Q[b][i] = CurrentQ
            # P[b][i] = CurrentP
            # N[b][i] = CurrentN

    for b in range(batchNum):
        q1 = Q[b][0]
        q2 = Q[b][1]
        q3 = Q[b][2]
        q4 = Q[b][3]

        p1 = P[b][0]
        p2 = P[b][1]
        p3 = P[b][2]
        p4 = P[b][3]

        n1 = N[b][0]
        n2 = N[b][1]
        n3 = N[b][2]
        n4 = N[b][3]
        D1 = min(EuclideanDistance(q1, p1), EuclideanDistance(q1, p2), EuclideanDistance(q1, p3), EuclideanDistance(q1, p4)) + \
             min(EuclideanDistance(q2, p1), EuclideanDistance(q2, p2), EuclideanDistance(q2, p3), EuclideanDistance(q2, p4)) + \
             min(EuclideanDistance(q3, p1), EuclideanDistance(q3, p2), EuclideanDistance(q3, p3), EuclideanDistance(q3, p4)) + \
             min(EuclideanDistance(q4, p1), EuclideanDistance(q4, p2), EuclideanDistance(q4, p3), EuclideanDistance(q4, p4))


        D2 = min(EuclideanDistance(q1, n1), EuclideanDistance(q1, n2), EuclideanDistance(q1, n3), EuclideanDistance(q1, n4)) + \
             min(EuclideanDistance(q2, n1), EuclideanDistance(q2, n2), EuclideanDistance(q2, n3), EuclideanDistance(q2, n4)) + \
             min(EuclideanDistance(q3, n1), EuclideanDistance(q3, n2), EuclideanDistance(q3, n3), EuclideanDistance(q3, n4)) + \
             min(EuclideanDistance(q4, n1), EuclideanDistance(q4, n2), EuclideanDistance(q4, n3), EuclideanDistance(q4, n4))

        D3 = min(EuclideanDistance(p1, n1), EuclideanDistance(p1, n2), EuclideanDistance(p1, n3),EuclideanDistance(p1, n4)) + \
             min(EuclideanDistance(p2, n1), EuclideanDistance(p2, n2), EuclideanDistance(p2, n3),EuclideanDistance(p2, n4)) + \
             min(EuclideanDistance(p3, n1), EuclideanDistance(p3, n2), EuclideanDistance(p3, n3),EuclideanDistance(p3, n4)) + \
             min(EuclideanDistance(p4, n1), EuclideanDistance(p4, n2), EuclideanDistance(p4, n3),EuclideanDistance(p4, n4))
        Dmin = min(D2, D3)
        if math.isnan(D1 + margin - Dmin):
            raise Exception("NAN")

        Loss[b] = torch.clamp_(D1 + margin - Dmin, min = 0.0)
    return Loss


def EuclideanDistance(x, y):
    return torch.sqrt(torch.sum((x-y)**2, dim=0))

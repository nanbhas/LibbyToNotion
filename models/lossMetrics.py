"""
Author: Nandita Bhaskhar
Defines the loss functions and metrics
"""

import os, sys
sys.path.append('../')

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    accuracy_score
)


def lossFnMC(logits, labels):
    """
    Compute the cross entropy loss given logits and labels for multi class classification
    Args:
        logits: raw scores from NN. Dimension: (batchSize, numClasses) - output of the model
        labels: targets from dataset. Dimension: (batchSize, 1) where each element is the class index
    Returns:
        loss: cross entropy loss for all slices in the batch
    """

    loss = F.cross_entropy(logits, labels)
    # print('##################')
    # print(logits)
    # print(labels)
    # print('#################')
    return loss

def getClassificationReport(logits, labels, labelIdx = None, labelNames = None):
    """
    Returns the full classification report given logits and labels for classification
    Args:
        logits: raw scores from NN. Dimension: (batchSize, numClasses) - output of the model
        labels: targets from dataset. Dimension: (batchSize, 1) where each element is the class index
        labelIdx: array-like of shape (n_labels,), default = None. Optional list of label indices to include in the report.
        labelNames: list of str of shape (n_labels,), default = None. Optional display names matching the labels (same order).
    Returns:
        report: string that prints all the required reports
    """
    outputs = F.softmax(logits, dim = 1)
    outputs = np.argmax(outputs.numpy(), axis=1)
    labels = labels.numpy()
    report = classification_report(labels, outputs, labelIdx, labelNames, digits = 4)

    return report

def getAUROC(logits, labels):
    """
    Returns the auroc given logits and labels for classification
    Args:
        logits: raw scores from NN. Dimension: (batchSize, numClasses) - output of the model
        labels: targets from dataset. Dimension: (batchSize, 1) where each element is the class index
    Returns:
        auroc: auroc value if binary, ovr (one-vs-rest if multiclass)
    """
    outputs = F.softmax(logits, dim = 1)
    outputs = outputs.numpy()

    labels = labels.numpy()
    numClasses = len(np.unique(labels))
    if numClasses > 2:
        return roc_auc_score(labels, outputs, multi_class='ovr')
    else: 
       return roc_auc_score(labels, outputs[:,1])

def getAccuracy(logits, labels):
    """
    Returns the accuracy given logits and labels for classification
    Args:
        logits: raw scores from NN. Dimension: (batchSize, numClasses) - output of the model
        labels: targets from dataset. Dimension: (batchSize, 1) where each element is the class index
    Returns:
        accuracy: 
    """
    outputs = F.softmax(logits, dim = 1)
    outputs = outputs.numpy()
    outputs = np.argmax(outputs, axis=1)

    labels = labels.numpy()
    
    return accuracy_score(labels, outputs)
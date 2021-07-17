""" 
Author: Nandita Bhaskhar
Torch model saving and loading functions
"""

import os
import shutil
import sys
sys.path.append('../')
from .utils import safeMkdir

import torch

def saveCheckpoint(state, isBest, checkpoint):
    """ Saves model and training parameters at checkpoint + 'last.pth.tar'. If isBest==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        isBest: (bool) True if it is the best model seen till now
        checkpoint: (string) path of folder where parameters are to be saved
    """
    safeMkdir(checkpoint)
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    # save state to last_checkpoint
    torch.save(state, filepath)

    # also save state to best_checkpoint if best
    if isBest:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def loadCheckpoint(checkpoint, model, optimizer=None):
    """ Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are to be loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    Return:
        checkpoint: (torch.checkpoint)
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    # load model state
    model.load_state_dict(checkpoint['state_dict'])
    # load optimizer state
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint
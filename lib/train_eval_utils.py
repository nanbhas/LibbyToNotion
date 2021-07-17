""" 
Author: Nandita Bhaskhar
Train and eval functions 
"""

import os, sys
sys.path.append('../')
import logging
from tqdm import tqdm
from subprocess import check_call

import numpy as np
import pandas as pd
import torch

from .utils import *
from .checkpoint_utils import loadCheckpoint, saveCheckpoint

from models.lossMetrics import *


def evaluate(model, device, lossFunc, dataloader, epoch = 0, labelIdx = None, labelNames = None):
    """ Evaluate the model on all batches for a given epoch.
    Args:
        model: (torch.nn.Module) the neural network
        device: (torch.device) GPU or CPU
        lossFunc: (a function that takes outputs and labels for a batch and computes the loss for the batch
        dataloader: (DataLoader) DataLoader object that fetches a data batch (inputBatch, labelsBatch)
        epoch: (scalar) epoch number in the numOfEpoch loop
        labelIdx: array-like of shape (n_labels,), default = None. Optional list of label indices to include in the report.
        labelNames: list of str of shape (n_labels,), default = None. Optional display names matching the labels (same order).
    Returns:
        netAccuracy: (float, numpy) net accuracy over one epoch
        netAUROC: (float, numpy) net AUROC over one epoch
        summLoss: (list of dicts) summary of losses over one epoch
                    dict keys are:  'type' (train/eval), 'name' (loss, report, etc), 
                                    'epochIdx' (epoch number), 'batchIdx' (batch number), 
                                    'value' (actual value)
        summReports: (list of dicts) summary of reports over one epoch
                    dict keys are:  'type' (train/eval), 'name' (loss, report, etc), 
                                    'epochIdx' (epoch number), 'batchIdx' (batch number), 
                                    'value' (actual value)
    """

    # set model to evaluation mode
    model.eval()

    # set metrics to track
    summLoss, summReports = [], []
    losses = []
    logits, labels = [], []
    
    # No gradient computation
    with torch.no_grad():
        with tqdm(total=len(dataloader)) as t:
            # iterate over the dataset
            for batchIdx, (dataBatch, labelsBatch, *rest) in enumerate(dataloader):

                # move to GPU if available
                dataBatch, labelsBatch = dataBatch.to(device), labelsBatch.to(device)

                # compute model output
                logitsBatch = model(dataBatch)

                # compute loss over batch
                loss = lossFunc(logitsBatch, labelsBatch)

                # move outputs and labels to cpu, convert to numpy arrays
                logitsBatch = logitsBatch.data.cpu()
                labelsBatch = labelsBatch.data.cpu()

                # append all batches together
                logits.append(logitsBatch)
                labels.append(labelsBatch)

                # track loss separately
                losses.append(loss.item())
                # track batch idx summaries
                summLoss.append({'type': 'eval', 'name': 'loss', 'epochIdx': epoch, 'batchIdx': batchIdx, 'value': loss.item()})
                summReports.append({'type': 'eval', 'name': 'report', 'epochIdx': epoch, 'batchIdx': batchIdx, 
                                        'value': getClassificationReport(logitsBatch, labelsBatch, labelIdx, labelNames) })
                t.update()
                
    # compute mean of all metrics in summary
    netLoss = np.mean(np.array(losses))
    catLogits = torch.cat(logits)
    catLabels = torch.cat(labels)
    netReport = getClassificationReport(catLogits, catLabels, labelIdx, labelNames)
    try:
        netAUROC = getAUROC(catLogits, catLabels)
    except:
        netAUROC = 0
        print("observed NaNs -- replacing with 0")
    netAccuracy = getAccuracy(catLogits, catLabels)

    # all the full epoch details 
    summLoss.append({'type': 'eval', 'name': 'loss', 'epochIdx': epoch, 'batchIdx': None, 'value': netLoss})
    summReports.append({'type': 'eval', 'name': 'report', 'epochIdx': epoch, 'batchIdx': None, 
                                    'value': netReport })

    # logging in logger
    logging.info("- Net Eval loss for epoch " + str(epoch) + " : " + str(netLoss))
    logging.info("- Net Eval accuracy for epoch " + str(epoch) + " : " + str(netAccuracy))
    logging.info("- Net Eval AUROC for epoch " + str(epoch) + " : " + str(netAUROC))
    logging.info("- Net Eval Report for epoch " + str(epoch) + " :\n\n " + str(netReport))

    return netAccuracy, netAUROC, summLoss, summReports, 


def train(model, device, lossFunc, optimizer, dataloader, epoch, iterMeter, logInterval, labelIdx = None, labelNames = None):
    """ Train the model on all batches for a given epoch.
    Args:
        model: (torch.nn.Module) the neural network
        device: (torch.device) GPU or CPU
        lossFunc: (a function that takes outputs and labels for a batch and computes the loss for the batch
        optimizer: (torch.optim) optimizer for parameters of model
        dataloader: (DataLoader) DataLoader object that fetches a (training) data batch (inputBatch, labelsBatch)
        epoch: (scalar) epoch number in the numOfEpoch loop
        iterMeter: (IterMeter) object that keeps track of step size
        logInterval: (int) interval to log values
        labelIdx: array-like of shape (n_labels,), default = None. Optional list of label indices to include in the report.
        labelNames: list of str of shape (n_labels,), default = None. Optional display names matching the labels (same order).
    Returns:
        netAccuracy: (float, numpy) net accuracy over one epoch
        netAUROC: (float, numpy) net AUROC over one epoch
        summLoss: (list of dicts) summary of losses over one epoch
                    dict keys are:  'type' (train/eval), 'name' (loss, report, etc), 
                                    'epochIdx' (epoch number), 'batchIdx' (batch number), 
                                    'step' (step number), 'value' (actual value)
        summReports: (list of dicts) summary of reports over one epoch
                    dict keys are:  'type' (train/eval), 'name' (loss, report, etc), 
                                    'epochIdx' (epoch number), 'batchIdx' (batch number), 
                                    'value' (actual value)
    """

    # set model to training mode
    model.train()

    # set metrics to track
    summLoss, summReports = [], []
    losses = []
    # running average object for training loss
    lossAvg = RunningAverage()
    # logits and labels
    logits, labels = [], []

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for batchIdx, (trainDataBatch, trainLabelsBatch, *rest) in enumerate(dataloader):
            
            # clear previous gradients
            optimizer.zero_grad()

            # move to GPU if available
            trainDataBatch, trainLabelsBatch = trainDataBatch.to(device), trainLabelsBatch.to(device)

            # compute model output
            logitsBatch = model(trainDataBatch)
            # compute loss over batch
            loss = lossFunc(logitsBatch, trainLabelsBatch)

            # compute gradients of all variables wrt loss
            loss.backward()
            # perform updates using calculated gradients
            optimizer.step()

            # track batch idx step summaries
            losses.append(loss.item())
            summLoss.append({'type': 'train', 'name': 'loss', 'epochIdx': epoch, 'batchIdx': batchIdx, 
                                        'value': loss.item(), 'step': iterMeter.get()})

            # increase step count
            iterMeter.step()

            # move outputs and labels to cpu, convert to numpy arrays
            logitsBatch = logitsBatch.data.cpu()
            trainLabelsBatch = trainLabelsBatch.data.cpu()

            # append all batches together
            logits.append(logitsBatch)
            labels.append(trainLabelsBatch)

            # Evaluate summaries only once in a while
            if batchIdx % logInterval == 0:
                # get batch idx summary reports once in a while
                summReports.append({'type': 'train', 'name': 'report', 'epochIdx': epoch, 'batchIdx': batchIdx, 
                                    'value': getClassificationReport(logitsBatch, trainLabelsBatch, labelIdx, labelNames) })

            # update the average loss
            lossAvg.update(loss.item())

            # display average loss in tqdm and update step
            t.set_postfix(loss='{:05.3f}'.format(lossAvg()))
            t.update()

    # compute mean of all metrics in summary
    netLoss = np.mean(np.array(losses))
    netReport = getClassificationReport(torch.cat(logits), torch.cat(labels), labelIdx, labelNames)
    try:
        netAUROC = getAUROC(torch.cat(logits), torch.cat(labels))
    except:
        netAUROC = 0
        print("observed NaNs -- replacing with 0")
    netAccuracy = getAccuracy(torch.cat(logits), torch.cat(labels))

    # all the full epoch details 
    summLoss.append({'type': 'train', 'name': 'loss', 'epochIdx': epoch, 'batchIdx': None, 'value': netLoss})
    summReports.append({'type': 'train', 'name': 'report', 'epochIdx': epoch, 'batchIdx': None, 
                                    'value': netReport })

    # logging in logger
    logging.info("- Net Train loss for epoch " + str(epoch) + " : " + str(netLoss))
    logging.info("- Net Train accuracy for epoch " + str(epoch) + " : " + str(netAccuracy))
    logging.info("- Net Train AUROC for epoch " + str(epoch) + " : " + str(netAUROC))
    logging.info("- Net Train Report for epoch " + str(epoch) + " :\n\n " + str(netReport))

    return netAccuracy, netAUROC, summLoss, summReports


def train_and_evaluate(model, device, lossFunc, optimizer, trainDataloader, valDataloader, iterMeter, 
                            numEpochs, logInterval, modelDir, restoreFile = None, labelIdx = None, labelNames = None, scheduler = None):
    """ Train the model and evaluate over all epochs. 
    Calls the following two fns:
        1) train(model, device, lossFunc, optimizer, dataloader, epoch, iterMeter, logInterval, labelIdx = None, labelNames = None)
        2) evaluate(model, device, lossFunc, dataloader, epoch = 0, labelIdx = None, labelNames = None)
    Args:
        model: (torch.nn.Module) the neural network
        device: (torch.device) GPU or CPU
        lossFunc: (a function that takes outputs and labels for a batch and computes the loss for the batch
        optimizer: (torch.optim) optimizer for parameters of model
        trainDataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        valDataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        iterMeter: (IterMeter) object that keeps track of step size
        numEpochs: (int) numEpochs to train for
        logInterval: (int) interval to log values
        modelDir: (string) directory containing config, weights and log
        restoreFile: (string) optional- name of file to restore from (without its extension .pth.tar)
        labelIdx: array-like of shape (n_labels,), default = None. Optional list of label indices to include in the report.
        labelNames: list of str of shape (n_labels,), default = None. Optional display names matching the labels (same order).
        scheduler (opt) scheduler 
    Returns:
        trainAcc, trainAUROC, trainSummLoss, trainSummReports, valAcc, valAUROC, valSummLoss, valSummReports
    """
    # reload weights from restoreFile if specified
    if restoreFile:
        restorePath = os.path.join(modelDir, restoreFile + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restorePath))
        loadCheckpoint(restorePath, model, optimizer)

    bestValAcc = 0.0
    trainAcc, trainAUROC, trainSummLoss, trainSummReports = [], [], [], []
    valAcc, valAUROC, valSummLoss, valSummReports = [], [], [], []

    for epoch in range(numEpochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, numEpochs))

        # Train the model over one epoch (one full pass over the training set)
        trAcc, trAUROC, trSummLoss, trSummReports = train(model, device, lossFunc, optimizer, trainDataloader, epoch + 1, iterMeter, logInterval, labelIdx, labelNames)
        # Append values for book-keeping
        trainAcc.append(trAcc)
        trainAUROC.append(trAUROC)
        trainSummLoss.extend(trSummLoss)
        trainSummReports.extend(trSummReports)

        # Evaluate for one epoch on validation set
        vAcc, vAUROC, vSummLoss, vSummReports = evaluate(model, device, lossFunc, valDataloader, epoch + 1, labelIdx, labelNames)
        # Append values for book-keeping
        valAcc.append(vAcc)
        valAUROC.append(vAUROC)
        valSummLoss.extend(vSummLoss)
        valSummReports.extend(vSummReports)

        # scheduler step
        if scheduler is not None:
            scheduler.step()

        isBest = vAcc > bestValAcc

        # Save weights
        saveCheckpoint({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optim_dict': optimizer.state_dict()},
                        isBest = isBest,
                        checkpoint = modelDir)

        # If bestEval, bestSavePath
        if isBest:
            logging.info("- Found new best accuracy")
            bestValAcc = vAcc
                
            # Save best val metrics in a csv file in the model directory
            tmp = pd.DataFrame.from_dict({'accuracy': [vAcc], 'auroc': [vAUROC]})
            tmp = tmp.rename_axis("epoch")
            tmp.to_csv(os.path.join(modelDir, "acc_auroc_val_best.csv"))
            tmp = pd.DataFrame(valSummLoss)
            tmp.to_csv(os.path.join(modelDir, "loss_val_best.csv"))
            tmp = pd.DataFrame(valSummReports)
            tmp.to_csv(os.path.join(modelDir, "reports_val_best.csv"))
            # saveDictToJson(valSummLoss, os.path.join(modelDir, "loss_val_best.json"))
            # saveDictToJson(valSummReports, os.path.join(modelDir, "reports_val_best.json"))

        # Save latest val metrics in a csv file in the model directory
        tmp = pd.DataFrame.from_dict({'accuracy': [vAcc], 'auroc': [vAUROC]})
        tmp = tmp.rename_axis("epoch")
        tmp.to_csv(os.path.join(modelDir, "acc_auroc_val_last.csv"))
        tmp = pd.DataFrame(valSummLoss)
        tmp.to_csv(os.path.join(modelDir, "loss_val_last.csv"))
        tmp = pd.DataFrame(valSummReports)
        tmp.to_csv(os.path.join(modelDir, "reports_val_last.csv"))

    return trainAcc, trainAUROC, trainSummLoss, trainSummReports, valAcc, valAUROC, valSummLoss, valSummReports


def launch_training_job(modelDirRoot, dataDir, restoreFile, settingsFile, expName, projectName, params):
    """ Launch training of the model with a set of hyperparameters in `modelDirParent/jobName`
    Args:
        modelDirParent: (string) directory containing config, weights and log
        dataDir: (string) directory containing the dataset
        restoreFile: (string) path to a saved checkpoint
        settingsFile: (string) path to the settings file
        jobName: (string) unique job name for exp directory
        params: (dict) containing hyperparameters
    """
    # Create a new folder in modelDirRoot with unique_name "expName"
    expDir = os.path.join(modelDirRoot, expName)
    safeMkdir(expDir)

    # Write parameters in json file
    jsonPath = os.path.join(expDir, 'params.json')
    params.save(jsonPath)

    PYTHON = sys.executable
    
    # Launch training with this config
    cmd = "{python} main_training.py --expName {expName} --dataDir {dataDir} "
    cmd = cmd + "--modelDirRoot {modelDirRoot} --settingsFile {settingsFile} "
    cmd = cmd + "--projectName {projectName} --restoreFile {restoreFile} "
    cmd = cmd.format(python=PYTHON, expName = expName, dataDir = dataDir,
                        modelDirRoot = modelDirRoot, settingsFile = settingsFile,
                        projectName = projectName, restoreFile = restoreFile)
    print(cmd)
    check_call(cmd, shell=True)   


def getOutputsAndActivations(model, device, dataloader):
    """
    Args:
        model: (torch.nn.Module) the neural network (loaded from a checkpoint already)
        device: (torch.device) GPU or CPU
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
    Returns:
        actualLabels: (np.array) shape: (N,) actual labels
        predictedLabels: (np.array) shape: (N,) predicted labels
        activationsLinear: (np.array) shape: (N,K) activations from linear layer
        activationsLogReg: (np.array) shape: (N,K) activations from logreg layer. 
                                                    (here): corresponds to the prob. outputs of the model
                                                    Taking argmax should give predicted labels
    """
    activation = {}
    def getActivation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # register forward hooks
    model.model[0].register_forward_hook(getActivation('linearLayer'))
    model.model[1].register_forward_hook(getActivation('logRegLayer'))

    actualLabels, predictedLabels = [], []
    activationsLinear, activationsLogReg = [], []
    for dataBatch, labelsBatch, _, _, _, _, _, _ in dataloader:
        # subtract mean and divide by std
        dataBatch = (dataBatch - mean) / std
        # move to GPU if available
        dataBatch = dataBatch.to(device)
        # compute model output
        outputsBatch = model(dataBatch)
        # compute activations
        activationsLinear.append(activation['linearLayer'].data.cpu().numpy())
        activationsLogReg.append(activation['logRegLayer'].data.cpu().numpy())
        # move to cpu, convert to numpy arrays
        actualLabels.append(labelsBatch.data.cpu().numpy())
        outputsBatch = outputsBatch.data.cpu().numpy()
        # get the predicted labels
        predictedLabels.append(np.argmax(outputsBatch, axis=1)) # max value of the output probability scores
    
    actualLabels = np.concatenate(actualLabels)
    predictedLabels = np.concatenate(predictedLabels)
    activationsLinear = np.concatenate(activationsLinear)
    activationsLogReg = np.concatenate(activationsLogReg)

    return actualLabels, predictedLabels, activationsLinear, activationsLogReg
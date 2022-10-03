# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# October 2022
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.ops as ops
import SimpleITK as sitk
from ImageBatcher import ImageBatcher
from roc import ComputeROC
from rcc_common import LoadImage, LoadMask, LoadMaskNoRelabel, SaveImage, ComputeLabelWeights, ShowWarnings, ExtractTumorDetections, CleanUpMask
from UNet import UNet
from Deterministic import NotDeterministic
from TumorDSC import CalculateTumorDSC

class RCCSeg:
    def __init__(self, numClasses=4):
        self.device = "cpu"
        self.multipleOf = 16
        self.numClasses=numClasses
        self.net = UNet(in_channels=4,out_channels=self.numClasses, activation=nn.LeakyReLU(negative_slope=0.2))
        self.dataRoot = None
        self.valSteps = 50
        self.dilateUnknown = False
        self.window_shape = (32,)*3
        self.strides = (16,)*3
        self.cropKidney = False

    def SetDevice(self, device):
        self.device = device
        self.net = self.net.to(device)

    def GetDevice(self):
        return self.device

    def SetDataRoot(self, dataRoot):
        self.dataRoot = dataRoot

    def GetDataRoot(self):
        return self.dataRoot

    def SaveModel(self, fileName):
        torch.save(self.net.state_dict(), fileName)

    def LoadModel(self, fileName):
        self.net.load_state_dict(torch.load(fileName, map_location=self.GetDevice()))

    def RunOne(self,patientId):
        #volumePath = os.path.join(self.dataRoot, "Images", patientId, "image.nii.gz")
        #volumePath = os.path.join(self.dataRoot, "Images", patientId, "normalized_aligned.nii.gz")
        #volume2Path = os.path.join(self.dataRoot, "Images", patientId, "normalized2_aligned.nii.gz")

        sitkVolumes = [None]*4
        npVolumes = [None]*len(sitkVolumes)

        for i in range(len(npVolumes)):
            maskFileName = "normalized_aligned.nii.gz" if i == 0 else f"normalized{i+1}_aligned.nii.gz"
            volumePath = os.path.join(self.dataRoot, "Images", patientId, maskFileName)

            sitkVolumes[i] = LoadImage(volumePath)

            if sitkVolumes[i] is None:
                return None

            npVolumes[i] = sitk.GetArrayViewFromImage(sitkVolumes[i])

            if npVolumes[0].shape != npVolumes[i].shape:
                raise RuntimeError("Error: Dimension mismatch between volumes ({npVolumes[0].shape} != {i}: {npVolumes[i].shape}).")


        halfXDim = int(npVolumes[0].shape[2]/2)

        npVolumeRight = np.concatenate(tuple((volume[None,:,:,:halfXDim] for volume in npVolumes)), axis=0)
        npVolumeLeft = np.concatenate(tuple((volume[None,:,:,halfXDim:] for volume in npVolumes)), axis=0)

        strides = [ int(0.9*w) for w in self.window_shape ]

        npPatchesRight = np.lib.stride_tricks.sliding_window_view(npVolumeRight, window_shape=self.window_shape, axis=(1,2,3))
        npPatchesRight = npPatchesRight[:, ::strides[0], ::strides[1], ::strides[2], ...]
        npPatchesRight = np.reshape(npPatchesRight, [npPatchesRight.shape[0], -1] + list(npPatchesRight.shape[4:]))
        npPatchesRight = npPatchesRight.transpose((1,0,2,3,4))

        npPatchesLeft = np.lib.stride_tricks.sliding_window_view(npVolumeLeft, window_shape=self.window_shape, axis=(1,2,3))
        npPatchesLeft = npPatchesLeft[:, ::strides[0], ::strides[1], ::strides[2], ...]
        npPatchesLeft = np.reshape(npPatchesLeft, [npPatchesLeft.shape[0], -1] + list(npPatchesLeft.shape[4:]))
        npPatchesLeft = npPatchesLeft.transpose((1,0,2,3,4))

        npProbMap = np.zeros([self.numClasses] + list(npVolumes[0].shape), dtype=np.float32)
        npCountMap = np.zeros(npVolumes[0].shape, dtype=np.int64)

        npProbMapRight = npProbMap[:, :, :, :halfXDim]
        npCountMapRight = npCountMap[:, :, :halfXDim]

        npProbMapLeft = npProbMap[:, :, :, halfXDim:]
        npCountMapLeft = npCountMap[:, :, halfXDim:]
 
        npProbMapRight = np.lib.stride_tricks.sliding_window_view(npProbMapRight, window_shape=self.window_shape, axis=(1,2,3), writeable=True)
        npProbMapRight = npProbMapRight[:, ::strides[0], ::strides[1], ::strides[2], ...]
        #npProbMapRight = np.reshape(npProbMapRight, [npProbMapRight.shape[0], -1] + list(npProbMapRight.shape[4:]))

        npCountMapRight = np.lib.stride_tricks.sliding_window_view(npCountMapRight, window_shape=self.window_shape, writeable=True)
        npCountMapRight = npCountMapRight[::strides[0], ::strides[1], ::strides[2], ...]
        #npCountMapRight = np.reshape(npCountMapRight, [-1] + list(npCountMapRight.shape[3:]))

        npProbMapLeft = np.lib.stride_tricks.sliding_window_view(npProbMapLeft, window_shape=self.window_shape, axis=(1,2,3), writeable=True)
        npProbMapLeft = npProbMapLeft[:, ::strides[0], ::strides[1], ::strides[2], ...]
        #npProbMapLeft = np.reshape(npProbMapLeft, [npProbMapLeft.shape[0], -1] + list(npProbMapLeft.shape[4:]))

        npCountMapLeft = np.lib.stride_tricks.sliding_window_view(npCountMapLeft, window_shape=self.window_shape, writeable=True)
        npCountMapLeft = npCountMapLeft[::strides[0], ::strides[1], ::strides[2], ...]
        #npCountMapLeft = np.reshape(npCountMapLeft, [-1] + list(npCountMapLeft.shape[3:]))

        softmax = nn.Softmax(dim=1).to(self.GetDevice())

        with torch.no_grad():
            self.net.eval()

            i = 0
            for z in range(npCountMapRight.shape[0]):
                for y in range(npCountMapRight.shape[1]):
                    for x in range(npCountMapRight.shape[2]):
                        batch = torch.from_numpy(npPatchesRight[i, ...]).type(torch.float).to(self.GetDevice()).unsqueeze(0)
                        #npProbMapRight[:, z,y,x, ...] += softmax(self.net(batch)).cpu().numpy()[0,...]
                        npProbMapRight[:, z,y,x, ...] = np.maximum(npProbMapRight[:, z,y,x, ...], softmax(self.net(batch)).cpu().numpy()[0,...])
                        npCountMapRight[z,y,x, ...] += 1
                        i += 1

            i = 0
            for z in range(npCountMapLeft.shape[0]):
                for y in range(npCountMapLeft.shape[1]):
                    for x in range(npCountMapLeft.shape[2]):
                        batch = torch.from_numpy(npPatchesLeft[i, ...]).type(torch.float).to(self.GetDevice()).unsqueeze(0)
                        npProbMapLeft[:, z,y,x, ...] = np.maximum(npProbMapLeft[:, z,y,x, ...], softmax(self.net(batch)).cpu().numpy()[0,...])
                        npCountMapLeft[z,y,x, ...] += 1
                        i += 1

        if npCountMap.max() == 0:
            raise RuntimeError("CountMap is also 0. This might mean memory is not being shared between left/right slices.")

        npCountMap[npCountMap == 0] = 1
        #npProbMap /= npCountMap

        npProbMap = npProbMap.transpose(1,2,3,0)
        npLabelMap = npProbMap.argmax(axis=3).astype(np.int16)

        sitkProbMap = sitk.GetImageFromArray(npProbMap)
        sitkLabelMap = sitk.GetImageFromArray(npLabelMap)

        sitkProbMap.SetSpacing(sitkVolumes[0].GetSpacing())
        sitkProbMap.SetDirection(sitkVolumes[0].GetDirection())
        sitkProbMap.SetOrigin(sitkVolumes[0].GetOrigin())

        sitkLabelMap.SetSpacing(sitkVolumes[0].GetSpacing())
        sitkLabelMap.SetDirection(sitkVolumes[0].GetDirection())
        sitkLabelMap.SetOrigin(sitkVolumes[0].GetOrigin())

        return sitkProbMap, sitkLabelMap

    def Test(self,valList):
        if isinstance(valList, str):
            with open(valList, "rt", newline='') as f:
                patientIds = [ patientId.strip() for patientId in f if len(patientId.strip()) > 0 ]
        else:
            patientIds = valList

        allScores = None
        allLabels = None

        allDices = dict()
        allTumorDices = dict()
        npDices = np.zeros([len(patientIds), self.numClasses])

        npTumorDices = None

        npConfusion = np.zeros([self.numClasses, self.numClasses], dtype=np.int64)

        i = 0
        for patientId in patientIds:
            print(f"Info: Running '{patientId}' ...")

            maskFile = os.path.join(self.GetDataRoot(), "Masks", patientId, "mask_aligned.nii.gz")

            # First DICE!
            gtMask = LoadMask(maskFile, self.numClasses, dilateUnknown=False)
            npGtMask = sitk.GetArrayFromImage(gtMask)

            probMap, labelMap = self.RunOne(patientId)

            npProbMap = sitk.GetArrayFromImage(probMap)
            npProbMap = npProbMap[:,:,:,-1]

            if self.numClasses == 4:
                labelMap = CleanUpMask(labelMap)
                npLabelMap = sitk.GetArrayViewFromImage(labelMap) # XXX: Very slow without making it numpy
                npProbMap[npLabelMap == 0] = 0

            npTmpTumorDices, _ = CalculateTumorDSC(npLabelMap, npGtMask, tumorLabel=(3 if self.numClasses == 4 else 1))

            print(npTmpTumorDices)

            if npTmpTumorDices.size > 0:
                allTumorDices[patientId] = npTmpTumorDices[:,1].copy()
                print(allTumorDices[patientId].shape)

                if npTumorDices is None:
                    npTumorDices = npTmpTumorDices
                else:
                    npTumorDices = np.concatenate((npTumorDices, npTmpTumorDices), axis=0)

            npLabelMap = sitk.GetArrayFromImage(labelMap)

            npTmpConfusion = np.zeros([self.numClasses, self.numClasses], dtype=np.int64)
            for actual in range(self.numClasses):
                for predicted in range(self.numClasses):
                    npTmpConfusion[actual, predicted] = np.logical_and((npGtMask == actual), (npLabelMap == predicted)).sum()

            print(f"patient confusion matrix =\n{npTmpConfusion}")

            npConfusion += npTmpConfusion

            tumorLabel = 3 if self.numClasses > 2 else 1
            #print(f"Sanity check, tumor sum: {(npGtMask == tumorLabel).sum()}")
            npTumorPredictionProfile = npTmpConfusion[tumorLabel, :]
            tumorPredictionCount = npTumorPredictionProfile.sum()
            #print(f"Count = {tumorPredictionCount}")

            if tumorPredictionCount > 0:
                npTumorPredictionProfile = npTumorPredictionProfile / tumorPredictionCount

            print(f"tumor prediction profile: {npTumorPredictionProfile}")

            npLabelMap[npGtMask == 255] = 0
            npGtMask[npGtMask == 255] = 0

            halfX = int(npGtMask.shape[2]/2)

            npRightGtMask = npGtMask[:,:,:halfX]
            npLeftGtMask = npGtMask[:,:,halfX:]

            npRightLabelMap = npLabelMap[:,:,:halfX]
            npLeftLabelMap = npLabelMap[:,:,halfX:]

            for label in range(1,self.numClasses):
                AintB, A, B = 0.0, 0.0, 0.0

                if npRightGtMask.max() > 0:
                    AintB += np.sum(np.logical_and((npRightGtMask == label), (npRightLabelMap == label)))
                    A += np.sum(npRightGtMask == label)
                    B += np.sum(npRightLabelMap == label)

                if npLeftGtMask.max() > 0:
                    AintB += np.sum(np.logical_and((npLeftGtMask == label), (npLeftLabelMap == label)))
                    A += np.sum(npLeftGtMask == label)
                    B += np.sum(npLeftLabelMap == label)

                dice = 1.0 if A+B <= 0.0 else 2.0 * AintB / ( A + B )

                #if AintB == 0.0:
                #    print(f"A = {A}, B = {B}")

                if patientId not in allDices:
                    allDices[patientId] = [ -1 ]*self.numClasses

                allDices[patientId][label] = dice
                npDices[i, label] = dice

                print(f"{label}: dice = {dice}")

            if npTmpTumorDices.size > 0:
                print(f"Tumor dice: {npTmpTumorDices.mean(axis=0)[1]} +/- {npTmpTumorDices.std(axis=0)[1]}")

            i += 1

            # Now AUC
            gtMask, labelDict = LoadMaskNoRelabel(maskFile)
            npGtMask = sitk.GetArrayFromImage(gtMask)

            npGtMask[npGtMask == 255] = 0

            npRightGtMask = npGtMask[:,:,:halfX]
            npLeftGtMask = npGtMask[:,:,halfX:]

            npRightProbMap = npProbMap[:,:,:halfX]
            npLeftProbMap = npProbMap[:,:,halfX:]

            if npRightGtMask.max() > 0:
                rightProbMap = sitk.GetImageFromArray(npRightProbMap)
                rightProbMap.SetSpacing(probMap.GetSpacing())

                rightGtMask = sitk.GetImageFromArray(npRightGtMask)
                rightGtMask.SetSpacing(probMap.GetSpacing())

                scores, labels = ExtractTumorDetections(rightProbMap, rightGtMask, labelDict)

                if allScores is None:
                    allScores = scores
                    allLabels = labels
                else:
                    allScores = np.concatenate((allScores, scores), axis=0)
                    allLabels = np.concatenate((allLabels, labels), axis=0)

            if npLeftGtMask.max() > 0:
                leftProbMap = sitk.GetImageFromArray(npLeftProbMap)
                leftProbMap.SetSpacing(probMap.GetSpacing())

                leftGtMask = sitk.GetImageFromArray(npLeftGtMask)
                leftGtMask.SetSpacing(probMap.GetSpacing())

                scores, labels = ExtractTumorDetections(leftProbMap, leftGtMask, labelDict)

                if allScores is None:
                    allScores = scores
                    allLabels = labels
                else:
                    allScores = np.concatenate((allScores, scores), axis=0)
                    allLabels = np.concatenate((allLabels, labels), axis=0)

        avgDice = [-1]*self.numClasses
        stdDice = [-1]*self.numClasses
        medDice = [-1]*self.numClasses

        for label in range(1,self.numClasses):
            npMask = (npDices[:,label] >= 0)

            if not npMask.any():
                continue

            avgDice[label] = npDices[npMask, label].mean()
            stdDice[label] = npDices[npMask, label].std()
            medDice[label] = np.median(npDices[npMask, label])

        roc = ComputeROC(torch.from_numpy(allScores), torch.from_numpy(allLabels))

        return (avgDice, stdDice, medDice), allDices, roc, (npTumorDices.mean(axis=0)[1], npTumorDices.std(axis=0)[1], np.median(npTumorDices, axis=0)[1]), allTumorDices, npConfusion

    def Train(self,trainList,valPerc=0.0,snapshotRoot="snapshots"):
        batchSize=16
        labelWeights = torch.Tensor([1.0]*self.numClasses)
        ShowWarnings(False)
        numEpochs=1000

        print(f"Info: numClasses = {self.numClasses}, dilateUnknown = {self.dilateUnknown}, window_shape = {self.window_shape}, strides = {self.strides}, cropKidney = {self.cropKidney}")

        with open(trainList, mode="rt", newline="") as f:
            patientIds = [ patientId.strip() for patientId in f if len(patientId.strip()) > 0 ]

        valList = None
        if valPerc > 0.0:
            mid = max(1, int(valPerc*len(patientIds)))
            valList = patientIds[:mid]
            trainList = patientIds[mid:]
        else:
            trainList = patientIds

        imageBatcher = ImageBatcher(self.GetDataRoot(), trainList, batchSize, window_shape=self.window_shape, strides=self.strides, numClasses=self.numClasses, dilateUnknown=self.dilateUnknown, cropKidney=self.cropKidney)

        criterion = nn.CrossEntropyLoss(ignore_index=-1,weight = labelWeights).to(self.GetDevice())

        #criterion = ops.sigmoid_focal_loss
        optimizer = optim.Adam(self.net.parameters(), lr = 1e-3)

        trainLosses = np.ones([numEpochs])*1000.0
        valAUCs = np.zeros([numEpochs])

        if not os.path.exists(snapshotRoot):
            os.makedirs(snapshotRoot)

        for e in range(numEpochs):
            imageBatcher.shuffle()

            runningLoss = 0.0
            count = 0

            for xbatch, ybatch in imageBatcher:
                #print(xbatch.shape)

                xbatch = xbatch.to(self.GetDevice())
                ybatch = ybatch.to(self.GetDevice())

                optimizer.zero_grad()            

                outputs = self.net(xbatch)

                with NotDeterministic():
                    loss = criterion(outputs, ybatch)

                    loss.backward()

                optimizer.step()

                runningLoss += loss
                count += 1

                print(f"loss = {loss.item()}", flush=True)

            if count > 0:
                runningLoss /= count

            snapshotFile=os.path.join(snapshotRoot, f"epoch_{e}.pt")
            rocFile=os.path.join(snapshotRoot, f"validation_roc_{e}.txt")
            diceFile=os.path.join(snapshotRoot, f"dice_stats_{e}.txt")

            print(f"Info: Saving {snapshotFile} ...", flush=True)
            self.SaveModel(snapshotFile)

            # For debugging
            #self.LoadModel(snapshotFile)

            trainLosses[e] = runningLoss

            if valList is None:
                print(f"Info: Epoch = {e}, training loss = {runningLoss}", flush=True)
            elif self.valSteps > 0 and ((e+1) % self.valSteps) == 0: 
                diceStats, allDices, roc, tumorDices = self.Test(valList)
                print(f"Info: Epoch = {e}, training loss = {runningLoss}, validation AUC = {roc[3]}, validation dices = {diceStats[0]} +/- {diceStats[1]}, tumor validation dices = {tumorDices[0]} +/- {tumorDices[1]}", flush=True)

                valAUCs[e] = roc[3]

                with open(rocFile, mode="wt", newline="") as f:
                    f.write("# Threshold\tFPR\tTPR\n")

                    for threshold, fpr, tpr in zip(roc[0], roc[1], roc[2]):
                        f.write(f"{threshold}\t{fpr}\t{tpr}\n")

                    f.write(f"# AUC = {roc[3]}\n")

                with open(diceFile, mode="wt", newline="") as f:
                    for patientId in allDices:
                        f.write(f"{patientId}: {allDices[patientId]}\n")

                    f.write(f"\nDice stats: {diceStats[0]} +/- {diceStats[1]}\n")
 
        return trainLosses, valAUCs


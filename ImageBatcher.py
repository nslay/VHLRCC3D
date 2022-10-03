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
import random
import SimpleITK as sitk
import numpy as np
from rcc_common import LoadImage, SaveImage, LoadMask, ShowWarnings
import torch

class ImageBatcher:
    def __init__(self, dataRoot, listFile, batchSize, window_shape, strides, numClasses=4, dilateUnknown=False, cropKidney=False):
        ShowWarnings(False)
        self.dataRoot = dataRoot
        self.multipleOf = 16
        self.numChannels = 4
        self.batchSize = batchSize
        self.numClasses = numClasses
        self.dilateUnknown = dilateUnknown
        self.window_shape = window_shape
        self.strides = strides
        self.cropKidney = cropKidney

        if isinstance(listFile, str):
            self._load_patient_ids(listFile)
        else:
            self.patientIds = listFile

    def seed(self, value):
        random.seed(value)

    def shuffle(self):
        random.shuffle(self.patientIds)

    def __iter__(self):
        self._patient_index = 0
        self._volume_index = 0
        self._batch_index = 0
        self._volume_pairs = None
        return self

    def __next__(self):
        while not self._volume_pairs or self._volume_index >= len(self._volume_pairs):

            if self._patient_index >= len(self.patientIds):
                raise StopIteration

            self._volume_pairs = self._load_patient(self.patientIds[self._patient_index])
            self._patient_index += 1
            self._volume_index = 0
            self._patch_index = random.randint(0,self.batchSize-1)

        npVolume = self._volume_pairs[self._volume_index][0]
        npMask = self._volume_pairs[self._volume_index][1]

        npImageBatch = np.zeros([ self.batchSize ] + list(npVolume.shape[1:]), dtype=np.float32)
        npMaskBatch = np.zeros([ self.batchSize ] + list(npMask.shape[1:]), dtype=np.int32)

        if self._patch_index >= npVolume.shape[0]:
            self._patch_index = random.randint(0,npVolume.shape[0]-1)
            #print("This happened?")

        ibegin = self._patch_index
        obegin = 0
        while obegin < self.batchSize:
            iend = min(ibegin + (self.batchSize - obegin), npVolume.shape[0])
            oend = obegin + (iend - ibegin)
            #print(f"input: {ibegin}, {iend}")
            #print(f"output: {obegin}, {oend}")

            npImageBatch[obegin:oend,...] = npVolume[ibegin:iend, ...]
            npMaskBatch[obegin:oend,...] = npMask[ibegin:iend, ...]

            obegin = oend
            ibegin = (iend % npVolume.shape[0])
            self._patch_index = max(self._patch_index, iend)

        """
        if self._patch_index + self.batchSize <= npVolume.shape[0]:
            begin = self._patch_index
            end = begin + self.batchSize

            npImageBatch[:,:,:,:,:] = npVolume[begin:end,:,:,:,:]
            npMaskBatch[:,:,:,:] = npMask[begin:end,:,:,:]

            self._patch_index = end
        else:
            begin = self._patch_index
            end = npVolume.shape[0]
            offset = end-begin
            print(f"{begin}, {end}, {offset}")
            npImageBatch[:offset,:,:,:,:] = npVolume[begin:end,:,:,:,:]
            npMaskBatch[:offset,:,:,:] = npMask[begin:end,:,:,:]

            begin = 0
            end = self.batchSize - offset
            npImageBatch[offset:,:,:,:,:] = npVolume[begin:end,:,:,:,:]
            npMaskBatch[offset:,:,:,:] = npMask[begin:end,:,:,:]

            self._patch_index = npVolume.shape[0]
        """

        if self._patch_index >= npVolume.shape[0]:
            self._volume_index += 1
            self._patch_index = random.randint(0,self.batchSize-1)

        imageBatch = torch.from_numpy(npImageBatch)
        maskBatch = torch.from_numpy(npMaskBatch).type(torch.long)

        return imageBatch, maskBatch


    def _load_patient_ids(self, listFile):
        self.patientIds = []

        with open(listFile, mode="rt", newline="") as f:
            self.patientIds = [ line.strip() for line in f if len(line.strip()) > 0 ]

    # Assume numpy convention
    def _get_roi_1d(self, size):
        remainder = (size % self.multipleOf)

        begin = int(remainder/2)
        end = begin + size - remainder

        return begin, end

    def _resize_image(self, npImg):
        return npImg
        beginX, endX = self._get_roi_1d(npImg.shape[-1])
        beginY, endY = self._get_roi_1d(npImg.shape[-2])
        beginZ, endZ = self._get_roi_1d(npImg.shape[-3])

        return npImg[beginZ:endZ, beginY:endY, beginX:endX].copy()

    def _crop_image_and_mask(self, npImg, npMask):
        if not self.cropKidney:
            return npImg, npMask

        indices = np.argwhere(npMask > 0)
        lower = indices.min(axis=0)
        upper = indices.max(axis=0)

        lower -= np.array(self.window_shape)
        upper += np.array(self.window_shape)

        lower = np.maximum(lower, 0)
        upper = np.minimum(upper, np.array(npMask.shape))

        return npImg[..., lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]].copy(), npMask[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]].copy()

    def _load_patient(self, patientId):
        imageFiles = [ f"normalized{i+1}_aligned.nii.gz" if i > 0 else "normalized_aligned.nii.gz" for i in range(self.numChannels) ]
        imageFiles = [ os.path.join(self.dataRoot, "Images", patientId, imageFile) for imageFile in imageFiles ]

        maskFile = os.path.join(self.dataRoot, "Masks", patientId, "mask_aligned.nii.gz")

        sitkVolumes = [ LoadImage(imageFile) for imageFile in imageFiles ]

        if not all(sitkVolumes):
            return

        npVolumes = [ sitk.GetArrayViewFromImage(sitkVolume) for sitkVolume in sitkVolumes ]

        if not all([npVolumes[0].shape == npVolume.shape for npVolume in npVolumes]):
            return

        sitkMask = LoadMask(maskFile, numClasses=self.numClasses, dilateUnknown=self.dilateUnknown)

        if not sitkMask:
            return

        npMask = sitk.GetArrayViewFromImage(sitkMask).astype(np.int32) # XXX: Prevent issue with -1

        if npMask.shape != npVolumes[0].shape:
            return

        npMask[npMask == 255] = -1
        
        # TODO: Review this
        #npMask[np.logical_and(npVolumes[0] < 650, npMask == 1)] = -1
        npMask[np.logical_and(npVolumes[0] < 253, npMask == 1)] = -1

        for npVolume in npVolumes:
            #npMask[npVolume < 209] = -1
            npMask[npVolume < -188] = -1
        ######

        halfX = int(npVolumes[0].shape[-1]/2)

        npMaskRight = self._resize_image(npMask[:,:,:halfX])
        npMaskLeft = self._resize_image(npMask[:,:,halfX:])

        pairs = []

        #if npMaskRight.any():
        if npMaskRight.max() > 0:
            npVolumesRight = [ self._resize_image(npVolume[:,:,:halfX]) for npVolume in npVolumes ]
            npCombinedRight = np.concatenate(tuple((volume[None, ...] for volume in npVolumesRight)), axis=0)

            #print(npCombinedRight.shape)

            npCombinedRight, npMaskRight = self._crop_image_and_mask(npCombinedRight, npMaskRight)

            #print(npCombinedRight.shape)
            #print(npMaskRight.shape)

            npCombinedRight = np.lib.stride_tricks.sliding_window_view(npCombinedRight, window_shape=self.window_shape, axis=(1,2,3))
            npCombinedRight = npCombinedRight[:, ::self.strides[0], ::self.strides[1], ::self.strides[2], ...]
            npCombinedRight = np.reshape(npCombinedRight, [npCombinedRight.shape[0], -1] + list(npCombinedRight.shape[4:]))
            npCombinedRight = npCombinedRight.transpose((1,0,2,3,4))

            npMaskRight = np.lib.stride_tricks.sliding_window_view(npMaskRight, window_shape=self.window_shape)
            npMaskRight = npMaskRight[::self.strides[0], ::self.strides[1], ::self.strides[2], ...]
            npMaskRight = np.reshape(npMaskRight, [-1] + list(npMaskRight.shape[3:]))

            pairs.append((npCombinedRight, npMaskRight))
            pairs.append((npCombinedRight[:,:,:,:,::-1].copy(), npMaskRight[:,:,:,::-1].copy()))

        #if npMaskLeft.any():
        if npMaskLeft.max() > 0:
            npVolumesLeft = [ self._resize_image(npVolume[:,:,:halfX]) for npVolume in npVolumes ]
            npCombinedLeft = np.concatenate(tuple((volume[None, ...] for volume in npVolumesLeft)), axis=0)

            #print(npCombinedLeft.shape)

            npCombinedLeft, npMaskLeft = self._crop_image_and_mask(npCombinedLeft, npMaskLeft)

            #print(npCombinedLeft.shape)
            #print(npMaskLeft.shape)

            npCombinedLeft = np.lib.stride_tricks.sliding_window_view(npCombinedLeft, window_shape=self.window_shape, axis=(1,2,3))
            npCombinedLeft = npCombinedLeft[:, ::self.strides[0], ::self.strides[1], ::self.strides[2], ...]
            npCombinedLeft = np.reshape(npCombinedLeft, [npCombinedLeft.shape[0], -1] + list(npCombinedLeft.shape[4:]))
            npCombinedLeft = npCombinedLeft.transpose((1,0,2,3,4))

            npMaskLeft = np.lib.stride_tricks.sliding_window_view(npMaskLeft, window_shape=self.window_shape)
            npMaskLeft = npMaskLeft[::self.strides[0], ::self.strides[1], ::self.strides[2], ...]
            npMaskLeft = np.reshape(npMaskLeft, [-1] + list(npMaskLeft.shape[3:]))

            pairs.append((npCombinedLeft, npMaskLeft))
            pairs.append((npCombinedLeft[:,:,:,:,::-1].copy(), npMaskLeft[:,:,:,::-1].copy()))

        return pairs

if __name__ == "__main__":
    dataRoot="/data/AIR/RCC/NiftiNew3D"
    listFile=os.path.join(dataRoot, "all.txt")
    listFile="/data/AIR/RCC/NiftiNew3D/train_easyhard_randomSplit1.txt"

    batcher = ImageBatcher(dataRoot, listFile, 16, window_shape=(16,32,32), strides=(8,16,16), numClasses=4, dilateUnknown=False, cropKidney=True)

    ShowWarnings(True)

    batcher.seed(7271)

    for batch in batcher:
        imageBatch, maskBatch = batch
        print(f"{type(imageBatch)}, {type(maskBatch)}")
        print(f"{imageBatch.shape}, {maskBatch.shape}")


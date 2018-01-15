import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os, sys
from os import walk
import json
from scipy.cluster.vq import vq, kmeans, whiten
# from scipy.stats import mode

descriptorMapping = [ 'lowLevel.spectral_centroid.mean',# 0
                      'lowLevel.spectral_centroid.stdev', #1
                      'lowLevel.dissonance.mean', #2
                      'lowLevel.dissonance.stdev', #3
                      'sfx.logattacktime.mean', #4
                      'sfx.logattacktime.stdev', #5
                      'sfx.inharmonicity.mean', #6
                      'sfx.inharmonicity.stdev', #7
                      'lowLevel.spectral_contrast.mean', #8
                      'lowLevel.spectral_contrast.stdev', #9
                      'lowLevel.mfcc.mean', #10
                      'lowLevel.barkbands_kurtosis.mean', #11
                      'lowLevel.barkbands_skewness.mean', #12
                      'lowLevel.barkbands_spread.mean', #13
                      'lowLevel.spectral_complexity.mean', #14
                      'lowLevel.spectral_crest.mean', #15
                      'lowLevel.spectral_energyband_high.mean', #16
                      'lowLevel.spectral_energyband_low.mean', #17
                      'lowLevel.spectral_energy.mean', #18
                      'lowLevel.spectral_energy.stdev', #19
                      'lowLevel.spectral_kurtosis.mean', #20
                      'lowLevel.spectral_flatness_db.mean', #21
                      'lowLevel.spectral_flux.mean', #22
                      'lowLevel.spectral_flux.stdev', #23
                      'lowLevel.spectral_spread.stdev', #24
                      'lowLevel.spectral_spread.mean', #25
                      'lowLevel.sccoeffs.mean', #26
                      'lowLevel.spectral_rolloff.mean', #27
                      'lowLevel.zerocrossingrate.mean', #28
                      'lowLevel.zerocrossingrate.stdev', #29
                      'lowLevel.strongpeak.mean' ,#30
                      'lowLevel.mfcc.mean', # 31
                      'lowLevel.hfc.mean' ,# 32
                      'lowLevel.silence_rate_30dB.mean',  #33
                      'lowLevel.silence_rate_20dB.mean' , #34
                      'lowLevel.silence_rate_60dB.mean',  #35
                      'lowLevel.spectral_rms.mean', # 36
                      'sfx.oddtoevenharmonicenergyratio.mean', # 37
                      'lowLevel.pitch_salience.mean', #38
                      'lowLevel.strongpeak.mean' #39

                    ]

descriptorMapping = np.array(descriptorMapping)

def fetchFeatures(inputPath):
    dataDetails = {}
    for path, dname, fnames in walk(inputPath):
        for fname in fnames:
            if 'features' in fname:
                filename = path+'/'+fname
                remain, rname, cname, sname = path.split('/')[:-3], path.split('/')[-3], path.split('/')[-2], path.split('/')[-1]
                if not dataDetails.has_key(cname):
                    dataDetails[cname]={}
                fDict = json.load(open(filename))
                dataDetails[cname][sname]={'file':fname, 'feature': fDict}
    return dataDetails


def convFtrDict2List(ftrDict, descriptorsToSelect):
    """
    Select the desired descriptors, given by the variable descriptors
    """
    ftr = []
    # print ftrDict.keys()

    try:
        for feature in ftrDict['lowLevel'].keys():
            feature_ = 'lowLevel.'+str(feature)+'.mean'

            if feature_ in descriptorMapping[descriptorsToSelect]:
                if hasattr(ftrDict['lowLevel'][feature]['mean'], '__len__'):
                    for val in ftrDict['lowLevel'][feature]['mean']:
                        ftr.append(float(val))
                else:
                    ftr.append(float(ftrDict['lowLevel'][feature]['mean']))

        for feature in ftrDict['sfx'].keys():
            feature_ = 'sfx.' + str(feature) + '.mean'

            if feature_ in descriptorMapping[descriptorsToSelect]:
                if hasattr(ftrDict['sfx'][feature]['mean'], '__len__') > 1:
                    for val in ftrDict['sfx'][feature]['mean']:
                        ftr.append(float(val))
                else:
                    ftr.append(float(ftrDict['sfx'][feature]['mean']))
    except:
        for feature in ftrDict['features']['lowLevel'].keys():
            feature_ = 'lowLevel.'+str(feature)+'.mean'

            if feature_ in descriptorMapping[descriptorsToSelect]:
                if hasattr(ftrDict['features']['lowLevel'][feature]['mean'], '__len__'):
                    for val in ftrDict['features']['lowLevel'][feature]['mean']:
                        ftr.append(float(val))
                else:
                    ftr.append(float(ftrDict['features']['lowLevel'][feature]['mean']))

        for feature in ftrDict['features']['sfx'].keys():
            feature_ = 'sfx.' + str(feature) + '.mean'

            if feature_ in descriptorMapping[descriptorsToSelect]:
                if hasattr(ftrDict['features']['sfx'][feature]['mean'], '__len__') > 1:
                    for val in ftrDict['features']['sfx'][feature]['mean']:
                        ftr.append(float(val))
                else:
                    ftr.append(float(ftrDict['features']['sfx'][feature]['mean']))


    return np.array(ftr)


#!/usr/bin/python
# -*- coding: utf-8 -*-


def clusterSounds(targetDir, nCluster=-1, descInput=[]):
    """
      This function clusters all the sounds in targetDir using kmeans clustering.
      Input:
        targetDir (string): Directory where sound descriptors are stored (all the sounds in this
                            directory will be used for clustering)
        nCluster (int): Number of clusters to be used for kmeans clustering.
        descInput (list) : List of indices of the descriptors to be used for similarity/distance
                           computation (see descriptorMapping)
      Output:
        Prints the class of each cluster (computed by a majority vote), number of sounds in each
        cluster and information (sound-id, sound-class and classification decision) of the sounds
        in each cluster. Optionally, you can uncomment the return statement to return the same data.
    """

    dataDetails = fetchFeatures(targetDir)

    # print convFtrDict2List(dataDetails['cello_note']['358222'])
    # print dataDetails

    ftrArr = []
    infoArr = []

    if nCluster == -1:
        nCluster = len(dataDetails.keys())
    for cname in dataDetails.keys():

    # iterating over sounds

        for sname in dataDetails[cname].keys():

        # print dataDetails[cname][sname]['feature']

            ftrArr.append(convFtrDict2List(dataDetails[cname][sname]['feature'], descInput))
            infoArr.append([sname, cname])

    # print ('out ')
    # print (ftrArr)

    ftrArr = np.array(ftrArr)
    infoArr = np.array(infoArr)

    ftrArrWhite = whiten(ftrArr)
    (centroids, distortion) = kmeans(ftrArrWhite, nCluster)
    clusResults = -1 * np.ones(ftrArrWhite.shape[0])

    for ii in range(ftrArrWhite.shape[0]):
        diff = centroids - ftrArrWhite[ii, :]
        diff = np.sum(np.power(diff, 2), axis=1)
        indMin = np.argmin(diff)
        clusResults[ii] = indMin

    ClusterOut = []
    classCluster = []
    globalDecisions = []
    for ii in range(nCluster):
        ind = np.where(clusResults == ii)[0]
        freqCnt = []
        for elem in infoArr[ind, 1]:
            freqCnt.append(infoArr[ind, 1].tolist().count(elem))
        indMax = np.argmax(freqCnt)
        classCluster.append(infoArr[ind, 1][indMax])

        #print '\n(Cluster: ' + str(ii) \
        #    + ') Using majority voting as a criterion this cluster belongs to ' \
        #    + 'class: ' + classCluster[-1]
        #print 'Number of sounds in this cluster are: ' + str(len(ind))
        decisions = []
        for jj in ind:
            if infoArr[jj, 1] == classCluster[-1]:
                decisions.append(1)
            else:
                decisions.append(0)
        globalDecisions.extend(decisions)
        #print 'sound-id, sound-class, classification decision'
        ClusterOut.append(np.hstack((infoArr[ind],
                          np.array([decisions]).T)))
        #print ClusterOut[-1]
    globalDecisions = np.array(globalDecisions)
    totalSounds = len(globalDecisions)
    nIncorrectClassified = len(np.where(globalDecisions == 0)[0])
    #print 'Out of %d sounds, %d sounds are incorrectly classified considering that one cluster should ideally contain sounds from only a single class' \
    #    % (totalSounds, nIncorrectClassified)
    #print 'You obtain a classification (based on obtained clusters and majority voting) accuracy of %.2f percentage' \
    #    % round(float(100.0 * float(totalSounds - nIncorrectClassified)
    #            / totalSounds), 2)

    return (totalSounds, nIncorrectClassified)

  # return ClusterOut


			

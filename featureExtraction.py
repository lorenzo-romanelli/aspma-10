import numpy as np
import essentia.standard as ess
import essentia.pool
from os import listdir, makedirs, path
import baselineCluster as BC
import soundAnalysis as SA
import clusterFeatures as CF

def extractDefaultFeatures(audio, outputDir):
    
    # compute all features for all sounds
    extractor = ess.Extractor(dynamics = True,
                                dynamicsFrameSize = 88200,
                                dynamicsHopSize = 44100,
                                highLevel = True,
                                lowLevel = True,
                                lowLevelFrameSize = 2048,
                                lowLevelHopSize = 1024,
                                midLevel = True,
                                namespace = "",
                                relativeIoi = False,
                                rhythm = True,
                                sampleRate = 44100,
                                tonalFrameSize = 4096,
                                tonalHopSize = 2048,
                                tuning = True)
        
    pool = essentia.Pool()
    pool = extractor(audio)
    aggPool = ess.PoolAggregator()(pool)

    if not path.exists(outputDir):
        makedirs(outputDir)

    ess.YamlOutput(filename = outputDir + "features.json", format = "json", doubleCheck = True)(aggPool)


def extractUsefulFeatures(audio, outputDir):
    
    # compute useful features for all sounds (low-level, mid-level, tuning)
    extractor = ess.Extractor(dynamics = False,
                                dynamicsFrameSize = 88200,
                                dynamicsHopSize = 44100,
                                highLevel = False,
                                lowLevel = True,
                                lowLevelFrameSize = 2048,
                                lowLevelHopSize = 1024,
                                midLevel = True,
                                namespace = "",
                                relativeIoi = False,
                                rhythm = False,
                                sampleRate = 44100,
                                tonalFrameSize = 4096,
                                tonalHopSize = 2048,
                                tuning = True)
                                
    pool = essentia.Pool()
    pool = extractor(audio)
    aggPool = ess.PoolAggregator()(pool)
    
    if not path.exists(outputDir):
        makedirs(outputDir)
    
    ess.YamlOutput(filename = outputDir + "features.json", format = "json", doubleCheck = True)(aggPool)


def highEnergyFrames(audioIn, threshold = 0.05, strip = False):
    
    # strip: if True, only low-energy frames at the beginning and at the end are discarded

    RMS = ess.RMS()
    highEnergyAudio = []

    frames = ess.FrameGenerator(audioIn, frameSize = 2048, hopSize = 1024, startFromZero = True)
    rmsValues = np.array([float(RMS(frame)) for frame in frames])
    highRMSFrames = np.where(rmsValues > threshold)[0]
    outSamples = [frame * 1024 for frame in highRMSFrames]
    
    if strip:
        # return the middle section of the audio
        highEnergyAudio = audioIn[outSamples[0] : outSamples[-1]] if len(highRMSFrames > 1) else audioIn
    else:
        # return all high-energy samples
        highEnergyAudio = audioIn[outSamples]
    
    return highEnergyAudio


def getSoundfilesList(instrumentsDir = "/home/oktopus/Documenti/Master/ASPMA/sms-tools/workspace/A10/sounds/"):

    instruments = listdir(instrumentsDir)
    soundfilesList = []
    
    for instrumentDir in instruments:
        soundsDir = listdir(instrumentsDir + instrumentDir + "/")
        
        for soundDir in soundsDir:
            
            if ".txt" in soundDir:
                continue
            
            files = listdir(instrumentsDir + instrumentDir + "/" + soundDir)
            audio = ""
                
            for f in files:
                if ".mp3" in f: 
                    audio = f 

            soundfilesList.append([instrumentsDir + instrumentDir + "/" + soundDir + "/" + audio, 
                                    instrumentsDir + instrumentDir + "/" + soundDir + "/"])

    return soundfilesList


def computeClusteringBaseline():
    
    nDescriptors = len(CF.descriptorMapping)
    iterations = 10
    accuracies = []
    errorsSteps = []
    best = 0.0
    bestDesc = (-1,-1)
    
    for i in range(nDescriptors):
        for j in range(nDescriptors):
            
            if j <= i:      # the descriptors are equal or the pair has already been computed
                continue
            #SA.descriptorPairScatterPlot("sounds", descInput = (i,j), anotOn = 0)
            
            # compute the mean k-means clustering performance over 10 iterations
            accumulator = 0.0
            for itr in range(iterations):
                try:
                    (totalSounds, nIncorrectClassified) = CF.clusterSounds("sounds", nCluster = 10, descInput = [i,j])
                    accuracy = float(100.0 * float(totalSounds - nIncorrectClassified) / totalSounds)
                    accumulator += accuracy
                except:
                    print("Error in clustering using {} and {}".format(CF.descriptorMapping[i], CF.descriptorMapping[j]))
                    errorsSteps.append((i,j))
                    pass
            
            baseline = float(accumulator/10)
            accuracies.append(baseline)
            if baseline > best: 
                best = baseline
                bestDesc = (i,j)
            
            print("Descriptors: {}-{}, {}-{}".format(i, CF.descriptorMapping[i], j, CF.descriptorMapping[j]))
            print("Accuracy: {}%".format(round(baseline,2)))

    print("Best performance (over {} iterations): {}%, obtained with descriptors {}".format(iterations, best, bestDesc))
    
    return (errorsSteps, accuracies)
    

def enhanceAccuracy(goodDescriptors):
    
    nDescriptors = len(CF.descriptorMapping)
    #nDescriptors = 1
    iterations = 10
    best = 0.0
    accuracies = []
    
    for i in range(nDescriptors):
        
        if i in goodDescriptors:
            continue

        testDescriptors = goodDescriptors + [i]
        #testDescriptors = goodDescriptors
        print testDescriptors
        
        accumulator = 0.0
        for itr in range(iterations):
            try:
                (totalSounds, nIncorrectClassified) = CF.clusterSounds("sounds", nCluster = 10, descInput = testDescriptors)
                accuracy = float(100.0 * float(totalSounds - nIncorrectClassified) / totalSounds)
                accumulator += accuracy
            except:
                print("Error at step {}".format(i))
                break
            
        baseline = float(accumulator/iterations)
        accuracies.append(baseline)
        if baseline > best: 
            best = baseline
            bestDesc = testDescriptors
            
    print("Best performance (over {} iterations): {}%, obtained with descriptors {}".format(iterations, best, bestDesc))


def main():
    
    rmsThreshold = 0.05
    computeEnergy = False
    extractFeatures = False
    doClustering = True

    #SA.descriptorMapping = SA.mappingEssentia

    soundfilesList = getSoundfilesList()
    for audiofile, outputDir in soundfilesList:
        
        loadMono = ess.MonoLoader(filename = audiofile)
        audio = loadMono()
        
        if computeEnergy:
            highEnergyAudio = highEnergyFrames(audioIn = audio,
                                                threshold = rmsThreshold,
                                                strip = True)
            try:
                extractUsefulFeatures(highEnergyAudio, outputDir)
                print (audiofile + ": useful features exported (trimmed)")
            except:
                try:
                    extractDefaultFeatures(highEnergyAudio, outputDir)
                    print (audiofile + ": default features exported (trimmed)")
                except:
                    print("Error extracting features from trimmed sound! :(")
        
        if extractFeatures:
            # extractDefaultFeatures(audio, outputDir)        
            extractUsefulFeatures(audio, outputDir)
            print (audiofile + ": useful features exported")                
        
    if doClustering:
        # computeClusteringBaseline()     # best descriptors are 25 and 31
                                          # (lowLevel.spectral_spread.mean and lowLevel.mfcc.mean)
                                        
        enhanceAccuracy([0,1,9,15,18,23,25,26,28,30,31,34,36])    
        
        # NON-TRIMMED SOUNDS
        # 25,31,33                              - 75.5%
        # 15,25,31,33                           - 75.55%
        # 15,25,31,33,35                        - 76.45%
        # 0,15,25,31,33,35                      - 77.65%
        # 0,15,25,26,31,33,35                   - 76.6%
        # 0,15,24,25,26,31,33,35                - 76.45%
        # 0,10,15,24,25,26,31,33,35             - 77.2%
        # 0,10,15,24,25,26,31,33,35,36          - 76.85%
        # 0,10,15,20,24,25,26,31,33,35,36       - 77.35%
        # 0,9,10,15,20,24,25,26,31,33,35,36     - 79.5%
        # 0,2,9,10,15,20,24,25,26,31,33,35,36   - 77.75%
        
        # TRIMMED SOUNDS
        # 25,28,31                              - 73.7%
        # 15,25,28,31                           - 75.4%
        # 15,25,26,28,31                        - 76.3%
        # 15,25,26,28,31,34                     - 77.05%
        # 15,25,26,28,30,31,34                  - 75.8%
        # 9,15,25,26,28,30,31,34                - 76.65%
        # 0,9,15,25,26,28,30,31,34              - 76.75%
        # 0,9,15,18,25,26,28,30,31,34           - 76.6%
        # 0,1,9,15,18,25,26,28,30,31,34         - 75.5%
        # 0,1,9,15,18,25,26,28,30,31,34,36      - 76.25%
        # 0,1,9,15,18,23,25,26,28,30,31,34,36   - 76.7%
        
        '''
        DESCRIPTORS USED (for reference)
        
        'lowLevel.spectral_centroid.mean',# 0
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
        '''
        
if __name__ == "__main__":
    main()


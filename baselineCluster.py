import soundAnalysis as SA

def computeClusteringBaseline():
    
    nDescriptors = len(SA.descriptorMapping)
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
                    (totalSounds, nIncorrectClassified) = SA.clusterSounds("sounds", nCluster = 10, descInput = [i,j])
                    accuracy = float(100.0*float(totalSounds-nIncorrectClassified)/totalSounds)
                    accumulator += accuracy
                except:
                    print("Error clustering {} and {}".format(SA.descriptorMapping[i], SA.descriptorMapping[j]))
                    errorsSteps.append((i,j))
                    pass
            
            baseline = float(accumulator/10)
            accuracies.append(baseline)
            if baseline > best: 
                best = baseline
                bestDesc = (i,j)
            
            print("Descriptors: {}-{}, {}-{}".format(i, SA.descriptorMapping[i], j, SA.descriptorMapping[j]))
            print("Accuracy: {}%".format(round(baseline,2)))

    print("Best performance (over {} iterations): {}%, obtained with descriptors {}".format(iterations, best, bestDesc))
    
    return (errorsSteps, accuracies)
    
def enhanceAccuracy():
    
    nDescriptors = len(SA.descriptorMapping)
    #nDescriptors = 1
    iterations = 10
    best = 0.0
    accuracies = []
    
    for i in range(nDescriptors):
        
        accumulator = 0.0
        for itr in range(iterations):
            try:
                (totalSounds, nIncorrectClassified) = SA.clusterSounds("sounds", nCluster = 10, descInput = [5,12,11,14,9])
                accuracy = float(100.0*float(totalSounds-nIncorrectClassified)/totalSounds)
                accumulator += accuracy
            except:
                print("Error at step {}".format(i))
            
        baseline = float(accumulator/10)
        accuracies.append(baseline)
        if baseline > best: 
            best = baseline
            bestDesc = i
            
    print("Best performance (over {} iterations): {}%, obtained with descriptors (5, 12, 11, 14, 9)".format(iterations, best, bestDesc))
        
if __name__ == "__main__":
    
    SA.descriptorMapping = SA.mappingFreesound
    
    #computeClusteringBaseline()    # best descriptors are 5 (lowlevel.spectral_contrast.mean.0)
                                    # and 12 (lowlevel.mfcc.mean.1)
    
    enhanceAccuracy()               # best descriptors are 5,9,11,12,14
                                    # (accuracy of 74.8%)
    

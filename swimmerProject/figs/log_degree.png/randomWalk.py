import numba
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numba import njit
import scipy
from scipy.spatial.transform import Rotation as Rot

@njit
def getAction(state,ratio):
    #t=1 if state=0 and t = ratio if state = 1   ???
    t1,t2,t3,t4 = np.where(state==0,1,ratio)
    sum = t1*t2*t3 + t1*t2*t4 + t1*t3*t4 + t2*t3*t4
    randomNumber = np.random.uniform(0,1,1)
    if randomNumber <= t4*t2*t3/sum:
        changingJoint = 0
        tc= t1
    elif(randomNumber <= (t4*t3*t1+t4*t2*t3)/sum):  
        changingJoint = 1
        tc=t2
    elif randomNumber <= (t4*t2*t1+t4*t2*t3+t4*t3*t1)/sum:
        changingJoint = 2
        tc=t3
    else:
        changingJoint = 3
        tc=t4

    # deltat = getDeltat2(t1,t2,t3,t4)   ### check detla t
    # print("t= ",getDeltat2(t1,t2,t3,t4) )
    #deltat = np.array([t1,t2,t3,t4])[changingJoint]
    #print(getDeltat(t1,t2,t3,t4,tc))
    # deltat = getDeltat(t1,t2,t3,t4,tc)


    if state[changingJoint] == 0:
        deltat = ratio
    else:
        deltat = 1


    return changingJoint, 1 / deltat


    # return changingJoint,deltat


@njit
def doStep(Data,state,ratio):
    action,deltat = getAction(state,ratio)
    new_state = state.copy()
    new_state[action] = 1 - new_state[action]
    delta = Data[fromBinary(state)*4 + action][0,:]    ### check data
    rotation = Data[fromBinary(state)*4 + action][1:,:]
    return new_state,delta,rotation,deltat

states = np.zeros((10,4),dtype=np.int64)

@njit
def getDeltat(t1,t2,t3,t4,tc):
    tav = 18.844
    return (tav/tc)/(((1/t1 + 1/t2 + 1/t3 + 1/t4)**2))


@njit
def getDeltat2(t1,t2,t3,t4):
    return 1/(1/t1 + 1/t2 + 1/t3 + 1/t4)


@njit
def getBinary(a):
    return np.array([a//8,(a%8)//4,(a%4)//2,a%2])
@njit 
def fromBinary(binary):
    x = 0
    for i in range(len(binary)):
        x += binary[::-1][i]*(2**i)
    return x


allPossbileActions = np.load("allPossbileActions.npy")
actionsData = np.load("actionDatasForE=3.npy")

@njit
def perturbe(Data,perturbingSteps,ratio):
    time = 0
    pertrubingPose = np.array([0.,0.,0.])
    perturbingRotation = np.eye(3)
    state = np.ones(4,dtype=np.int64)
    pertrubingPoses = np.zeros((perturbingSteps,3))
    pertrubingTimes = np.zeros(perturbingSteps)
        
    for t in range(perturbingSteps):
        newState,delta,rotation,deltat = doStep(Data,state,ratio)
        state = newState
        pertrubingPose += perturbingRotation @ delta
        perturbingRotation = rotation @ perturbingRotation
        pertrubingPoses[t] = pertrubingPose
        time += deltat
        pertrubingTimes[t] = time
        
    return pertrubingPoses,pertrubingTimes


# @njit
# def getPerturbingDiffusionCoefForRatio(Data,ratio,iterations=1000,perturbingSteps=1000):
#     # iterations = 10_00
#     # perturbingSteps = 10_00
#     meanFinalPoses = np.zeros((perturbingSteps))
#     meanFinalTimes = np.zeros(perturbingSteps)
#     for iteration in range(iterations):
#         pertrubingPoses,pertrubingTimes = perturbe(Data,perturbingSteps,ratio)
#         #print(np.sum(np.square(pertrubingPoses[:]),axis=1).shape,meanFinalPoses.shape)
#         meanFinalPoses +=   np.sum(np.square(pertrubingPoses[:]),axis=1)
#         meanFinalTimes += pertrubingTimes
#         #print(iteration)
     
#     return meanFinalTimes/iterations,meanFinalPoses/iterations

# def getPerturbingDiffusionCoef(Data,ratios,iterations=1000,perturbingSteps=1000):
#     diffs =np.zeros(ratios.shape)
#     for i in range(ratios.shape[0]):
#         print(i)
#         meanFinalTimes,meanFinalPoses= getPerturbingDiffusionCoefForRatio(Data,ratios[i],iterations,perturbingSteps)
#         slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(meanFinalPoses,meanFinalTimes)
#         diffs[i] =  slope 
#     return diffs



def getPerturbingDiffusionCoefForRatio(Data,ratio,iterations=1000,perturbingSteps=1000):
    iterationDiffs = np.zeros(iterations)
    for iteration in range(iterations):
        pertrubingPoses,pertrubingTimes = perturbe(Data,perturbingSteps,ratio)


        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pertrubingTimes,np.sum(np.square(pertrubingPoses),axis=1))
        iterationDiffs[iteration] =  slope 

     
    return np.mean(iterationDiffs)

def getPerturbingDiffusionCoef(Data,ratios,iterations=1000,perturbingSteps=1000):
    diffs =np.zeros(ratios.shape)
    for i in range(ratios.shape[0]):
        print(i)
        diffs[i] = getPerturbingDiffusionCoefForRatio(Data,ratios[i],iterations,perturbingSteps)
    return diffs


ratios = np.exp(np.linspace(-5,5,50))


diffs = getPerturbingDiffusionCoef(actionsData,ratios,10000,10000)

np.save("diffs1",diffs)
plt.plot(np.log(ratios),diffs)
plt.savefig("plot.png")
plt.show()
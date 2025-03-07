import numba
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numba import njit
import scipy
from scipy.spatial.transform import Rotation as Rot
import datetime

T=1
R=1
a = 10*R
epsilons = np.linspace(0.1,5,50)*R

steps = 1000
nsteps = 1000
dt = T/steps



#initial positions 
r0 = np.array([[0, 0, 0],
[0, 0, a],
[a*np.sqrt(8)/3, 0 , -a/3],
[-a*np.sqrt(8)/6, a*np.sqrt(24)/6 , -a/3],
[-a*np.sqrt(8)/6, -a*np.sqrt(24)/6, -a/3]])

def getR0(ls):
    a1,a2,a3,a4 = ls
    
    R0 = [[0, 0, 0],
    [0, 0, a1],
    [a2*np.sqrt(8)/3, 0 , -a2/3],
    [-a3*np.sqrt(8)/6, a3*np.sqrt(24)/6 , -a3/3],
    [-a4*np.sqrt(8)/6, -a4*np.sqrt(24)/6, -a4/3]]
    
    return np.array(R0)

@njit
def getDistanceMatrix(A):   # dij = rj - ri
    tiledPoses = tile(A)
    distances = tiledPoses - tiledPoses.transpose((1,0,2))
    return distances


@njit
def tile(x):
    tiled = np.zeros((5,5,3))
    for i in range(5):
        tiled[i,:,:] = x
        
    return tiled


@njit
def tile2(x):
    tiled = np.zeros((5,5,3,3))
    for i in range(3):
        for j in range(3):
            tiled[:,:,i,j] = x
        
    return tiled


@njit
def getO(d):
    d2 = d**2    
    drNorm = np.sum(d2,axis=2)**0.5
            
    
    I5_5_3_3 = np.zeros((5,5,3,3))

    for i in range(5):
        for j in range(5):
            I5_5_3_3[i,j,:,:] = np.eye(3)


    c1 = d.reshape(5,5,3,1)
    c2 = c1.transpose(0,1,3,2)    
    B = np.multiply(c1,c2)    
    c3 = tile2(drNorm)

    O = R*(I5_5_3_3 + B/(c3**2))*0.75/c3 #*R/norm(l)

    for i in range(5):  #for i=j
        O[i,i,:,:] = np.eye(3)
    return O


@njit
def getRR(r):
    rr0 = np.array([
    
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]])
    
    
    rr1 = np.array([[0, -r[0,2] , r[0,1],  0, -r[1,2] , r[1,1],    0, -r[2,2] , r[2,1], 0, -r[3,2] , r[3,1]   ,0, -r[4,2] , r[4,1]],
    [r[0,2], 0 , -r[0,0],   r[1,2], 0 , -r[1,0],  r[2,2], 0 , -r[2,0],  r[3,2], 0 , -r[3,0],  r[4,2], 0 , -r[4,0]],
    [-r[0,1], r[0,0], 0 , -r[1,1], r[1,0], 0 , -r[2,1], r[2,0], 0 ,   -r[3,1], r[3,0], 0 ,  -r[4,1], r[4,0], 0 ]])
    
    


    
    return np.concatenate((rr0,rr1),axis=0)




@njit
def getOO(d):
    ##Check
    k2_3 = np.dot(d[1,0],d[1,0])*np.dot(d[2,0],d[2,0])/np.dot(d[1,0],d[2,0])
    k2_4 = np.dot(d[1,0],d[1,0])*np.dot(d[3,0],d[3,0])/np.dot(d[1,0],d[3,0])
    k2_5 = np.dot(d[1,0],d[1,0])*np.dot(d[4,0],d[4,0])/np.dot(d[1,0],d[4,0])
    k3_4 = np.dot(d[2,0],d[2,0])*np.dot(d[3,0],d[3,0])/np.dot(d[2,0],d[3,0])
    k3_5 = np.dot(d[2,0],d[2,0])*np.dot(d[4,0],d[4,0])/np.dot(d[2,0],d[4,0])

    p2_3 = np.dot(d[2,0],d[2,0])*d[0,1] - k2_3*d[0,2] 
    p2_4 = np.dot(d[3,0],d[3,0])*d[0,1] - k2_4*d[0,3] 
    p2_5 = np.dot(d[4,0],d[4,0])*d[0,1] - k2_5*d[0,4] 
    p3_4 = np.dot(d[3,0],d[3,0])*d[0,2] - k3_4*d[0,3] 
    p3_5 = np.dot(d[4,0],d[4,0])*d[0,2] - k3_5*d[0,4] 


    q2_3 = np.dot(d[1,0],d[1,0])*d[0,2] - k2_3*d[0,1] 
    q2_4 = np.dot(d[1,0],d[1,0])*d[0,3] - k2_4*d[0,1] 
    q2_5 = np.dot(d[1,0],d[1,0])*d[0,4] - k2_5*d[0,1] 
    q3_4 = np.dot(d[2,0],d[2,0])*d[0,3] - k3_4*d[0,2] 
    q3_5 = np.dot(d[2,0],d[2,0])*d[0,4] - k3_5*d[0,2] 
    
    
    OO = np.array([
        [d[1,0,0], d[1,0,1], d[1,0,2], -d[1,0,0], -d[1,0,1], -d[1,0,2], 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [d[2,0,0], d[2,0,1], d[2,0,2], 0, 0, 0, -d[2,0,0], -d[2,0,1], -d[2,0,2], 0, 0, 0, 0, 0, 0,   ],   
    [d[3,0,0], d[3,0,1], d[3,0,2], 0, 0, 0, 0, 0, 0, -d[3,0,0], -d[3,0,1], -d[3,0,2], 0, 0, 0],
    [d[4,0,0], d[4,0,1], d[4,0,2], 0, 0, 0, 0, 0, 0, 0, 0, 0, -d[4,0,0], -d[4,0,1], -d[4,0,2]],
    
    [-p2_3[0] -q2_3[0], -p2_3[1] -q2_3[1], -p2_3[2] -q2_3[2],                p2_3[0], p2_3[1], p2_3[2], q2_3[0], q2_3[1], q2_3[2], 0, 0, 0, 0, 0, 0], #2_3,
    [-p2_4[0] -q2_4[0], -p2_4[1] -q2_4[1], -p2_4[2] -q2_4[2],                p2_4[0], p2_4[1], p2_4[2], 0, 0, 0, q2_4[0], q2_4[1], q2_4[2], 0, 0, 0],  #2_4,
    [-p2_5[0] -q2_5[0], -p2_5[1] -q2_5[1], -p2_5[2] -q2_5[2],                p2_5[0], p2_5[1], p2_5[2], 0, 0, 0, 0, 0, 0, q2_5[0], q2_5[1], q2_5[2]], #2_5
    [-p3_4[0] -q3_4[0], -p3_4[1] -q3_4[1], -p3_4[2] -q3_4[2],                0, 0, 0, p3_4[0], p3_4[1], p3_4[2], q3_4[0], q3_4[1], q3_4[2], 0, 0, 0],  #3_4
    [-p3_5[0] -q3_5[0], -p3_5[1] -q3_5[1], -p3_5[2] -q3_5[2],                0, 0, 0, p3_5[0], p3_5[1], p3_5[2], 0, 0, 0, q3_5[0], q3_5[1], q3_5[2]] #3_5
    
    ])
    
    return OO


@njit
def getC(l,u):
    lu = np.zeros(4)
    for i in range(4):
        lu[i]= (l[i]*u[i])
        
    
    return np.array([*list(lu), 0,0,0,0,0,0,0,0, 0,0,0])  

#  u -> u/steps maybe


def getV(r,l,u):
    d = getDistanceMatrix(r)

    O = getO(d)
    newO = np.zeros((15,15))
    for i in range(5):
        for j in range(5):
            newO[3*i:3*i+3,3*j:3*j+3] = O[i,j]
        
    rr= getRR(r)
    mm = np.linalg.inv(newO)


    NN = np.matmul(rr,mm)
    OO = getOO(d)
    AA = np.concatenate((OO,NN),axis=0)
    
    
    

    BB = getC(l,u)
    
    return np.linalg.solve(AA,BB)


def step(r,L):
    for t in range(nsteps):
        try:
            U = L[:,t+1]-L[:,t]
        except IndexError:
            U = L[:,t]-L[:,t-1]
            
        v = getV(r,L[:,t],U).reshape((5,3))
        
        r += v  ##########
        
        
    return r    
        
        
        
def act(L):
    r = getR0(L[:,0])
    r_final = step(r.copy(),L)
    delta = list(r_final - r)[0]

    # dTheta = getTheta(r_final) - getTheta(r)
    # dPhi = getPhi(r_final) - getPhi(r)
    
    # np.matmul(getE(rf6), np.linalg.inv(getE(r0)))
    
    return r_final,delta,np.matmul(getE(r_final), np.linalg.inv(getE(r)))


def getE(r):
    e1 = ((r[1]-r[0])/np.linalg.norm((r[1]-r[0]))).reshape((3,1))
    e2 = ((r[2]-r[0])/np.linalg.norm((r[2]-r[0]))).reshape((3,1))
    e3 = ((r[3]-r[0])/np.linalg.norm((r[3]-r[0]))).reshape((3,1))
    
    return np.concatenate((e1,e2,e3),axis=1)



@njit
def getAction(state,ratio):
    #t=1 if state=0 and t = ratio if state = 1   ???
    t1,t2,t3,t4 = np.where(state==0,1,ratio)
    sum = t1*t2*t3 + t1*t2*t4 + t1*t3*t4 + t2*t3*t4
    #probs = np.array([t2*t3*t4/sum,t1*t3*t4/sum,t1*t2*t4/sum,t2*t3*t1/sum])
    randomNumber = np.random.uniform(0,1,1)
    if randomNumber <= t4*t2*t3/sum:
        changingJoint = 0
    elif(randomNumber <= (t4*t3*t1+t4*t2*t3)/sum):  
        changingJoint = 1
    elif randomNumber <= (t4*t2*t1+t4*t2*t3+t4*t3*t1)/sum:
        changingJoint = 2
    else:
        changingJoint = 3


    # if state[changingJoint] == 0:
    #     deltat = ratio
    # else:
    #     deltat = 1


    # return changingJoint, 1 / deltat    
    return changingJoint,getDeltat2(t1,t2,t3,t4)


@njit
def doStep(Data,state,ratio):
    action,deltat = getAction(state,ratio)
    new_state = state.copy()
    new_state[action] = 1 - new_state[action]
    delta = Data[fromBinary(state)*4 + action][0,:]  #???
    rotation = Data[fromBinary(state)*4 + action][1:,:]
    return new_state,delta,rotation,deltat

@njit
def getDeltat(t1,t2,t3,t4,tc):
    tav = 18.844
    return ((1-1/(tav*t1))*(1-1/(tav*t2))*(1-1/(tav*t3))*(1-1/(tav*t4))*tav/tc)/(((1/t1 + 1/t2 + 1/t3 + 1/t4)**2)*(1-1/(tav*tc)))

@njit
def getDeltat2(t1,t2,t3,t4):
    return 1/(1/t1 + 1/t2 + 1/t3 + 1/t4)
    

@njit
def getBinary(int: a):
    return np.array([int//8,(int%8)//4,(int%4)//2,int%2])
@njit 
def fromBinary(binary):
    x = 0
    for i in range(len(binary)):
        x += binary[::-1][i]*(2**i)
    return x

@njit 
def fromBinary2d(binaries):
    x = np.zeros(binaries.shape[0],dtype=np.int64)
    for i in range(binaries.shape[0]):
        x[i] = fromBinary(binaries[i])
    return x

def getAllPossibleActions(a,epsilon,steps):
    
    closing = a - np.arange(steps+1)*epsilon/steps
    closed = (a-epsilon)*np.ones(steps+1)
    opening = a - epsilon + np.arange(steps+1)*epsilon/steps
    opened  = a *np.ones(steps+1)
    
    possibleActions = np.zeros((64,4,steps+1))
    
    
    
    for i in range(16):
        for j in range(4):
            actionBinaryState = getBinary(i)
            # closed for 0 and opened for 1 
            ca = np.array([closed,closed,closed,closed])*actionBinaryState.reshape(4,1) + np.array([opened,opened,opened,opened])*(1-actionBinaryState).reshape(4,1)
            if actionBinaryState[j] == 0:
                ca[j,:] = opening
            else:
                ca[j,:] = closing
                
            possibleActions[4*i+j] = ca 
        
    return possibleActions



epsilon = 3
allPossbileActions = getAllPossibleActions(a,epsilon, steps)
print("all possible actions calculated")
# np.save("allPossbileActions",allPossbileActions)


actionsData = np.zeros((64,4,3))

for i in range(64):
    #print(i)
    _,delta,Rotation = act(allPossbileActions[i])
    actionsData[i,0,:] = delta
    actionsData[i,1:4,:] = Rotation
    

print("actions data stored")
# np.save("actionDatasForE=3",actionsData)



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


@njit
def getOrientedAction(state,ratio,rotation,grad):
     #TODO *ratio
    #t=1 if state=0 and t = ratio if state = 1
    # print(np.dot(rotation,r0.T).T,cGrad)
    # print(np.dot(np.dot(rotation,r0.T).T,cGrad))

    grad = ratio*np.array([0., 0., grad])   #TODO
    ratios = ratio + np.dot(np.dot(rotation,r0[1:].T).T,grad)

    
    t1,t2,t3,t4 = np.where(state==0,np.ones(4),ratios)
    #print(t1,t2,t3,t4)
    
    sum = t1*t2*t3 + t1*t2*t4 + t1*t3*t4 + t2*t3*t4
    #probs = np.array([t2*t3*t4/sum,t1*t3*t4/sum,t1*t2*t4/sum,t2*t3*t1/sum])
    randomNumber = np.random.uniform(0,1,1)
    if randomNumber <= t4*t2*t3/sum:
        changingJoint = 0
     #   deltat = getDeltat(t1,t2,t3,t4,t1)
    elif(randomNumber <= (t4*t3*t1+t4*t2*t3)/sum):  
        changingJoint = 1
#        deltat = getDeltat(t1,t2,t3,t4,t2)
    elif randomNumber <= (t4*t3*t1+t4*t2*t3+t4*t2*t1)/sum:
        changingJoint = 2
        # deltat = getDeltat(t1,t2,t3,t4,t3)
    else:
        changingJoint = 3
        # deltat = getDeltat(t1,t2,t3,t4,t4)

    #changingJoint = np.random.choice(np.arange(4),p=probs)
    # action = np.concatenate((state,changingJoint.reshape((1,))),axis=0)
    # deltat = 1 if state[changingJoint]==0 else ratio
    return changingJoint,getDeltat2(t1,t2,t3,t4)

@njit
def doOrientedStep(Data,state,ratio,rotation,grad):
    action,deltat = getOrientedAction(state,ratio,rotation,grad)
    new_state = state.copy()
    new_state[action] = 1 - new_state[action]
    delta = Data[fromBinary(state)*4 + action][0,:]
    rotation = Data[fromBinary(state)*4 + action][1:,:]
    return new_state,delta,rotation,deltat



@njit
def Orientedperturbe(Data,perturbingSteps,ratio,grad):
    time = 0
    pertrubingPose = np.array([0.,0.,0.])
    perturbingRotation = np.eye(3)
    state = np.ones(4,dtype=np.int64)
    pertrubingPoses = np.zeros((perturbingSteps,3))
    pertrubingTimes = np.zeros(perturbingSteps)

    for t in range(perturbingSteps):
        newState,delta,rotation,deltat = doOrientedStep(Data,state,ratio,perturbingRotation,grad)
        state = newState
        pertrubingPose += perturbingRotation @ delta
        perturbingRotation = rotation @ perturbingRotation
        pertrubingPoses[t] = pertrubingPose
        time += deltat
        pertrubingTimes[t] = time

    return pertrubingPoses,pertrubingTimes



@njit
def getOrientedPerturbingDiffusionCoefForRatio(Data,ratio,grad,iterations = 10_00,perturbingSteps = 10_00):
    # iterations = 10_00
    # perturbingSteps = 10_00
    meanFinalPoses = np.zeros((perturbingSteps))
    meanFinalTimes = np.zeros(perturbingSteps)
    for iteration in range(iterations):
        pertrubingPoses,pertrubingTimes = Orientedperturbe(Data,perturbingSteps,ratio,grad)
        #print(np.sum(np.square(pertrubingPoses[:]),axis=1).shape,meanFinalPoses.shape)
        meanFinalPoses += pertrubingPoses[:,2]
        meanFinalTimes += pertrubingTimes
        #print(iteration)
     
    # plt.plot(meanFinalTimes,meanFinalPoses)
    # plt.show()
    
    # return slope  #*some Coef
    return meanFinalTimes/iterations,-meanFinalPoses/iterations

def getOrientedPerturbingDiffusionCoef(Data,ratio,grads,iterations = 10_00,perturbingSteps = 10_00):
    diffs =np.zeros(grads.shape)
    for i in range(grads.shape[0]):
        print(i)
        meanFinalTimes,meanFinalPoses= getOrientedPerturbingDiffusionCoefForRatio(Data,ratio,grads[i],iterations,perturbingSteps)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(meanFinalTimes,meanFinalPoses)
        diffs[i] =  slope 
    return diffs


def getOrientedPerturbingDiffusionCoef2(Data,ratios,grad,iterations = 10_00,perturbingSteps = 10_00):
    diffs =np.zeros(grads.shape)
    for i in range(ratios.shape[0]):
        print(i)
        meanFinalTimes,meanFinalPoses= getOrientedPerturbingDiffusionCoefForRatio(Data,ratios[i],grad*ratios[i])
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(meanFinalTimes,meanFinalPoses)
        diffs[i] =  slope 
    return diffs


while(True):
    grads = 10**(np.arange(-250,-150)/100)
    ratio = 1
    vs = getOrientedPerturbingDiffusionCoef(actionsData,ratio,grads,10,10000)
    name = str(datetime.datetime.now())
    np.save(name ,vs)
    print(name)

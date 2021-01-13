import numpy as np
import matplotlib.pyplot as plt

dimension = 2
numberOfScaffoldNucleotides = 5
numberOfStapleNucleotides = 3

# rows are particles, than first entry A = 1 G = 2 C = 3 T = 4
# second x, third = y

class Bases:
    A = 1
    G = 2
    C = -2
    T = -1

baseSpacing = 1
rCutoff = 0.4#baseSpacing /4
α = 1.0
T = 100
angleSigma = 0.1
boxSize = 2 * (numberOfScaffoldNucleotides + numberOfStapleNucleotides) * baseSpacing
scaffold = np.zeros((numberOfScaffoldNucleotides, dimension + 1))
staple = np.zeros((numberOfStapleNucleotides, dimension + 1))
# first is scafold base number and second is staple base number
# [[1,2]] = > first scafold sticks to second nucleotide  
stickTogether =[]
removeBondProb = 0.1

scaffoldNucleotide = [Bases.A, Bases.G, Bases.A, Bases.A, Bases.C]
stapleNucleotide = [Bases.C, Bases.A, Bases.G]

for i in range(numberOfScaffoldNucleotides):
    scaffold[i][0] = scaffoldNucleotide[i]
    scaffold[i][1] = -(boxSize / 4) + 4
    scaffold[i][2] = -(boxSize / 4) + baseSpacing * i + 4

for i in range(numberOfStapleNucleotides):
    staple[i][0] = stapleNucleotide[i]
    staple[i][1] = -(boxSize / 4) + baseSpacing * i + 4 +0.25
    staple[i][2] = +(boxSize / 4) - 3

def mutate(scaffold, staple, stickTogether):
    # sticking together stuff
    removeBond = np.random.rand() > 1 - removeBondProb
    if removeBond and len(stickTogether) > 0:
        del stickTogether[np.random.randint(low = 0, high = len(stickTogether))]

    # rotate
    base = np.random.randint(0, len(scaffold) + len(staple))
    up = np.random.rand() > 0.5
    ΔAngle = np.random.randn() * angleSigma

    if base < len(scaffold):
        
        rotatePointsAround(scaffold, base, ΔAngle, up)
        # find nearest sticking point

        alsoRotateStaple = False
        for point in stickTogether:
            if (up and point[0] > base) or (not up and point[0] < base):
                alsoRotateStaple = True
        
        if alsoRotateStaple:
            for point in staple:
                point[1:] = rotateAround(scaffold[base, 1:], point[1:], ΔAngle)
    
    # staple rotation
    else:
        pass 
        '''
        base = base - len(scaffold)
        if(stickTogether.size == 0):
            rotatePointsAround(staple, base, -ΔAngle, up)
        else: 
            pass
        '''

    #stick fixed together agian:
    #for point in stickTogether:
    #    staple[:,1:] = staple

    #add bond
    for a in range(len(scaffold)):
        for b in range(len(staple)):
            r = np.linalg.norm(scaffold[a, 1:] - staple[b,1:])
            if [a,b] not in stickTogether and scaffold[a,0] + staple[b,0] == 0 and r < rCutoff:
                stickTogether.append([a,b])

                #shift bond directly together if close.
                # TODO fix that it bends if there are more bonds.
                staple[:, 1:] -= staple[b,1:]-scaffold[a,1:]

    #shift back to center
    staple[:, 1:] -= scaffold[0, 1:]
    scaffold[:, 1:] -= scaffold[0, 1:]
    


    

def flip(a, b, c):
    return a + b - c

#not needed in new veri
def rotatePointsAround(points, basePoint, ΔAngle, up):
    origin = points[basePoint][1:]
    if up:
        toRotate = range(basePoint + 1, len(points))
    else:
        toRotate = range(0, basePoint)

    for i in toRotate:
        points[i][1:] = rotateAround(origin, points[i][1:], ΔAngle)


def rotateAround(origin, point, ΔAngle):
    offset = point - origin
    newOffset = np.zeros_like(offset)
    newOffset[0] =  np.cos(ΔAngle) * offset[0] - np.sin(ΔAngle) * offset[1]
    newOffset[1] =  np.sin(ΔAngle) * offset[0] + np.cos(ΔAngle) * offset[1]
    return origin + newOffset

def H(scaffold, staple):
    return -len(stickTogether)

oldH = H(scaffold, staple)
Hs = []
for i in range(10000):
    newScaffold = scaffold.copy()
    newStaple = staple.copy()
    newStickTogether = stickTogether.copy()
    for _ in range(1):
        #mutateWithStickTogether(newScaffold, newStaple, stickTogether)
        mutate(newScaffold, newStaple, newStickTogether)
    ΔH = H(newScaffold, newStaple) - oldH
    print(oldH, ΔH)
    r = np.random.rand()

    if r < np.minimum(1., np.exp(-ΔH / T)):
        scaffold = newScaffold.copy()
        staple = newStaple.copy()
        stickTogether = newStickTogether.copy()
        oldH += ΔH

    Hs.append(oldH)

    if (i % 30 == 0):
        plt.clf()
        plt.plot(scaffold[:, 1], scaffold[:, 2], marker="o", color="red")
        plt.plot(staple[:, 1], staple[:, 2], marker="o", color="green")
        plt.xlim(-boxSize/2, boxSize/2)
        plt.ylim(-boxSize/2, boxSize/2)
        if len(stickTogether) > 0:
            plt.title("stick")
        else:
            plt.title("")
        plt.pause(0.0001)

plt.figure()
plt.plot(Hs)
plt.show()

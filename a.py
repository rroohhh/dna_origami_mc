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
rCutoff = baseSpacing
α = 1.0
T = 1 / 100
angleSigma = 0.1
boxSize = 2 * (numberOfScaffoldNucleotides + numberOfStapleNucleotides) * baseSpacing
scaffold = np.zeros((numberOfScaffoldNucleotides, dimension + 1))
staple = np.zeros((numberOfStapleNucleotides, dimension + 1))

scaffoldNucleotide = [Bases.G, Bases.A, Bases.A, Bases.A, Bases.C]
stapleNucleotide = [Bases.C, Bases.A, Bases.G]

for i in range(numberOfScaffoldNucleotides):
    scaffold[i][0] = scaffoldNucleotide[i]
    scaffold[i][1] = -(boxSize / 4)
    scaffold[i][2] = -(boxSize / 4) + baseSpacing * i

for i in range(numberOfStapleNucleotides):
    staple[i][0] = stapleNucleotide[i]
    staple[i][1] = -(boxSize / 4) + baseSpacing * i + 2.5
    staple[i][2] = +(boxSize / 4) - 2.5

def mutate(scaffold, staple):
    # rotate
    base = np.random.randint(0, len(scaffold) + len(staple))
    if np.random.rand() > 0.5:
        up = np.random.rand() > 0.5
        ΔAngle = np.random.randn() * angleSigma

        if base < len(scaffold):
            rotatePointsAround(scaffold, base, ΔAngle, up)
        else:
            base = base - len(scaffold)
            rotatePointsAround(staple, base, -ΔAngle, up)

        scaffold[:, 1:] -= scaffold[0, 1:]
        staple[:, 1:] -= scaffold[0, 1:]
    else: # flip
        if base > 0 and base < (len(scaffold) - 1):
            a = scaffold[base - 1, 1:]
            b = scaffold[base + 1, 1:]
            c = scaffold[base, 1:]
            scaffold[base, 1:] = flip(a, b, c)
        elif base > len(scaffold) and (base < (len(scaffold) + len(staple) - 1)):
            base -= len(scaffold)
            a = staple[base - 1, 1:]
            b = staple[base + 1, 1:]
            c = staple[base, 1:]
            staple[base, 1:] = flip(a, b, c)

def flip(a, b, c):
    return a + b - c

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
    H = 0
    for a in scaffold:
        for b in staple:
            if (a[0] + b[0] == 0) and (r := np.linalg.norm(a[1:] - b[1:])) < rCutoff:
                H += α * (r**4 - rCutoff**4)
        # for b in scaffold:
        #     if (a[0] + b[0] == 0) and (r := np.linalg.norm(a[1:] - b[1:])) < rCutoff:
    return H

oldH = H(scaffold, staple)
Hs = []
for i in range(10000):
    newScaffold = scaffold.copy()
    newStaple = staple.copy()
    for _ in range(1):
        mutate(newScaffold, newStaple)
    ΔH = H(newScaffold, newStaple) - oldH
    print(oldH, ΔH)
    r = np.random.rand()

    if r < np.minimum(1., np.exp(-ΔH / T)):
        scaffold = newScaffold.copy()
        staple = newStaple.copy()
        oldH += ΔH

    Hs.append(oldH)

    if (i % 30 == 0):
        plt.clf()
        plt.plot(scaffold[:, 1], scaffold[:, 2], marker="o", color="red")
        plt.plot(staple[:, 1], staple[:, 2], marker="o", color="green")
        plt.xlim(-boxSize/2, boxSize/2)
        plt.ylim(-boxSize/2, boxSize/2)
        plt.pause(0.0001)

plt.figure()
plt.plot(Hs)
plt.show()
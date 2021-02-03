import numba
import sys
import copy
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    pass
from typing import *
import logging

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.DEBUG)

# np.random.seed(2)

Base = int

class Bases:
    A: Base = 1
    G: Base = 2
    C: Base = -2
    T: Base = -1

@numba.jitclass([("id", numba.int32), ("stickToId", numba.optional(numba.int32)), ("stickToPointIdx", numba.optional(numba.int64[:])), ("base", numba.int32), ("coords", numba.float64[:])])
class Point:
    id: int
    stickToId: int
#    stickToPoint: Point
    base: Base
    coords: numba.float32[:]

    def __repr__(self):
        return f"(Point id={self.id}, stickToId={self.stickToId}, base={self.base}, coords={self.coords})"

    def __init__(self, id, coords, base):
        self.id = id
        self.stickToId = None
        self.stickToPointIdx = None
        self.base = base
        self.coords = coords

    # def copy(self):
    #     p = Point(id = self.id, coords = self.coords.copy(), base = self.base)
    #     p.stickToId = self.stickToId
    #     p.stickToPointIdx = self.stickToPointIdx.copy()
    #     return p

@numba.jitclass([("points", numba.types.ListType(Point.class_type.instance_type))])
class Molecule:
    # first id from ourselves, second id from other molecule
    points: List[Point]

    def __init__(self, points):
        self.points = points

    def __repr__(self):
        return f"(Molecule {self.points})"

    def ids(self) -> [int]:
        return [point.id for point in self.points]

    # def reversed(self):
    #     return Molecule(self.points[::-1])

    def copy(self):
        newPoints = numba.typed.List()
        for point in self.points:
            p = Point(point.id, point.coords.copy(), point.base)
            p.stickToId = point.stickToId

            if point.stickToPointIdx is not None:
                p.stickToPointIdx = point.stickToPointIdx.copy()

            newPoints.append(p)
        return Molecule(newPoints)

    def plot(self, molecules):
        x = [point.coords[0] for point in self.points]
        y = [point.coords[1] for point in self.points]
        plt.plot(x, y, marker="o")

        label = "sticks to"
        for point in self.points:
            if point.stickToId is not None:
                other = molecules[point.stickToPointIdx[0]].points[point.stickToPointIdx[1]]
                label = str(point.id) + "<->" + str(other.id)
                plt.plot([point.coords[0]], [point.coords[1]], marker = "x", color = "black", label = label)

@numba.extending.overload_method(numba.types.misc.ClassInstanceType, 'reversed')
def ol_molecule_reversed(inst,):
    if inst is Molecule.class_type.instance_type:
        def impl(inst,):
            return Molecule(inst.points[::-1])
        return impl

@numba.njit()
def arccos(angle):
    if angle > 1.0:
        angle = 1.0
    if angle < -1.0:
        angle = -1.0

    try:
        a = np.arccos(angle)
        return a
    except:
        print("bad angle", angle)
        return 0.


@numba.njit()
def mutate(moleculesIn): #: List[Molecule]):
    dont = False

    stickPoints = []
    for molecule in moleculesIn:
        for point in molecule.points:
            if point.stickToId is not None:
                stickPoints.append(point)

    oldNumberOfSticks = len(stickPoints)

    # sticking together stuff
    removeBond = np.random.rand() > 1 - removeBondProb
    if removeBond:
        if len(stickPoints) > 0:
            toDelete = stickPoints[np.random.randint(0, len(stickPoints))]
            other = moleculesIn[toDelete.stickToPointIdx[0]].points[toDelete.stickToPointIdx[1]]
            other.stickToId = None
            other.stickToPointIdx = None
            toDelete.stickToId = None
            toDelete.stickToPointIdx = None

    # rotate
    moleculeIdx = np.random.randint(0, len(moleculesIn))
    baseIdx = np.random.randint(0, len(moleculesIn[moleculeIdx].points) - 1)

    molecules = moleculesIn

    down = np.random.rand() > 0.5
    if down:
        molecules = numba.typed.List()
        for molecule in moleculesIn:
            molecules.append(molecule.reversed())

    ΔAngle = np.random.randn() * angleSigma

    mainMolecule = molecules[moleculeIdx]
    mainRotationOrigin = mainMolecule.points[baseIdx]
    otherMolecules = [molecule for i, molecule in enumerate(molecules) if i != moleculeIdx]

    # Rotade main molecule
    rotatePointsAround(mainRotationOrigin, mainMolecule.points[baseIdx + 1:], ΔAngle)

    # Rotated the other molecules that stick to the main one
    for otherMolecule in otherMolecules:
        lowerStickMainPoint, lowerStickIdx = findStickingIdx(mainMolecule.points[0:baseIdx][::-1], otherMolecule.ids())
        upperStickMainPoint, upperStickIdx = findStickingIdx(mainMolecule.points[baseIdx:], otherMolecule.ids())

        if upperStickIdx is None:
            continue
        else:
            rotatePointsAround(mainRotationOrigin, otherMolecule.points, ΔAngle)

            if lowerStickIdx is not None:
                low = min(lowerStickIdx, upperStickIdx) + 1
                high = max(lowerStickIdx, upperStickIdx)

                if low < high:
                    # calc distance between both sticking points
                    c = lowerStickMainPoint.coords - upperStickMainPoint.coords
                    newDistance = np.linalg.norm(c)

                    # select random point in between
                    adaptionPointIdx = np.random.randint(min(lowerStickIdx, upperStickIdx) + 1, max(lowerStickIdx, upperStickIdx))
                    adaptionPoint = otherMolecule.points[adaptionPointIdx]
                    a = adaptionPoint.coords - upperStickMainPoint.coords
                    b = adaptionPoint.coords - moleculesIn[lowerStickMainPoint.stickToPointIdx[0]].points[lowerStickMainPoint.stickToPointIdx[1]].coords
                    maximumDistance = np.linalg.norm(a) + np.linalg.norm(b)
                    if maximumDistance < newDistance:
                        newDistance = maximumDistance
                        dont = True

                    # figure out the angle to stretch the otherMolecule between the two sticking points to the new distance
                    newcosβ = (np.linalg.norm(a)**2 + np.linalg.norm(b)**2 - newDistance**2) / (2 * np.linalg.norm(a) * np.linalg.norm(b))
                    newβ = arccos(newcosβ)
                    oldβ = arccos(cosAngleBetween(a, b))
                    adaptionΔAngle = newβ - oldβ

                    clockwiseOldβ = clockwiseAngleBetween(a, b)
                    adaptionΔAngle *= np.sign(clockwiseOldβ)

                    rotatePointsAround(adaptionPoint, otherMolecule.points[0:adaptionPointIdx], adaptionΔAngle)

                # correct the orientation of the otherMolecule to fit the mainMolecule
                lowerStickIdx = int(lowerStickIdx)
                otherOrientation = otherMolecule.points[lowerStickIdx].coords - upperStickMainPoint.coords
                mainOrientation = lowerStickMainPoint.coords - upperStickMainPoint.coords
                orientationΔAngle = arccos(cosAngleBetween(otherOrientation, mainOrientation))
                clockwiseOrientationΔAngle = clockwiseAngleBetween(otherOrientation, mainOrientation)

                upperStickOtherPoint = moleculesIn[upperStickMainPoint.stickToPointIdx[0]].points[upperStickMainPoint.stickToPointIdx[1]]
                rotatePointsAround(upperStickOtherPoint, otherMolecule.points[0:upperStickIdx], orientationΔAngle * np.sign(clockwiseOrientationΔAngle))


    newNumberOfSticks = 0
    # find the old bonds that got broken due to being moved too far away
    for molecule in molecules:
        for point in molecule.points:
            if point.stickToId is not None:
                other = moleculesIn[point.stickToPointIdx[0]].points[point.stickToPointIdx[1]]
                if np.linalg.norm(point.coords - other.coords) > rCutoff:
                    other.stickToId = None
                    other.stickToPointIdx = None
                    point.stickToId = None
                    point.stickToPointIdx = None
                else:
                    newNumberOfSticks += 1

    realNewSticks = 0
    # search for new bonds if the points get close
    for i, moleculeA in enumerate(moleculesIn):
        for j, moleculeB in enumerate(moleculesIn):
            if i == j:
                continue
            for k, pointA in enumerate(moleculeA.points):
                for l, pointB in enumerate(moleculeB.points):
                    if (pointA.base + pointB.base == 0) and (np.linalg.norm(pointA.coords - pointB.coords) < rCutoff):
                        if pointA.stickToId is None and pointB.stickToId is None:
                            newNumberOfSticks += 2
                            realNewSticks += 1
                            pointA.stickToId = pointB.id
                            pointA.stickToPointIdx = np.array([j, l])
                            pointB.stickToId = pointA.id
                            pointB.stickToPointIdx = np.array([i, k])

    # undo reversing the points to handle rotation of the lower part instea of the upper part
    if down:
        newMolecules = numba.typed.List()
        for molecule in molecules:
            newMolecules.append(molecule.reversed())
        molecules = newMolecules

    # shift back everything to have the first point of the first molecule at (0, 0)
    origin = molecules[0].points[0].coords
    for molecule in molecules:
        for point in molecule.points:
            point.coords = point.coords - origin

    return dont or ((oldNumberOfSticks > newNumberOfSticks) and not removeBond), realNewSticks

@numba.njit()
def clockwiseAngleBetween(a, b) -> float:
    return np.arctan2(a[0] * b[1] - a[1] * b[0], np.dot(a, b))

@numba.njit()
def cosAngleBetween(a, b) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@numba.njit()
def findStickingIdx(mainPoints, otherIds):
    for point in mainPoints:
        try:
            return point, otherIds.index(point.stickToId)
        except:
            pass
    return None, None

@numba.njit()
def rotatePointsAround(origin, points, ΔAngle):
    for point in points:
        point.coords = rotateAround(origin.coords, point.coords, ΔAngle)

@numba.njit()
def rotateAround(origin, point, ΔAngle):
    offset = point - origin
    newOffset = np.zeros_like(offset, dtype=np.float64)
    newOffset[0] =  np.cos(ΔAngle) * offset[0] - np.sin(ΔAngle) * offset[1]
    newOffset[1] =  np.sin(ΔAngle) * offset[0] + np.cos(ΔAngle) * offset[1]
    return origin + newOffset

def H(molecules):
    H = 0
    for moleculeA in molecules:
        for point in moleculeA.points:
            if point.stickToId is not None:
                other = molecules[point.stickToPointIdx[0]].points[point.stickToPointIdx[1]]
                r = np.linalg.norm(point.coords - other.coords)
                H += β * (r**bindingPower - rCutoff**bindingPower)

        for moleculeB in molecules:
            if moleculeA == moleculeB:
                continue
            for pointA in moleculeA.points:
                for pointB in moleculeB.points:
                    if (pointA.base + pointB.base == 0) and pointA.stickToId is None and pointB.stickToId is None:
                        r = np.linalg.norm(pointA.coords - pointB.coords)
                        H += α * r

    return H


dimension = 2

baseSpacing = 1
rCutoff = 0.2 #baseSpacing /4
β = 10
bindingPower = 4
rCutoffLocalForce = rCutoff * 4
α = 0.001
T = 1 / 100
if len(sys.argv) > 1:
    T = float(sys.argv[1])
angleSigma = 1.0

removeBondProb = 0.1

# figure eight
# scaffoldNucleotide = [Bases.T, *([Bases.C]*5), Bases.A, *([Bases.C]*5), Bases.T]
# stapleNucleotide = [Bases.A, *([Bases.C]*5), Bases.T, *([Bases.C]*5), Bases.A]

scaffoldNucleotide = [*([Bases.T] * 10)]
stapleNucleotide = [*([Bases.A] * 5)]

numberOfScaffoldNucleotides = len(scaffoldNucleotide)
numberOfStapleNucleotides = len(stapleNucleotide)
boxSize = 2 * (numberOfScaffoldNucleotides + numberOfStapleNucleotides) * baseSpacing
scaffold = np.zeros((numberOfScaffoldNucleotides, dimension + 1), dtype=np.float64)
staple = np.zeros((numberOfStapleNucleotides, dimension + 1), dtype=np.float64)

for i in range(numberOfScaffoldNucleotides):
    scaffold[i][0] = scaffoldNucleotide[i]
    scaffold[i][1] = -(boxSize / 4) + 4
    scaffold[i][2] = -(boxSize / 4) + baseSpacing * i + 4

for i in range(numberOfStapleNucleotides):
    staple[i][0] = stapleNucleotide[i]
    staple[i][1] = -(boxSize / 4) + baseSpacing * i + 10 + 0.25
    staple[i][2] = +(boxSize / 4) - 3

def generateMolecules(*packedPositionBases):
    id = 0
    molecules = numba.typed.List()
    for packedPositionBase in packedPositionBases:
        points = numba.typed.List()
        for posBase in packedPositionBase:
            newPoint = Point(id = id, base = posBase[0], coords = posBase[1:])
            points.append(newPoint)
            id += 1
        molecules.append(Molecule(points))

    return molecules


def plot(self, molecules):
    x = [point.coords[0] for point in self.points]
    y = [point.coords[1] for point in self.points]
    plt.plot(x, y, marker="o")

    label = "sticks to"
    for point in self.points:
        if point.stickToId is not None:
            other = molecules[point.stickToPointIdx[0]].points[point.stickToPointIdx[1]]
            label = str(point.id) + "<->" + str(other.id)
            plt.plot([point.coords[0]], [point.coords[1]], marker = "x", color = "black", label = label)

if __name__ == '__main__':
    molecules = generateMolecules(scaffold, staple)

    oldH = H(molecules)
    Hs = []
    sticks = []
    j = 0
    for i in range(10000):
        newMolecules = numba.typed.List([molecule.copy() for molecule in molecules])
        dont, realNewSticks = mutate(newMolecules)
        ΔH = H(newMolecules) - oldH
        # print(oldH, ΔH)
        r = np.random.rand()

        newSticks = 0
        j += 1
        if r < np.minimum(1., np.exp(-ΔH / T)) and not dont:
            # print("accepted")
            molecules = newMolecules
            oldH += ΔH
            newSticks = realNewSticks
            # for molecule in molecules:
            #     print("new lengths:", [np.linalg.norm(a.coords - b.coords) for a, b in zip(molecule.points[1:], molecule.points[:-1])])

        Hs.append(oldH)
        sticks.append(newSticks)

        # if (j % 30 == 0):
        #     plt.clf()
        #     for molecule in molecules:
        #         plot(molecule, molecules)
        #     plt.xlim(-boxSize/2, boxSize/2)
        #     plt.ylim(-boxSize/2, boxSize/2)
        #     plt.pause(0.0001)

    # plt.figure()
    # plt.plot(Hs)
    # plt.figure()
    # plt.plot(sticks)
    # plt.show()
print(np.sum(sticks))

    # bindungsrate

# vary binding power

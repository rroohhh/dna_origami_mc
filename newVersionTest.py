import numba
import numpy as np
import matplotlib.pyplot as plt
from typing import *

Base = int

class Bases:
    A: Base = 1
    G: Base = 2
    C: Base = -2
    T: Base = -1

class Point:
    id: int
    stickToId: int
#    stickToPoint: Point
    base: Base
    coords: numba.float32[:]

    def __repr__(self):
        return f"(Point id={self.id}, stickToId={self.stickToId}, base={self.base}, coords={self.coords})"

    def __init__(self, *, id, coords, base):
        self.id = id
        self.stickToId = None
        self.stickToPoint = None
        self.base = base
        self.coords = coords

    def copyWithCoords(self, coords):
        p = Point(id = self.id, coords = coords, base = self.base)
        p.stickToId = self.stickToId
        p.stickToPoint = self.stickToPoint

        return p

class Molecule:
    # first id from ourselves, second id from other molecule
    points: List[Point]

    def __init__(self, points):
        self.points = points

    def __repr__(self):
        return f"(Molecule {self.points})"

    def ids(self) -> [int]:
        return [point.id for point in self.points]

    def reversed(self):
        return Molecule(self.points[::-1])

    def plot(self):
        x = [point.coords[0] for point in self.points]
        y = [point.coords[1] for point in self.points]
        plt.plot(x, y, marker="o")

        label = "sticks to"
        for point in self.points:
            if point.stickToId is not None:
                label = str(point.id) + "<->" + str(point.stickToPoint.id)
                plt.plot([point.coords[0]], [point.coords[1]], marker = "x", color = "black", label = label)
        

def mutate(moleculesIn: List[Molecule], *, removeBond = None, moleculeIdx = None, baseIdx = None, ΔAngle = None, down = None):
    # sticking together stuff
    if removeBond is None:
        removeBond = np.random.rand() > 1 - removeBondProb
        if removeBond:
            stickPoints = []
            for molecule in moleculesIn:
                for point in molecule.points:
                    if point.stickToId is not None:
                        stickPoints.append(point)

            if len(stickPoints) > 0:
                toDelete = stickPoints[np.random.randint(0, len(stickPoints))] 
                other = toDelete.stickToPoint
                other.stickToId = None
                other.stickToPoint = None
                toDelete.stickToId = None
                toDelete.stickToPoint = None

    # rotate
    if moleculeIdx is None:
        moleculeIdx = np.random.randint(0, len(moleculesIn))
    if baseIdx is None:
        baseIdx = np.random.randint(0, len(moleculesIn[moleculeIdx].points))
    
    molecules = moleculesIn

    if down is None:
        down = np.random.rand() > 0.5
    if down: 
        molecules = [molecule.reversed() for molecule in molecules]

    if ΔAngle is None:
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

        print(f"lowerStickIdx: {lowerStickIdx}, lowerStickMainPoint: {lowerStickMainPoint}")
        print(f"upperStickIdx: {upperStickIdx}, upperStickMainPoint: {upperStickMainPoint}")

        if upperStickIdx is None:
            continue
        else:
            rotatePointsAround(mainRotationOrigin, otherMolecule.points, ΔAngle)

            if lowerStickIdx is not None:
                
                # TODO check what happens if there is no point in between the sticking points
                # calc distance between both sticking points
                c = lowerStickMainPoint.coords - upperStickMainPoint.coords
                newDistance = np.linalg.norm(c)

                # select random point in between 
                adaptionPointIdx = np.random.randint(min(lowerStickIdx, upperStickIdx) + 1, max(lowerStickIdx, upperStickIdx))
                adaptionPoint = otherMolecule.points[adaptionPointIdx]
                a = adaptionPoint.coords - upperStickMainPoint.coords
                b = adaptionPoint.coords - lowerStickMainPoint.stickToPoint.coords
                maximumDistance = np.linalg.norm(a) + np.linalg.norm(b)
                if maximumDistance < newDistance:
                    newDistance = maximumDistance
                
                # figure out the angle to stretch the otherMolecule between the two sticking points to the new distance

                newcosβ = (np.linalg.norm(a)**2 + np.linalg.norm(b)**2 - newDistance**2) / (2 * np.linalg.norm(a) * np.linalg.norm(b))
                newβ = np.arccos(newcosβ)
                oldβ = np.arccos(cosAngleBetween(a, b))
                adaptionΔAngle = newβ - oldβ

                rotatePointsAround(adaptionPoint, otherMolecule.points[lowerStickIdx:adaptionPointIdx], adaptionΔAngle)

                # correct the orientation of the otherMolecule to fit the mainMolecule
                otherOrientation = otherMolecule.points[lowerStickIdx].coords - upperStickMainPoint.coords
                mainOrientation = lowerStickMainPoint.coords - upperStickMainPoint.coords
                orientationΔAngle = np.arccos(cosAngleBetween(otherOrientation, mainOrientation))
                rotatePointsAround(upperStickMainPoint, otherMolecule.points[0:upperStickIdx], orientationΔAngle)

                # TODO(robin): figure out if we need to adapt the orientation of the part of the otherMolecule below the lower sticking point

    # find the old bonds that got broken due to being moved too far away
    for molecule in molecules:
        for point in molecule.points:
            if point.stickToId is not None:
                if np.linalg.norm(point.coords - point.stickToPoint.coords) > rCutoff:
                    point.stickToPoint.stickToId = None
                    point.stickToPoint.stickToPoint = None
                    point.stickToId = None
                    point.stickToPoint = None

    # search for new bonds if the points get close
    for moleculeA in molecules:
        for moleculeB in molecules:
            if moleculeA == moleculeB:
                continue
            for pointA in moleculeA.points:
                for pointB in moleculeB.points:
                    if (pointA.base + pointB.base == 0) and (np.linalg.norm(pointA.coords - pointB.coords) < rCutoff):
                        if pointA.stickToId is None:
                            pointA.stickToId = pointB.id
                            pointA.stickToPoint = pointB
                            pointB.stickToId = pointA.id
                            pointB.stickToPoint = pointA

    # undo reversing the points to handle rotation of the lower part instea of the upper part
    if down:
        molecules = [molecule.reversed() for molecule in molecules]

    # shift back everything to have the first point of the first molecule at (0, 0) 
    origin = molecules[0].points[0].coords
    for molecule in molecules:
        for point in molecule.points:
            point.coords = point.coords - origin

    for i, molecule in enumerate(molecules):
        moleculesIn[i] = molecule
                
def cosAngleBetween(a, b) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def findStickingIdx(mainPoints, otherIds):
    for point in mainPoints:
        try:
            return point, otherIds.index(point.stickToId)
        except:
            pass
    return None, None

def rotatePointsAround(origin, points, ΔAngle):
    for point in points:
        point.coords = rotateAround(origin.coords, point.coords, ΔAngle)

def rotateAround(origin, point, ΔAngle):
    offset = point - origin
    newOffset = np.zeros_like(offset)
    newOffset[0] =  np.cos(ΔAngle) * offset[0] - np.sin(ΔAngle) * offset[1]
    newOffset[1] =  np.sin(ΔAngle) * offset[0] + np.cos(ΔAngle) * offset[1]
    return origin + newOffset

def H(molecules):
    H = 0
    for moleculeA in molecules:
        for point in moleculeA.points:
            if point.stickToId is not None:
                other = point.stickToPoint
                r = np.linalg.norm(point.coords - other.coords)
                H += β * (r**bindingPower - rCutoff**bindingPower )

        for moleculeB in molecules:
            if moleculeA == moleculeB:
                continue
            for pointA in moleculeA.points:
                for pointB in moleculeB.points: 
                    if (pointA.base + pointB.base == 0):
                        r = np.linalg.norm(pointA.coords[1:] - pointB.coords[1:]) 
                        H += α * r

    return H



dimension = 2
numberOfScaffoldNucleotides = 5
numberOfStapleNucleotides = 3

baseSpacing = 1
rCutoff = 0.2 #baseSpacing /4
β = 1
bindingPower = 4
rCutoffLocalForce = rCutoff * 4
α = 0.1
T = 1 / 100
angleSigma = 0.1
boxSize = 2 * (numberOfScaffoldNucleotides + numberOfStapleNucleotides) * baseSpacing
scaffold = np.zeros((numberOfScaffoldNucleotides, dimension + 1))
staple = np.zeros((numberOfStapleNucleotides, dimension + 1))

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


def generateMolecules(*packedPositionBases):
    id = 0
    molecules = []
    for packedPositionBase in packedPositionBases:
        points = []
        for posBase in packedPositionBase:
            points.append(Point(id = id, base = posBase[0], coords = posBase[1:]))
            id += 1
        molecules.append(Molecule(points))

    return molecules

if __name__ == '__main__':
    molecules = generateMolecules(scaffold, staple)

    oldH = H(molecules)
    Hs = []
    for i in range(10000):
        newMolecules = molecules.copy()
        mutate(molecules)
        ΔH = H(newMolecules) - oldH
        print(oldH, ΔH)
        r = np.random.rand()

        if r < np.minimum(1., np.exp(-ΔH / T)):
            molecules = newMolecules
            oldH += ΔH

        Hs.append(oldH)

        if (i % 30 == 0):
            plt.clf()
            for molecule in molecules:
                molecule.plot()
            plt.xlim(-boxSize/2, boxSize/2)
            plt.ylim(-boxSize/2, boxSize/2)
            plt.pause(0.0001)

    plt.figure()
    plt.plot(Hs)
    plt.show()
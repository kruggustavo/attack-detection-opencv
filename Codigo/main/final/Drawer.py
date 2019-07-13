import cv2
import numpy as np
from MathUtilities import MathUtilities


class Drawer:
    POSE_PAIRS_MPI = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13]]

    colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
              [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
              [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]

    # Referencias
    # c = cabeza, n = cuello, h = hombro (i, d), c = codo (i, d), m = mano (i, d), p = pecho
    # hp = cadera (i, d), r = rodilla(i, d), p = pie (i, d)
    POINTS_LABELS = ["c", "n", "hi", "ci", "mi", "hd", "cd", "md", "hpi", "ri", "pi", "hpd", "rd", "pd", "p"]

    nPoints = 14

    keypoints_list = np.zeros((0, 3))
    
    defaultAngleArms = 0
    defaultAngleLegs = 0

    def drawSkeletonPoints(self, frame, points, margin=0):
        # Draw Skeleton
        coloridx = 0

        for pair in self.POSE_PAIRS_MPI:
            partA = self.POINTS_LABELS[pair[0]]
            partB = self.POINTS_LABELS[pair[1]]

            if partA in points:
                x, y = points[partA]
                x += margin
                y += margin
                cv2.circle(frame, (x, y), 3, self.colors[coloridx], -1)
                cv2.putText(frame, str(x) + " " + str(y), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, self.colors[coloridx], 1, lineType=cv2.LINE_AA)
                #cv2.putText(frame, partA, (x + 5,y + 8), cv2.FONT_HERSHEY_SIMPLEX,  0.2, (150, 150, 150), 1, lineType=cv2.LINE_AA)

            if partB in points:
                x, y = points[partB]
                x += margin
                y += margin
                cv2.circle(frame, (x, y), 3, self.colors[coloridx], -1)
                cv2.putText(frame, str(x) + " " + str(y), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, self.colors[coloridx], 1, lineType=cv2.LINE_AA)
                #cv2.putText(frame, partB, (x + 5,y + 8), cv2.FONT_HERSHEY_SIMPLEX,  0.2, (150, 150, 150), 1, lineType=cv2.LINE_AA)


            try:
                cv2.line(frame, points[partA], points[partB], (200, 55, 25), 1, lineType=cv2.LINE_AA)
            except:
                #print("Can not draw line between " + partA + " and " + partB)
                pass

            coloridx += 1

        return frame

    def drawSkeletonLines(self, frame, lines, margin=0):
        # Draw Skeleton
        for label in lines:
            lineWidth = 1
            lineColor = (200, 55, 25)

            pointA = lines[label][0]
            pointB = lines[label][1]

            x1, y1 = pointA
            x1 += margin
            y1 += margin

            x2, y2 = pointB
            x2 += margin
            y2 += margin

            # Dibujamos tronco de otro color y anchura
            if label == "trunkPoints":
                lineWidth = 3
                lineColor = (44, 185, 5)

            cv2.line(frame, (x1, y1), (x2, y2), lineColor, lineWidth, lineType=cv2.LINE_AA)
            cv2.circle(frame, (x1, y1), 3, (0, 0, 255), -1)
            cv2.circle(frame, (x2, y2), 3, (0, 0, 255), -1)

            cv2.putText(frame, str(x1) + " " + str(y1), (x1 + 5, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (150, 150, 150), 1,
                        lineType=cv2.LINE_AA)
            cv2.putText(frame, str(x2) + " " + str(y2), (x2 + 5, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (150, 150, 150), 1,
                        lineType=cv2.LINE_AA)

        return frame

    def getLabeledPoints(self, points):
        newPoints = {}
        for idx in points:
            if points[idx] != None:
                newPoints[self.POINTS_LABELS[idx]] = points[idx]
        return newPoints

    # Devuelve puntos de linea si existen, de lo contrario None
    def getLinePointsIfExists(self, point1, point2):
        line = []
        try:
            line.append(point1)
            line.append(point2)
        except:
            line = []
        return line

    def putPointsInLines(self, lineLabel, pointsArray, point1Label, point2Label, lines):
        # Línea formada por una Parte del cuerpo = {(x1,y1), (x2,y2)}:
        if point1Label in pointsArray and point2Label in pointsArray:
            if pointsArray[point1Label] != None and pointsArray[point2Label] != None:
                lines[lineLabel] = self.getLinePointsIfExists(pointsArray[point1Label], pointsArray[point2Label])

        #if lineLabel not in lines:
        #    print("No complete points for line label " + lineLabel)

    # Devuelve array de angulos entre lineas relevantes
    def getBodyAngles(self, points, targetFrame=None):
        lines = {}

        # Obtenemos lineas de acuerdo a los puntos
        # lines[indice parte del cuerpo] = {{puntos de la recta
        self.putPointsInLines("leftForearmPoints", points, "ci", "mi", lines)     # Antebrazo izq
        self.putPointsInLines("rightForearmPoints", points, "cd", "md", lines)    # Antebrazo der
        self.putPointsInLines("leftArmPoints", points, "hi", "ci", lines)         # Brazo izq
        self.putPointsInLines("rightArmPoints", points, "hd", "cd", lines)        # Brazo der
        self.putPointsInLines("leftThighPoints", points, "hpi", "ri", lines)      # Muslo izq
        self.putPointsInLines("rightThighPoints", points, "hpd", "rd", lines)     # Muslo der
        self.putPointsInLines("leftLegPoints", points, "ri", "pi", lines)         # Pierna izq (pantorrilla)
        self.putPointsInLines("rightLegPoints", points, "rd", "pd", lines)        # Pierna izq (pantorrilla)

        # Definimos el tronco de acuerdo a puntos existentes
        # El tronco es la linea perpendicular a caderas, u hombros
        upperTrunkPoint = None              # Punto superior del tronco, se define por ciertos puntos en caso de que existan
        lowerTrunkPoint = None              # Punto inferior

        # Tronco a partir de puntos superiores
        if "hd" in points and "hi" in points:
            x1, y1 = points["hi"]
            x2, y2 = points["hd"]
            # Punto medio entre hombros
            newX = int((x1 + x2) / 2)
            newY = int((y1 + y2) / 2)
            upperTrunkPoint = (newX, newY)

            try:
                # Obtenemos parte inferior de tronco a partir de linea perpendicular a hombros y un punto por debajo de ellos
                distance = MathUtilities.distance(points["hi"], points["hd"])
                x, y = MathUtilities.getPerpendicularLinePoints(points["hi"], points["hd"], upperTrunkPoint, newY + distance)
                lowerTrunkPoint = (x, y)
            except:
                pass
        # Existe un solo hombro
        else:
            upperTrunkPoint = (points["hi"] if "hi" in points else (points["hd"] if "hd" in points else None))

        # Tronco a partir de puntos inferiores
        if "hpi" in points and "hpd" in points:
            x1, y1 = points["hpi"]
            x2, y2 = points["hpd"]
            # Punto medio entre caderas
            newX = int((x1 + x2) / 2)
            newY = int((y1 + y2) / 2)
            lowerTrunkPoint = (newX, newY)

            if upperTrunkPoint == None:
                try:
                    # Obtenemos parte superior de tronco a partir de linea perpendicular a caderas y un punto por encima de ellos
                    distance = MathUtilities.distance(points["hpi"], points["hpd"])
                    x, y = MathUtilities.getPerpendicularLinePoints(points["hpi"], points["hpd"], lowerTrunkPoint, newY - (distance * 2))
                    upperTrunkPoint = (x, y)
                except:
                    pass
        # Existe una sola cadera
        else:
            lowerTrunkPoint = lowerTrunkPoint if lowerTrunkPoint != None else (points["hpi"] if "hpi" in points else (points["hpd"] if "hpd" in points else None))

        # Si tenemos puntos para tronco, lo definimos
        if lowerTrunkPoint != None and upperTrunkPoint != None:
            trunkline = []
            trunkline.append(upperTrunkPoint)
            trunkline.append(lowerTrunkPoint)
            lines["trunkPoints"] = trunkline

        angles = {}
        # Obtenemos angulos entre lineas SOLAMENTE SI EXISTE TRONCO
        if "trunkPoints" in lines:
            # Si no se especifica angulo, predeterminado es 180
            angles["leftArm"] = self.getAngle(lines["leftArmPoints"], lines["trunkPoints"], targetFrame, 0) if "leftArmPoints" in lines else self.defaultAngleArms
            angles["rightArm"] = self.getAngle(lines["rightArmPoints"], lines["trunkPoints"], targetFrame, 0) if "rightArmPoints" in lines else self.defaultAngleArms
            angles["leftForearm"] = self.getAngle(lines["leftForearmPoints"], lines["trunkPoints"], targetFrame, 0) if "leftForearmPoints" in lines else self.defaultAngleArms
            angles["rightForearm"] = self.getAngle(lines["rightForearmPoints"], lines["trunkPoints"], targetFrame, 0) if "rightForearmPoints" in lines else self.defaultAngleArms
            angles["leftThigh"] = self.getAngle(lines["leftThighPoints"], lines["trunkPoints"], targetFrame, 0) if "leftThighPoints" in lines else self.defaultAngleLegs
            angles["rightThigh"] = self.getAngle(lines["rightThighPoints"], lines["trunkPoints"], targetFrame, 0) if "rightThighPoints" in lines else self.defaultAngleLegs
            angles["leftLeg"] = self.getAngle(lines["leftLegPoints"], lines["trunkPoints"], targetFrame, 0) if "leftLegPoints" in lines else self.defaultAngleLegs
            angles["rightLeg"] = self.getAngle(lines["rightLegPoints"], lines["trunkPoints"], targetFrame, 0) if "rightLegPoints" in lines else self.defaultAngleLegs

        return angles, lines

    # Obtiene angulos entre lineas y las dibuja
    # angleFix: conversion de angulo para ciertas lineas
    def getAngle(self, line1, line2, targetFrame, angleFix):
        angle = MathUtilities.getAngleBetweenLines(line1, line2)
        angle = angle if angleFix == 0 else angleFix - angle
        
        if targetFrame != None:
            cv2.putText(targetFrame, "Angle : " + str(angle), (10, targetFrame.shape[0]- (9 * self.yStartTextPoint)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (55, 55, 55), 1, cv2.LINE_AA)
            try:
                for point in (line1 + line2):
                    cv2.circle(targetFrame, point, 4, (0, 65, 0), -1)
    
                cv2.putText(targetFrame, str(angle), (line1[0][0], line1[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)
    
                cv2.line(targetFrame, line1[0], line1[1], (200, 150, 25), 1, lineType=cv2.LINE_4)
                cv2.line(targetFrame, line2[0], line2[1], (200, 150, 25), 1, lineType=cv2.LINE_4)
            except:
                pass

        return angle


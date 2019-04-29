import cv2
import numpy as np
from utils.MathUtilities import MathUtilities

#
# Dibuja puntos geometricos y permite operaciones con ellos
# Extremidades superiores referencia: https://t1.uc.ltmcdn.com/images/2/9/3/img_24392_apa_253395_600.jpg

class Drawer:
    POINTS_NUMBER_FOR_ARMS = 8          # Puntos que conforman los brazos, hombros y cabeza
    POINTS_NUMBER_FOR_LEGS = 5          # Puntos de piernas

    yStartTextPoint = 0

    # Referencias
    # c = cabeza, h = hombro (i, d), c = codo (i, d), m = mano (i, d), p = pecho
    # h = cadera, r = rodilla(i, d), p = pie (i, d)
    POSE_PAIRS = [["c", "p"], ["hd", "hi"], ["hi", "ci"], ["ci", "mi"], ["hd", "cd"], ["cd", "md"], ["p", "h"], ["h", "ri"], ["ri", "pi"], ["h", "rd"], ["rd", "pd"]]

    _targetFrame = np.zeros((320, 260, 3), np.uint8)
    _height, _width = _targetFrame.shape[:2]
    _origins = {POINTS_NUMBER_FOR_ARMS: 0, POINTS_NUMBER_FOR_LEGS: int(_height / 2) - 10}  # Origen en el plano para dibujar puntos


    def getEmptyFrame(self):
        return self._targetFrame

    # Dibuja puntos y lineas en un frame vacio
    def drawHumanPose(self, posekeypoints):
        targetFrame = self._targetFrame.copy()
        cv2.putText(targetFrame, "Pose", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        if len(posekeypoints) > 0:
            pointsMap = {}
            #cv2.putText(targetFrame, "Half Parts detected: " + str(len(posekeypoints)), (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
            #print("Half Parts detected: " + str(len(posekeypoints)))

            for bodyparts in posekeypoints:
                yOrigin = self._origins[len(bodyparts)]
                for point in bodyparts:
                    for id in point:                # Identificador de punto: mi, md, c, ri, ...
                        x = int(point[id]["x"])
                        y = int(point[id]["y"])
                        y = int(y / 2) + yOrigin
                        pointsMap[id] = (x, y)      # Coordenadas para el punto

                        cv2.circle(targetFrame, (x, y), 3, (0, 0, 255), -1)
                        cv2.putText(targetFrame, id, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(targetFrame, str(x) + " " + str(y), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)

            for pair in self.POSE_PAIRS:
                try:
                    cv2.line(targetFrame, pointsMap[pair[0]], pointsMap[pair[1]], (0, 150, 25), 1, lineType=cv2.LINE_AA)
                except:
                    pass

            return targetFrame


    # Devuelve array de angulos entre lineas relevantes
    def getBodyAngles(self, posekeypoints):
        targetFrame = self._targetFrame.copy()
        cv2.putText(targetFrame, "Angles", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        pointsMap = {}

        for bodyparts in posekeypoints:
            yOrigin = self._origins[len(bodyparts)]
            for point in bodyparts:
                for id in point:  # Point identificator: mi, md, c, ri, ...
                    x = int(point[id]["x"])
                    y = int(point[id]["y"])
                    y = int(y / 2) + yOrigin
                    pointsMap[id] = (x, y)

        pointsNumber = len(pointsMap)

        # Línea formada por una Parte del cuerpo = {(x1,y1), (x2,y2)}
        lines = {}
        # Obtenemos lineas de acuerdo a los puntos
        # lines[indice parte del cuerpo] = {{puntos de la recta
        if pointsNumber == self.POINTS_NUMBER_FOR_ARMS:
            # Se detectaron puntos correspondientes a brazos, parte alta del cuerpo
            lines["leftForearmPoints"] = [pointsMap["mi"], pointsMap["ci"]]     # Antebrazo izq
            lines["rightForearmPoints"] = [pointsMap["md"], pointsMap["cd"]]    # Antebrazo der
            lines["leftArmPoints"] = [pointsMap["hi"], pointsMap["ci"]]         # Brazo izq
            lines["rightArmPoints"] = [pointsMap["hd"], pointsMap["cd"]]        # Brazo der
            lines["trunkPoints"] = [pointsMap["c"], pointsMap["p"]]             # Tronco (Cabeza a pecho)

        elif pointsNumber == self.POINTS_NUMBER_FOR_LEGS:
            # Se detectaron puntos de partes bajas del cuerpo (sin tronco)
            lines["leftThighPoints"] = [pointsMap["h"], pointsMap["ri"]]        # Muslo izq
            lines["rightThighPoints"] = [pointsMap["h"], pointsMap["rd"]]       # Muslo der
            lines["leftLegPoints"] = [pointsMap["ri"], pointsMap["pi"]]         # Pierna izq (pantorrilla)
            lines["rightLegPoints"] = [pointsMap["rd"], pointsMap["pd"]]        # Pierna izq (pantorrilla)
            # Estimacion de tronco a partir del punto de cadera
            hip = (pointsMap["h"][0], 1)                                        # Punto (X = x .. Y = 1) linea recta vertical al punto
            lines["trunkPoints"] = {hip, pointsMap["h"]}
        else:
            # Se detectaron partes altas y bajas (cuerpo completo)
            lines["leftForearmPoints"] = [pointsMap["mi"], pointsMap["ci"]]     # Antebrazo izq
            lines["rightForearmPoints"] = [pointsMap["md"], pointsMap["cd"]]    # Antebrazo der
            lines["leftArmPoints"] = [pointsMap["hi"], pointsMap["ci"]]         # Brazo izq
            lines["rightArmPoints"] = [pointsMap["hd"], pointsMap["cd"]]        # Brazo der
            lines["leftThighPoints"] = [pointsMap["h"], pointsMap["ri"]]        # Muslo izq
            lines["rightThighPoints"] = [pointsMap["h"], pointsMap["rd"]]       # Muslo der
            lines["leftLegPoints"] = [pointsMap["ri"], pointsMap["pi"]]         # Pierna izq (pantorrilla)
            lines["rightLegPoints"] = [pointsMap["rd"], pointsMap["pd"]]        # Pierna izq (pantorrilla)
            lines["trunkPoints"] = [pointsMap["p"], pointsMap["h"]]             # Tronco (Pecho a cadera)

        self.yStartTextPoint = 0
        # Obtenemos angulos entre lineas
        # Si no se especifica angulo, predeterminado es 180
        angles = {}

        angles["leftArm"] = self.getAngle(lines["leftArmPoints"], lines["trunkPoints"], targetFrame, 180) if "leftArmPoints" in lines else 180
        angles["rightArm"] = self.getAngle(lines["rightArmPoints"], lines["trunkPoints"], targetFrame, 180) if "rightArmPoints" in lines else 180
        angles["leftForearm"] = self.getAngle(lines["leftForearmPoints"], lines["trunkPoints"], targetFrame, 0) if "leftForearmPoints" in lines else 180
        angles["rightForearm"] = self.getAngle(lines["rightForearmPoints"], lines["trunkPoints"], targetFrame, 0) if "rightForearmPoints" in lines else 180
        angles["leftThigh"] = self.getAngle(lines["leftThighPoints"], lines["trunkPoints"], targetFrame, 180) if "leftThighPoints" in lines else 180
        angles["rightThigh"] = self.getAngle(lines["rightThighPoints"], lines["trunkPoints"], targetFrame, 180) if "rightThighPoints" in lines else 180
        angles["leftLeg"] = self.getAngle(lines["leftLegPoints"], lines["trunkPoints"], targetFrame, 180) if "leftLegPoints" in lines else 180
        angles["rightLeg"] = self.getAngle(lines["rightLegPoints"], lines["trunkPoints"], targetFrame, 180) if "rightLegPoints" in lines else 180

        return targetFrame, angles

    # Obtiene angulos entre lineas y las dibuja
    # angleFix: conversion de angulo para ciertas lineas
    def getAngle(self, line1, line2, inFrame, angleFix):
        self.yStartTextPoint = self.yStartTextPoint + 1

        angle = MathUtilities.getAngleBetweenLines(line1, line2)
        angle = angle if angleFix == 0 else angleFix - angle
        cv2.putText(inFrame, "Angle : " + str(angle), (10, self._height - (9 * self.yStartTextPoint)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (55, 55, 55), 1, cv2.LINE_AA)

        try:
            for point in (line1 + line2):
                #x, y = point
                cv2.circle(inFrame, point, 4, (0, 65, 0), -1)

            cv2.putText(inFrame, str(angle), (line1[0][0], line1[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.line(inFrame, line1[0], line1[1], (200, 150, 25), 1, lineType=cv2.LINE_4)
            cv2.line(inFrame, line2[0], line2[1], (200, 150, 25), 1, lineType=cv2.LINE_4)
        except:
            pass

        return angle
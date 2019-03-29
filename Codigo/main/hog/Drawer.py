import cv2
import numpy as np
import time
from utils.MathUtilities import MathUtilities

#
# Dibuja puntos geometricos y permite operaciones con ellos
#

class Drawer:
    POINTS_NUMBER_FOR_ARMS = 8          # Puntos que conforman los brazos, hombros y cabeza
    POINTS_NUMBER_FOR_LEGS = 5          # Puntos de piernas

    # Referencias
    # c = cabeza, h = hombro (i, d), c = codo (i, d), m = mano (i, d), p = pecho
    # h = cadera, r = rodilla(i, d), p = pie (i, d)
    POSE_PAIRS = [["c", "p"], ["hd", "hi"], ["hi", "ci"], ["ci", "mi"], ["hd", "cd"], ["cd", "md"], ["p", "h"], ["h", "ri"], ["ri", "pi"], ["h", "rd"], ["rd", "pd"]]

    _targetFrame = np.zeros((320, 260, 3), np.uint8)
    _height, _width = _targetFrame.shape[:2]
    _origins = {POINTS_NUMBER_FOR_ARMS: 0, POINTS_NUMBER_FOR_LEGS: int(_height / 2) - 10}  # Origen en el plano para dibujar puntos


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
            lines["leftThighPoints"] = [pointsMap["h"], pointsMap["ri"]]        # Muslo izq
            lines["rightThighPoints"] = [pointsMap["h"], pointsMap["rd"]]       # Muslo der
            lines["leftLegPoints"] = [pointsMap["ri"], pointsMap["pi"]]         # Pierna izq (pantorrilla)
            lines["rightLegPoints"] = [pointsMap["rd"], pointsMap["pd"]]        # Pierna izq (pantorrilla)
            lines["trunkPoints"] = [pointsMap["p"], pointsMap["h"]]             # Tronco (Pecho a cadera)

        # Obtenemos angulos entre lineas
        angles = {}
        if "leftForearmPoints" in lines:
            angles["leftForearm"] = MathUtilities.getAngleBetweenLines(lines["leftForearmPoints"], lines["trunkPoints"])
            cv2.putText(targetFrame, "Angle between lines : " + str(angles["leftForearm"]), (10, self._height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

            for point in (lines["leftForearmPoints"] + lines["trunkPoints"]):
                x, y = point
                cv2.circle(targetFrame, (x, y), 4, (0, 105, 0), -1)
                cv2.putText(targetFrame, str(x) + " " + str(y), (x + 10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.line(targetFrame, lines["leftForearmPoints"][0], lines["leftForearmPoints"][1], (200, 150, 25), 1, lineType=cv2.LINE_4)
            cv2.line(targetFrame, lines["trunkPoints"][0], lines["trunkPoints"][1], (200, 150, 25), 1, lineType=cv2.LINE_4)

        return targetFrame
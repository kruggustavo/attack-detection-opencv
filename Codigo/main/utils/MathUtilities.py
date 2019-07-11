import math
#
# Runciones matematicas
#

class MathUtilities:

    # Angulo de una recta
    @staticmethod
    def getAngle(pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        dX = x2 - x1
        dY = y2 - y1
        rads = math.atan2(-dY, dX)  # wrong for finding angle/declination?
        return math.degrees(rads)

    # Angulo entre dos rectas (diferencia de angulos de 2 rectas)
    @staticmethod
    def getAngleBetweenLines(line1, line2):
        # Pendientes de las rectas (o angulo entre recta y eje x)
        try:
            angle1 = MathUtilities.getAngle(line1[0], line1[1])
        except:
            angle1 = 180

        try:
            angle2 = MathUtilities.getAngle(line2[0], line2[1])
        except:
            angle2 = 180

        return round(abs(angle1 - angle2), 1)

    # Pendiente de una recta
    @staticmethod
    def getSlope(pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2

        # Evitar division por cero
        result = 0
        if x2 - x1 == 0:
            result = (y2 - y1) / 1
        else:
            result = (y2 - y1) / (x2 - x1)

        return result

    # Perpendicular de una recta
    @staticmethod
    def getPerpendicularLinePoints(pt1, pt2, sharedPoint, targetY):
        newPoint = None
        x, y = sharedPoint

        # Ecuacion de la recta es: y = mx + b    (m : pendiente)
        m = MathUtilities.getSlope(pt1, pt2) + 0.1
        m = 1 if m == 0 else m                                      # Evitar division por cero

        # Pendiente negativa
        neg_m = (1 / m) * -1

        # Obtenemos b a partir de las coordenadas de sharedPoint y la pendiente
        b = y - (neg_m * x)

        #Obtenemos X en funcion de targetY utilizando la ecuacion de la recta, la ecuacion despejada para X es: x = (y - b) / m
        try:
            neg_m = 1 if neg_m == 0 else neg_m                      # Evitar division por cero
            newX = int((targetY - b) / neg_m)

            newPoint = (newX, targetY)
        except:
            pass

        return newPoint

    # Disancia entre dos puntos
    @staticmethod
    def distance(p0, p1):
        return int(math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2))
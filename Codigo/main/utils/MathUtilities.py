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
        angle1 = MathUtilities.getAngle(line1[0], line1[1])
        angle2 = MathUtilities.getAngle(line2[0], line2[1])

        return round(abs(angle1 - angle2), 1)
import math
#
# Clase con funciones matematicas
#

class MathUtilities:

    # Pendiente de una recta
    @staticmethod
    def getSlope(self, point1, point2):
        m = (point2.y - point1.y) / (point2.x - point1.x)
        return m

    # Angulo de una recta
    @staticmethod
    def getAngle(self, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        dX = x2 - x1
        dY = y2 - y1
        rads = math.atan2(-dY, dX)  # wrong for finding angle/declination?
        return math.degrees(rads)

    # Angulo entre dos rectas (diferencia de angulos de 2 rectas)
    @staticmethod
    def getAngleBetweenLines(self, line1, line2):
        # Line 1
        pt1 = line1[0]
        pt2 = line1[1]
        angleLine1 = self.getAngle()


        return math.degrees(rads)
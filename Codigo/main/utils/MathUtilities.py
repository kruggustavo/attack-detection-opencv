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

    # Angulo entre dos rectas
    @staticmethod
    def getAngle(pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        inner_product = (x1 * x2) + (y1 * y2)
        len1 = math.hypot(x1, y1)
        len2 = math.hypot(x2, y2)
        return math.acos(inner_product / (len1 * len2))
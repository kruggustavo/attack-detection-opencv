import random
# Generador de poses aleatorias utilizando reglas de Angulos

ITERATIONS = 100000             # Cantidad de samples de entrenamiento (cantidad de combinaciones a generar, iteraciones)

# Tipos de poses (ataque con cuchillo, ataque con arma de fuego)
KNIFE_POSE = 0
GUN_POSE = 1

max_min_angles = {}

# Angulos maximos y minimos para una pose de ataque con cuchillos
max_min_angles[KNIFE_POSE] = [
            ["leftForearm", [0, 90]],           # Antebrazos
            ["rightForearm", [0, 90]],
            ["leftArm", [10, 170]],             # Brazos
            ["rightArm", [10, 170]],
            ["leftThigh", [45, 230]],           # Muslos
            ["rightThigh", [45, 230]],
            ["leftLeg", [90, 180]],             # Pantorrillas
            ["rightLeg", [90, 180]]
]

max_min_angles[GUN_POSE] = [
            ["leftForearm", [50, 120]],         # Antebrazos
            ["rightForearm", [50, 120]],
            ["leftArm", [50, 180]],             # Brazos
            ["rightArm", [50, 180]],
            ["leftThigh", [45, 230]],           # Muslos
            ["rightThigh", [45, 230]],
            ["leftLeg", [90, 180]],             # Pantorrillas
            ["rightLeg", [90, 180]]
]

f = open("trainingangles.txt","w")

resultAngles = {}
for x in range(ITERATIONS):
    poseNumber = random.randint(KNIFE_POSE, GUN_POSE)
    print()
    print("Generando pose " + str(x + 1) + " --+ " + str(poseNumber))
    selectedAngles = max_min_angles[poseNumber]
    for angles in selectedAngles:
        label = angles[0]
        min = angles[1][0]
        max = angles[1][1]
        randomAngle = round(random.uniform(min, max),1)
        resultAngles[label] = randomAngle
    print(resultAngles)
    f.write(str(resultAngles) + "\n")

f.close()
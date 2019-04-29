import random
# Generador de poses aleatorias utilizando reglas de Angulos

# Accion a ejecutarr
TRAINING_SAMPLES = 0
EVALUATION_SAMPLES = 1

ACTION = TRAINING_SAMPLES

filename = ("training" if ACTION == TRAINING_SAMPLES else "evaluation")
ITERATIONS = 500000            # Cantidad de samples de entrenamiento/evaluacion (cantidad de combinaciones a generar)

print("Generador aleatorio de angulos de poses basado en reglas estaticas")
print("Modo: " + filename)
print("Iteraciones: " + str(ITERATIONS))

# Tipos de poses (ataque con cuchillo, ataque con arma de fuego, sin ataque)
KNIFE_POSE = 0
GUN_POSE = 1
NO_ATTACK_POSE = 2

# Limite global de angulos posibles
start_angle = 0
end_angle = 360

# Angulos min-max para poses compatibles con agresiones
min_max_angles = {}

# Angulos que deberan ser corregidos de acuerdo a si es piernas o brazos. Ej: antebrazo no puede ser mayor a brazo.
angles_to_fix = {}

# Angulos maximos y minimos para una pose de ataque con cuchillos
min_max_angles[KNIFE_POSE] = [
            ["leftArm", [40, 100]],             # Brazos
            ["rightArm", [40, 100]],
            ["leftForearm", [5, 90]],           # Antebrazos
            ["rightForearm", [5, 90]],
            ["leftThigh", [90, 130]],           # Muslos
            ["rightThigh", [90, 130]],
            ["leftLeg", [90, 180]],             # Pantorrillas
            ["rightLeg", [90, 180]]
]

min_max_angles[GUN_POSE] = [
            ["leftArm", [80, 130]],             # Brazos
            ["rightArm", [80, 130]],
            ["leftForearm", [70, 130]],         # Antebrazos
            ["rightForearm", [70, 130]],
            ["leftThigh", [135, 230]],          # Muslos
            ["rightThigh", [135, 230]],
            ["leftLeg", [90, 180]],             # Pantorrillas
            ["rightLeg", [90, 180]]
]

min_max_angles[NO_ATTACK_POSE] = [
            ["leftArm", [170, 190]],            # Brazos
            ["rightArm", [170, 190]],
            ["leftForearm", [170, 200]],        # Antebrazos
            ["rightForearm", [170, 200]],
            ["leftThigh", [160, 190]],          # Muslos
            ["rightThigh", [160, 190]],
            ["leftLeg", [170, 225]],            # Pantorrillas
            ["rightLeg", [170, 225]]
]

POSE_ATTACKS = {KNIFE_POSE, GUN_POSE}

# Angulo indice(i) sera igual a angulo valor(v) si angulo i > v BRAZOS o angulo i < v PIERNAS
angles_to_fix["leftForearm"] = "leftArm"
angles_to_fix["rightForearm"] = "rightArm"
angles_to_fix["leftLeg"] = "leftThigh"
angles_to_fix["rightLeg"] = "rightThigh"

f = open(filename + "angles.csv","w")

resultAngles = {}
for x in range(ITERATIONS):
    # Pose aleatoria
    poseNumber = random.randint(0, len(min_max_angles) - 1)
    selectedAngles = min_max_angles[poseNumber]

    # Generar: 1 = Todos los puntos del cuerpo, 0 = Solo mitad de puntos del cuerpo (puede ser solo piernas, o solo brazos)
    fullOrHalfBody = random.randint(0, 1)
    # Que parte se generara: 1 = Solo brazos, 0 = Solo piernas
    halfPart = random.randint(0, 1)

    csvline = ""

    # Es pose de ataque, o neutral. (Salida esperada)
    y = 0
    if poseNumber in POSE_ATTACKS:
        y = 1

    print("Generando pose " + str(x + 1) + " ... ")

    for angles in selectedAngles:
        label = angles[0]
        min = angles[1][0]
        max = angles[1][1]
        randomAngle = round(random.uniform(min, max), 1)

        try:
            get_label_angle_from = angles_to_fix[label]
            val = resultAngles[get_label_angle_from]

            # Verificamos si angulo es mayor a miembro adyacente (caso positivo igualar) BRAZOS
            if "arm" in label:
                if randomAngle > val:
                    #print("Angulo corregido : " + label + "  .. " + str(randomAngle) + " --> " + str(val))
                    randomAngle = val
            # Verificamos si angulo es menor a miembro adyacente (caso positivo igualar) PIERNAS
            else:
                if randomAngle < val:
                    #print("Angulo corregido : " + label + "  .. " + str(randomAngle) + " --> " + str(val))
                    randomAngle = val
        except:
            pass

        # Casos aleatorios donde solo se pudo definir pose de ataque de brazos, o solo piernas (solo mitad de cuerpo)
        lowerlabel = label.lower()
        if fullOrHalfBody == 0:
            if halfPart == 0:
                # Se obtienen solo piernas (convertir brazos a 180)
                if "arm" in lowerlabel:
                    randomAngle = 180
            else:
                # Se obtienen solo brazos (convertir piernas a 180)
                if "leg" in lowerlabel or "thigh" in lowerlabel:
                    randomAngle = 180

        csvline = csvline + " " + str(randomAngle)
        resultAngles[label] = randomAngle

    #print(resultAngles)
    csvline = csvline + " " + str(y)

    csvline = csvline.strip().replace(" ", ",")
    #print(csvline)

    f.write(csvline + "\n")

f.close()
print("Fin.")
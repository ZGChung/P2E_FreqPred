from sklearn.datasets import load_boston  # chargement des données
from sklearn.linear_model import LinearRegression  # importe le modèle
from sklearn.model_selection import train_test_split  # importe le modèle de test
import matplotlib.pyplot as plt
import numPy as np

donnees = load_boston()
X = donnees['donnees']
y = donnees['but']


def lwlr(testPoint, X, y, k=0.1):
    """

    Crée et estime pour n'importe quel point dans l'espace de x
    Paramètres
    ----------
    testPoint: float
        point dans l'espace de x
    k: float
        determine le poids des points
    Sorties
    -------
    ps : float
        poids de la regression
    """
    xMat = X
    yMat = np.mat(y).T
    # Créer matrice diagonale
    m = np.shape(xMat)[0]
    poids = np.mat(np.eye(m))
    # Créer valeurs exponentiellement décroissantes des poids
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        poids[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))

    # Trouver estimatives pour les points
    xTwx = xMat.T * (np.weights * xMat)
    if np.linalg.det(xTwx) == 0.0:
        print("matrice non inversible")
        return

    ps = xTwx.I * (xMat.T * (poids * yMat))
    return testPoint * ps


# Créer les y ponderees
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat
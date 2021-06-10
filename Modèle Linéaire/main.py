from sklearn.datasets import load_boston # chargement des données
from sklearn.linear_model import LinearRegression # importe le modèle
from sklearn.model_selection import train_test_split # importz le modèle de test
import matplotlib.pyplot as plt

# charge les données - X et Y définis selon nos archives chargés
donnees = load_boston()
X = donnees['donnees']
y = donnees['but']

# diviser en entrainement et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) # test_size determine la taille de l'echantillon d'entrainement, random_state controle la randomisation

regr = LinearRegression()
regr.fit(X_train, y_train)

r2_train = regr.score(X_train, y_train)
r2_test = regr.score(X_test, y_test)
print('R² dans le set dentreinament : %.2f' % r2_train)
print('R² dans le set de test: %.2f' % r2_test)

#matplotlib inline
ax = plt.scatter(y_test, y_train)
ax.figure.set_size_inches(12, 6)
plt.title('Prevision x Réel', fontdict={'fontsize': 16})
plt.xlabel('Prevision', fontdict={'fontsize': 16})
plt.ylabel('Réel', fontdict={'fontsize': 16})
ax
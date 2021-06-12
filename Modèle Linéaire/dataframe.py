import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
from datetime import datetime
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

## Importation des données, créations des covariables

# Ouverture du .csv et création d'un dataframe
df = pd.read_csv("D:\Bureau léger\ECN\P2E\data\data_daily.csv",sep = ";")

# Création de nouvelles covariables

df['time'] = [datetime.date(datetime.fromtimestamp(x)) for x in df['time']]
df['year'] = [x.year for x in df['time']]
df['month'] = [x.month for x in df['time']] # Au format Janvier = 1 , Décembre = 12
df['day'] = [x.isoweekday() for x in df['time']] # Au format Lundi = 1 , Dimanche = 7
df['isWeekend'] = [x >= 6 for x in df['day']]
df['quarter'] = [math.ceil(x/3) for x in df['month']]

# Je retire les jours fermés ( i.e. où il y a 0 entrée)

data = df[(df['entries_daily'] > 0)]

methode = input("Entrez la méthode (1 = Sans travail , 2 = Logarithme , 3 = méthode OLS , 4 = méthode GLS , 5 = méthode WLS ) : ")

## Modèle linéaire, sans travail des données

if (methode == '1') :
    y = data['entries_daily']
    X = data[['year','DJU heat','isWeekend','mean_cloudiness_daily','quarter','mean_windspeed_daily']]

    regr = LinearRegression()
    regr.fit(X,y)

    r_sq = regr.score(X,y) #Obtention du r2
    print('coefficient de determination:', r_sq)
    print('intercept:', regr.intercept_) # Estimateur de alpha 0 (ordonnée à l'origine)
    print('facteurs:', regr.coef_) # Estimateur des coefficients

    y_pred = regr.predict(X)
    # print('réponse prédite:', y_pred, sep='\n')
# Pour prédire avec d'autres données, il suffit de répéter y_pred = regr.predict(X_new)

    y_plt = y[:,None]
    y_plt = [ c[0] for c in y_plt]

    plt.subplot(121)

    plt.scatter(y_plt, y_pred)
    plt.plot([0,400], [0, 400], color = 'red', linestyle = 'solid') # Tracé de la ligne de tendance à atteindre
    plt.title('Nuage de points', fontdict={'fontsize': 15})
    plt.xlabel('Mesuré', fontdict={'fontsize': 15})
    plt.ylabel('Simulé', fontdict={'fontsize': 16})


    plt.subplot(122)
    plt.title('Tracé ', fontdict={'fontsize': 15})

    plt.plot(y_plt)
    plt.plot(y_pred)
    plt.show()

"""
Si les données sont traitées en l'état, on a :
coefficient de determination: 0.3968537421766686
intercept: 144592.4658697193
facteurs: [ -71.46187733    4.57232148 -102.51524382   64.01755279  -44.51342985  -6.18192241]
Le résultat n'est pas forcément très bon, même si le nuage des points montre un caractère plutôt linéaire
"""

## Modèle linéaire, avec passage au logarithme

if methode == '2' :
    y = np.log(data['entries_daily'])
    X = data[['year','DJU heat','isWeekend','mean_cloudiness_daily','quarter','mean_windspeed_daily']]

    regr = LinearRegression()
    regr.fit(X,y)

    r_sq = regr.score(X,y) #Obtention du r2
    print('coefficient de determination:', r_sq)
    print('intercept:', regr.intercept_) # Estimateur de alpha 0 (ordonnée à l'origine)
    print('facteurs:', regr.coef_) # Estimateur des coefficients

    y_pred = np.exp(regr.predict(X))
    # print('réponse prédite:', y_pred, sep='\n')


    y_plt = y[:,None]
    y_plt = [ np.exp(c[0]) for c in y_plt]

#Tracé des résultats

    plt.subplot(121)

    plt.scatter(y_plt, y_pred)
    plt.plot([0,400], [0, 400], color = 'red', linestyle = 'solid') # Tracé de la ligne de tendance à atteindre
    plt.title('Nuage de points', fontdict={'fontsize': 15})
    plt.xlabel('Mesuré', fontdict={'fontsize': 15})
    plt.ylabel('Simulé', fontdict={'fontsize': 16})


    plt.subplot(122)
    plt.title('Tracé ', fontdict={'fontsize': 15})

    plt.plot(y_plt)
    plt.plot(y_pred)
    plt.show()

"""
Si les données sont traitées en passant au logarithme, on a :
coefficient de determination: 0.3514815798710861
intercept: 818.6611871537954
facteurs: [-0.40272091  0.02048947 -0.58220777  0.27826947 -0.21660009 -0.00525502]
Le résultat est meilleur que sans traitement, ce qui est encourageant
"""

## Modèle linéaire, méthode OLS (autre librairie)

if methode == '3' :
    y = np.log(data['entries_daily'])
    X = data[['year','DJU heat','day','mean_cloudiness_daily','quarter','mean_windspeed_daily']]

    X = pd.DataFrame(X).to_numpy()
    y = pd.DataFrame(y).to_numpy()
    X = sm.add_constant(X)
    model = sm.OLS(y,X)
    results = model.fit()
    print('Paramètres :',results.params)
    print(results.summary())

    y_pred = np.exp(results.predict(X))

    y_plt = y[:,None]
    y_plt = [ np.exp(c[0]) for c in y_plt]

    plt.subplot(121)

    plt.scatter(y_plt, y_pred)
    plt.plot([0,400], [0, 400], color = 'red', linestyle = 'solid') # Tracé de la ligne de tendance à atteindre
    plt.title('Nuage de points', fontdict={'fontsize': 15})
    plt.xlabel('Mesuré', fontdict={'fontsize': 15})
    plt.ylabel('Simulé', fontdict={'fontsize': 16})


    plt.subplot(122)
    plt.title('Tracé ', fontdict={'fontsize': 15})

    plt.plot(y_plt)
    plt.plot(y_pred)
    plt.show()

"""
Si les données sont traitées en passant au logarithme et par la méthode OLS (Moindre carrés simples) , on a :
coefficient de determination: 0.343
Le résultat est encore meilleur !
"""
## Modèle linéaire, méthode GLS

if methode == '4' :
    y = np.log(data['entries_daily'])
    X = data[['year','DJU heat','day','mean_cloudiness_daily','quarter','mean_windspeed_daily']]

    X = pd.DataFrame(X).to_numpy()
    y = pd.DataFrame(y).to_numpy()
    X = sm.add_constant(X)
    model = sm.GLS(y,X)
    results = model.fit()
    print('Paramètres :',results.params)
    print(results.summary())

    y_pred = np.exp(results.predict(X))

    y_plt = y[:,None]
    y_plt = [ np.exp(c[0]) for c in y_plt]

    plt.subplot(121)

    plt.scatter(y_plt, y_pred)
    plt.plot([0,400], [0, 400], color = 'red', linestyle = 'solid') # Tracé de la ligne de tendance à atteindre
    plt.title('Nuage de points', fontdict={'fontsize': 15})
    plt.xlabel('Mesuré', fontdict={'fontsize': 15})
    plt.ylabel('Simulé', fontdict={'fontsize': 16})


    plt.subplot(122)
    plt.title('Tracé ', fontdict={'fontsize': 15})

    plt.plot(y_plt)
    plt.plot(y_pred)
    plt.show()

"""
Si les données sont traitées en passant au logarithme et par la méthode GLS (Moindre carrés généralisés) , on a :
coefficient de determination: 0.343
Le résultat est identique à la méthode précédente
"""

## Modèle linéaire, méthode WLS

if methode == '5' :
    y = np.log(data['entries_daily'])
    X = data[['year','DJU heat','day','mean_cloudiness_daily','quarter','mean_windspeed_daily']]

    X = pd.DataFrame(X).to_numpy()
    y = pd.DataFrame(y).to_numpy()
    X = sm.add_constant(X)
    model = sm.WLS(y,X)
    results = model.fit()
    print('Paramètres :',results.params)
    print(results.summary())

    y_pred = np.exp(results.predict(X))

    y_plt = y[:,None]
    y_plt = [ np.exp(c[0]) for c in y_plt]

    plt.subplot(121)

    plt.scatter(y_plt, y_pred)
    plt.plot([0,400], [0, 400], color = 'red', linestyle = 'solid') # Tracé de la ligne de tendance à atteindre
    plt.title('Nuage de points', fontdict={'fontsize': 15})
    plt.xlabel('Mesuré', fontdict={'fontsize': 15})
    plt.ylabel('Simulé', fontdict={'fontsize': 16})


    plt.subplot(122)
    plt.title('Tracé ', fontdict={'fontsize': 15})

    plt.plot(y_plt)
    plt.plot(y_pred)
    plt.show()

"""
Si les données sont traitées en passant au logarithme et par la méthode wLS (Moindre carrés pondérés) , on a :
coefficient de determination: 0.343
Le résultat est identique aux deux autres méthodes de la libraire statsmodel.
"""
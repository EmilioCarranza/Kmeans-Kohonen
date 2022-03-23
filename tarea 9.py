import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


X = pd.read_excel('C:/Users/Emilio/Desktop/IDI II/Tareas/Tarea 8/data.xlsx')
X = X.to_numpy()

K=3

np.random.seed(2)

rand1 = np.random.randint(1,len(X))
rand2 = np.random.randint(1,len(X))
rand3 = np.random.randint(1,len(X))

#Definicion de centroides iniciales
cen1 = X[rand1]
cen2 = X[rand2]
cen3 = X[rand3]

n=2 #umero de pasos

cota = 10
while cota > 1:
    centroides1=[]
    centroides2=[]
    centroides3=[]
    for i in range(len(X)):
        distance1=np.sqrt(np.sum(X[i]-cen1)**2)
        distance2=np.sqrt(np.sum(X[i]-cen2)**2)
        distance3=np.sqrt(np.sum(X[i]-cen3)**2)
        if min(distance1, distance2, distance3) == distance1: #Comparar distancias y cambiar centroides
            ncen1 = cen1 +(X[i]-cen1)/n
            cen1 = ncen1
            centroides1.append(cen1)
        elif min(distance1, distance2, distance3) == distance2:
            ncen2 = cen2 +(X[i]-cen2)/n
            cen2 = ncen2
            centroides2.append(ncen2)
        else:
            ncen3 = cen3 +(X[i]-cen3)/n
            cen3 = ncen3
            centroides3.append(ncen3)
    n += 1      
    cambio_cen1 = np.linalg.norm(centroides1[-1]-centroides1[0])  
    cambio_cen2 = np.linalg.norm(centroides2[-1]-centroides2[0])
    cambio_cen3 = np.linalg.norm(centroides3[-1]-centroides3[0])
    
    cota = max(cambio_cen1,cambio_cen2,cambio_cen3)
    
#Clasificacion de cada punto segun su distancia con los centroides aprendidos
G1 = [] 
G2 = []   
G3 = []
for i in range(len(X)):
    d1= np.sqrt(np.sum(X[i]-cen1)**2)
    d2= np.sqrt(np.sum(X[i]-cen2)**2)
    d3= np.sqrt(np.sum(X[i]-cen3)**2)
    if min(d1,d2,d3) == d1: 
        G1.append(X[i])
    elif min(d1,d2,d3) == d2: 
        G2.append(X[i])
    else:
        G3.append(X[i])

#Pasar los arreglos a DataFrames para su graficación 
G1 = pd.DataFrame(G1)
G2 = pd.DataFrame(G2)
G3 = pd.DataFrame(G3)

#Grafica de los puntos 
plt.scatter(x=G1[0], y=G1[1], c='green')
plt.scatter(x=G2[0], y=G2[1], c='yellow')
plt.scatter(x=G3[0], y=G3[1], c='purple')
plt.scatter(cen1[0], cen1[1], c='red')
plt.scatter(cen2[0], cen2[1], c='red')
plt.scatter(cen3[0], cen3[1], c='red')
plt.show()
#
def clasificador_kohonen(x,y,z):
    distance1=np.sqrt(np.sum((x,y,z)-cen1)**2)
    distance2=np.sqrt(np.sum((x,y,z)-cen2)**2)
    distance3=np.sqrt(np.sum((x,y,z)-cen3)**2)
    if min(distance1, distance2, distance3) == distance1: #Comparar distancias y cambiar centroides
        print("El par", (x,y,z), "pertenese a G1")
    elif min(distance1, distance2, distance3) == distance2:
        print("El par", (x,y,z), "pertenese a G2")
    else:
        print("El par", (x,y,z), "pertenese a G3")


clasificador_kohonen(80,60,100)
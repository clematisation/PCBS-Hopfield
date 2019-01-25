import math
import matplotlib.pyplot as plt
import numpy as np
import random as rd
    
#Cette fonction permet de créer un motif "damier" de la taille souhaitée
    
def checkerboard(size):
    pattern = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i%2==0:
                if j%2==0:
                    pattern[i][j]=-1
                else:
                    pattern[i][j]=1
            else:
                if j%2==0:
                    pattern[i][j]=1
                else:
                    pattern[i][j]=-1
    return pattern
    
    
#Cette fonction permet de créer un motif en L de la taille souhaitée
    
def lshaped(size):
    pattern = np.zeros((size, size))
    for i in range(size-1):
        for j in range(1):
            pattern[i][j]=-1
        for j in range(1, size):
            pattern[i][j]=1
    for i in range(size-1,size):
        for j in range(size):
            pattern[i][j]=-1   
    return pattern
    
#Cette fonction permet de créer un motif carré de la taille souhaitée
    
def square(size):
    pattern=np.zeros((size, size))
    for i in range(1):
        for j in range(size):
            pattern[i][j]=-1
    for i in range(1, size-1):
        for j in range(1):
            pattern[i][j]=-1
        for j in range(1,size-1):
            pattern[i][j]=1
        for j in range(size-1,size):
            pattern[i][j]=-1
    for i in range(size-1,size):
        for j in range(size):
            pattern[i][j]=-1
    return pattern
    
#Cette fonction permet de créer un motif en croix de la taille souhaitée
    
def cross(size):
    pattern=np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i==j) or (i+j==size-1):
                pattern[i][j]=-1
            else:
                pattern[i][j]=1
    return pattern
    
#Cette fonction permet de créer un motif aléatoire de la taille souhaitée
    
def randompattern(size):
    pattern=np.full((size, size), 1)
    L=[]
    a=0
    number=int((size**2)/2)
    while len(L)<number:
        x=rd.randint(0, size-1)
        y=rd.randint(0, size-1)
        if (x,y) in L:
            a=x
        else:
            L.append((x,y))
            pattern[x][y]=-1
    return pattern    
        
            
#Cette fonction permet de bruiter un motif (d neurones sont bruités)

def noisy(pattern, d):
    L=[]
    a=0 #cette variable sert à ce qu'il ne se passe rien si le pixel sélectionné aléatoirement a déjà été bruité
    number=len(pattern[0])
    while len(L)<d:
        x=rd.randint(0, number-1)
        y=rd.randint(0, number-1)
        if (x,y) in L:
            a=x
        else:
            L.append((x,y))
            pattern[x][y]=-pattern[x][y]
    return pattern
    

#Cette fonction permet de convertir une matrice en liste

def convert(matrix):
    number = len(matrix[0])
    c=0
    M=[0 for i in range(number**2)]
    for m in range(number):
        for n in range(number):
            M[c]=matrix[m][n]
            c=c+1
    return M
    
#Cette fonction permet de convertir une liste en matrice

def convert_lm(list):
    n=int(math.sqrt(len(list)))
    M=np.zeros((n,n))
    c=0
    for i in range(n):
        for j in range(n):
            M[i][j]=list[c]
            c=c+1
    return M
    

#Cette fonction crée une matrice des poids synaptiques à partir d'une liste de motifs de même taille

def SynapticWM(Patterns):
    number=len(Patterns[0])    
    SWM=np.zeros((number,number))
    for p in Patterns:
        for i in range(number):
            for j in range(number):
                SWM[i, j] += p[i] * p[j]
    SWM /= number
    np.fill_diagonal(SWM, 0)
    return SWM
    

#Cette fonction permet de calculer le taux de corrélation de l'état du réseau et des motifs stockés

def overlap(pattern1, pattern2):
    number=len(pattern1)
    m=(1/number)*np.dot(pattern1, pattern2)
    return m

#Cette fonction correspond à la dynamique du réseau. Elle affiche le motif mis à jour ainsi que les taux de corrélation à chaque étape

def dynamics(pattern, Motifs, steps):
    h=0
    M=[]
    S=[]
    L=[((i+1)/5) for i in range(len(Motifs))]
    number=len(pattern[0])
    initstate=convert(pattern)
    Weights=SynapticWM(Motifs)
    fig, (ax0, ax1) = plt.subplots(1, 2)
    matrix = pattern
    ax0.imshow(matrix, cmap=plt.cm.spring)
    for p in Motifs :
        m=overlap(initstate, p)
        M.append(m)
        S.append('Motif'+str(Motifs.index(p)))
    ax1.bar(L, tick_label=S, height=M) 
    M=[]
    S=[]
    for s in range(steps):
        for i in range(len(initstate)):
            h=np.dot(Weights[i], initstate)
            initstate[i]=np.sign(h)
        for p in Motifs :
            m=overlap(initstate, p)
            M.append(m)
            S.append('Motif'+str(Motifs.index(p)))
        pattern=convert_lm(initstate)
        fig, (ax0, ax1) = plt.subplots(1, 2)
        matrix = pattern
        ax0.imshow(matrix, cmap=plt.cm.spring)    
        ax1.bar(L, tick_label=S, height=M)
        M=[]
        S=[]


#On crée quatre motifs de taille 5x5 grâce aux fonctions précédentes

Damier5=checkerboard(5)
MotifL5=lshaped(5)
Carre5=square(5)
Croix5=cross(5)

#On affiche les quatre motifs

fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
matrix1 = Damier5
matrix2 = MotifL5
matrix3 = Carre5
matrix4 = Croix5
ax0.imshow(matrix1, cmap=plt.cm.spring)
ax1.imshow(matrix2, cmap=plt.cm.spring)
ax2.imshow(matrix3, cmap=plt.cm.spring)
ax3.imshow(matrix4, cmap=plt.cm.spring)
fig.suptitle('Motifs')


#On convertit chaque motif sous forme de liste

Liste_Damier5=convert(Damier5)  
Liste_MotifL5=convert(MotifL5)  
Liste_Carre5=convert(Carre5)  
Liste_Croix5=convert(Croix5) 

#On crée une liste de motifs

Liste_Motifs=[Liste_Damier5, Liste_MotifL5, Liste_Carre5, Liste_Croix5]

#On calcule la matrice des poids synaptiques

Weights = SynapticWM(Liste_Motifs)

#On affiche la matrice des poids synaptiques

fig, ax = plt.subplots()
matrix = Weights
ax.imshow(matrix, cmap=plt.cm.viridis)
plt.title("Matrice des poids synaptiques")
plt.xlabel("neurone i")
plt.ylabel("neurone j")
cb = plt.colorbar(ax.imshow(matrix, cmap=plt.cm.viridis))
plt.show()

#On bruite notre motif en L

MotifL_Bruite=noisy(MotifL5, 3)

#On affiche notre motif bruité

fig, ax = plt.subplots()
matrix = MotifL_Bruite
ax.imshow(matrix, cmap=plt.cm.viridis)
plt.title("Motif bruité")
cb = plt.colorbar(ax.imshow(matrix, cmap=plt.cm.spring))
plt.show()

print(dynamics(MotifL_Bruite, Liste_Motifs, 5))
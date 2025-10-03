#STEP 1: créer un modèle tabulaire simple avec de fortes variations de Vp et Vs mais une constante augmentation:

import numpy as np
import pandas as pd

# Vecteur des épaisseurs en km
thickness = np.array([8, 12, 30]) * 1e-3
# Vecteur des vitesses Vs en km/s
Vs = np.array([267.22, 412.27, 1055.74]) * 1e-3
# Vecteur des vitesses Vp en km/s
Vp = np.array([491.99, 1815.82, 2115.03]) * 1e-3
# Vecteur des densités en g/cm³
density = np.array([1057.16, 1294.62, 1807.33]) * 1e-3
velocity_model = np.column_stack((thickness, Vp, Vs, density))

#STEP2: créer la courbe de dispersion associée:
from disba import PhaseDispersion

def compute_dispersion_curve(velocity_model):
    '''
    Input:
    Vs: vector, Shear waves velocities

    Output:
    d: vector, Rayleigh phase velocities at different frequencies
    '''
    #définir plage de fréquence d'observation
    fmin= 1
    fmax= 50

    #créer vecteur de temps correspondant:
    t = np.linspace(1/fmax, 1/fmin, 50)
    print('1/t',1/t)

    # Implement the computation of dispersion curve based on the velocity model
    pd = PhaseDispersion(*velocity_model.T)
    cpr = [pd(t, mode=i, wave="rayleigh") for i in range(3)]
    return cpr

cpr=compute_dispersion_curve(velocity_model)

#STEP3: représentation graphique de la courbe de dispersion:
from matplotlib import pyplot as plt

def polynomial_regression(x, y, degree=3):
    # Ajustement polynomial (exemple degré 3)
    coeffs = np.polyfit(x, y, degree)  # Ajuste un polynôme de degré 3
    poly = np.poly1d(coeffs)  # Crée une fonction polynomiale
    return poly

from scipy.interpolate import interp1d

def interpolate_curve(x, y, x_new):
    # Interpolation de y sur z
    interp_func = interp1d(x, y, kind='linear', fill_value="extrapolate")
    y_projected = interp_func(x_new)
    # mais faire en sorte que les valeurs au dessus de xmax et en dessous de xmin soient nulles:
    for i in range(len(y_projected)):
        if x_new[i]>max(x):
            y_projected[i]=0
        if x_new[i]<min(x):
            y_projected[i]=0

    return y_projected

def plot_dispcurves(cpr):
    '''
    Plot the dispersion curve
    '''
    fig, ax = plt.subplots()
    color=['blue','red','green']
    projection=[0,0,0]
    for i in range(3):
        col=color[i]
        print('len M:',len(cpr[i].velocity*1e3))
        ax.plot(cpr[i].velocity*1e3,1/cpr[i].period, label=f"Mode {i}",color=col)

        freqs = np.linspace(1, 50, 50)
        projection[i]=interpolate_curve(1/cpr[i].period,cpr[i].velocity*1e3, freqs)
        #projection graphique = projection sans les zeros (aussi les retirer dans les frequences)
        graphproj=[projection[i][j] for j in range(len(projection[i])) if projection[i][j]!=0]
        graphfreqs=[freqs[j] for j in range(len(projection[i])) if projection[i][j]!=0]
        ax.plot(graphproj, graphfreqs, label=f"Interpolation", color=col, linestyle=':')

    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Dispersion curve')
    #xlim et ylim:
    ax.set_xlim([0, max(cpr[0].velocity*1e3)])
    ax.set_ylim([1, 50])
    ax.invert_yaxis()
    ax.legend()
    plt.show()
    plt.close()

    return freqs,projection

freqs,projection=plot_dispcurves(cpr)

#STEP4: Faire en sorte de mettre toutes les courbes au même format pour les utiliser comme input dans un modèle de deep learning:

#Appliquer une série de transformations:
#1) Créer un vecteur de fréquences global:
print('Frequency vector:',freqs)

#3) Projection sur un vecteur freqs global:
#construire un tableau de fréquences pour chaque mode:
proj1=projection[0];print('Projection 1:',proj1)
proj2=projection[1];print('Projection 2:',proj2)
proj3=projection[2];print('Projection 3:',proj3)







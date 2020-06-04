#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tarea 2
"""

""" 
Autor: Jorge Muñoz Taylor
Carné: A53863
Curso: Modelos probabilísticos de señales y sistemas para ingeniería
Grupo: 01
Fecha: 3/06/2020
"""

import sys
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import rayleigh
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import expon
from scipy.stats import kurtosis
from scipy.stats import skew
from math import sqrt


if __name__=="__main__":

    archivo = sys.argv
    BINS = 120

    # Array vacío, posteriormente se llenará con los datos extraídos del CSV.
    DATOS = genfromtxt (archivo[1], delimiter=',')
    

    """
    4) Graficar un histograma.
    """
    plt.hist (DATOS, color ="lightblue",bins=BINS)
    plt.xlabel("Datos")
    plt.ylabel("Frecuencia de aparición")
    plt.show() 


    """
    5) Con los datos contenidos en datos.csv, encontrar la mejor curva de ajuste y 
      graficar la curva de ajuste encontrada
    """
    #Gráfica todas las curvas.
    plt.hist (DATOS, bins=BINS, alpha=0.7, color="lightblue", normed=True)

    xt = plt.xticks()[0]  
    xmin, xmax = 0, max(xt)  
    lnspc = np.linspace(xmin, xmax, len(DATOS))

    c, d = rayleigh.fit(DATOS)   
    modelo = rayleigh (c,d)
    pdf_g = rayleigh.pdf(lnspc, c, d)   
    plt.plot(lnspc, pdf_g, 'k-', lw=5, alpha=1, color="blue", label='Rayleigh') 
   
    e, f = norm.fit(DATOS)   
    pdf_g = norm.pdf(lnspc, e, f)   
    plt.plot(lnspc, pdf_g, 'r-', lw=2, alpha=0.5, color="red", label='Normal') 

    g, h = uniform.fit(DATOS)   
    pdf_g = uniform.pdf(lnspc, g, h)   
    plt.plot(lnspc, pdf_g, 'r-', lw=2, alpha=0.5, color="black", label='Uniforme') 

    i, j = expon.fit(DATOS)   
    pdf_g = expon.pdf(lnspc, i, j)   
    plt.plot(lnspc, pdf_g, 'r-', lw=2, alpha=0.5, color="orange", label='Exponencial') 

    plt.xlabel("Datos")
    plt.ylabel("Frecuencia de aparición")
    plt.legend()
    plt.show()

    #Grafica sólo la curva de rayleigh.
    plt.hist (DATOS, bins=BINS, color="lightblue", normed=True)
    pdf_g = rayleigh.pdf(lnspc, c, d)   
    plt.plot(lnspc, pdf_g, 'k-', lw=5, alpha=1, color="blue", label='Rayleigh') 

    plt.xlabel("Datos")
    plt.ylabel("Frecuencia de aparición")
    plt.legend()
    plt.show()


    """
    6) Encontrar la probabilidad en el intervalo [a, b] en el modelo encontrado y 
      contrastarlo con la frecuencia relativa de los elementos de datos.csv que están 
      en realidad en ese mismo intervalo. En el carné A12345, a = min(23, 45) y b = max(23, 45). 
      Para este ejemplo, el intervalo es [23, 45]. Si fueran iguales, reemplazar por (12, 45).
    """
    a = min ( 38,63 ) 
    b = max ( 38,63 ) 

    print ("-> Intervalo: [", a , "," , b , "]\n")
    
    #Calcula la cantidad de datos en datos.csv.
    N = 0
    for dato in DATOS:
        N = N + 1

    #Calcula la cantidad de datos en el rango.
    N_intervalo = 0
    for dato in DATOS:
        if dato>=a and dato<=b:
            N_intervalo = N_intervalo+1

    print("-> Cantidad de números en el intervalo:", N_intervalo, "\n")

    #Obtiene la frecuencia relativa en datos.csv
    P_intervalo = N_intervalo / N

    print ("-> Frecuencia relativa de datos.csv:", P_intervalo, "\n") 
    print ("-> Probabilidad en el modelo encontrado:", modelo.cdf(b) - modelo.cdf(a) )


    """
    7) Calcular los cuatro primeros momentos y comentar sobre el “significado” de cada uno 
      y la correspondencia con la gráfica observada y los valores teóricos.
    """

    print ("\nMomentos a partir del modelo encontrado:\n")
    m,v,s,k = modelo.stats(moments='mvsk')
    print ("-> Media:    ", m) 
    print ("-> Varianza: ", v)
    print ("-> Skew:     ", s)
    print ("-> Curtosis: ", k)

    print ("\nMomentos a partir de los datos:\n")
    print( "-> Media:    " , np.mean (DATOS))
    print( "-> Varianza: " , np.var (DATOS))
    print( "-> Skew:     " , skew (DATOS))
    print( "-> Curtosis: " , kurtosis (DATOS))


    """
    8) Si los valores de datos.csv son X y pasa por la transformación Y = sqrt(X), 
      graficar el histograma de Y. Oferta de inicio de época lluviosa: puntos extra (generosos)
      si encuentran la expresión para función de densidad de Y, por medio de la deducción
      analítica. Pero llame ya.
    """
    DATO_TRANSFORMADO = []
    
    for dato in DATOS:
        DATO_TRANSFORMADO.append ( sqrt(dato) )

               
    #Grafica el histograma.
    plt.hist (DATO_TRANSFORMADO, color ="green",bins=BINS)
    plt.xlabel("sqrt(Datos)")
    plt.ylabel("Frecuencia de aparición")
    plt.show()

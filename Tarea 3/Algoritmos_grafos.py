#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:39:54 2019

@author: hectorgarcia
"""



import networkx as nx
#import matplotlib.pyplot as plt
from time import time
from networkx.algorithms import approximation as app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from random import randint
from datetime import datetime

tiempo_global_i=time()

random.seed(datetime.now())

dicrio={}      
archivo=pd.read_csv('/Users/hectorgarcia/Desktop/2Semestre/Redes/Tarea3/lectura_grafo3.txt', sep='  ', header=None)
adyacencia=np.matrix(archivo)
#---------------------------------Max Flow-------------------------------------
dicrio['Max Flow']={}

F=nx.Graph()
G=nx.Graph()
H=nx.Graph()
I=nx.Graph()
J=nx.Graph()

for n in (F,G,H,I,J):
    cota=randint(20,40)
    restantes=randint(0,40-cota)
    
    for i in range(cota):
        for j in range(cota):
            if adyacencia[i+restantes,j+restantes]!=0 and j!=cota-1:
                n.add_edges_from([(i,j)], capacity=randint(20,40))
              
auxiliar={}
for i in range(30):
    auxiliar[i+1]=[]
    tiempo_acumulado=0
    while True:
        tiempo_inicial=time()
        for n in (F,G,H,I,J):
            nx.maximum_flow(n,0,len(n.nodes)-1)   
        tiempo_final=time()
        tiempo_ejecucion=tiempo_final-tiempo_inicial
        tiempo_acumulado+=tiempo_ejecucion
        auxiliar[i+1].append(1000*tiempo_ejecucion)
        if(tiempo_acumulado>=5):
            break

k=1
for n in (F,G,H,I,J):
    dicrio['Max Flow'][k]={}
    dicrio['Max Flow'][k]['nodos']=len(n.nodes)
    dicrio['Max Flow'][k]['aristas']=len(n.edges)
    k+=1
    
medias_diccionario=[]
varianza_diccionario=[]
for i in range(30):
    medias_diccionario.append(np.mean(auxiliar[i+1]))
    varianza_diccionario.append(np.std(auxiliar[i+1]))
dicrio['Max Flow']['medias_repeticiones']=medias_diccionario
dicrio['Max Flow']['std_repeticiones']=varianza_diccionario
    
dicrio['Max Flow']['media_algoritmo']=np.mean(dicrio['Max Flow']['medias_repeticiones'])
dicrio['Max Flow']['std_algoritmo']=np.std(dicrio['Max Flow']['medias_repeticiones'])

#----------------------------Treewidth Min Degree------------------------------
dicrio['Tree_ Min Degree']={}

F=nx.Graph()
G=nx.Graph()
H=nx.Graph()
I=nx.Graph()
J=nx.Graph()

for n in (F,G,H,I,J):
    cota=randint(20,40)
    restantes=randint(0,40-cota)
    
    for i in range(cota):
        for j in range(cota):
            if adyacencia[i+restantes,j+restantes]!=0 and j!=cota-1:
                n.add_weighted_edges_from([(i,j,randint(20,40))])
                 
auxiliar={}
for i in range(30):
    auxiliar[i+1]=[]
    tiempo_acumulado=0
    while True:
        tiempo_inicial=time()
        for n in (F,G,H,I,J):
            app.treewidth_min_degree(n)
        tiempo_final=time()
        tiempo_ejecucion=tiempo_final-tiempo_inicial
        tiempo_acumulado+=tiempo_ejecucion
        auxiliar[i+1].append(10000*tiempo_ejecucion)
        if(tiempo_acumulado>=5):
            break

k=1
for n in (F,G,H,I,J):
    dicrio['Tree_ Min Degree'][k]={}
    dicrio['Tree_ Min Degree'][k]['nodos']=len(n.nodes)
    dicrio['Tree_ Min Degree'][k]['aristas']=len(n.edges)
    k+=1
    
medias_diccionario=[]
varianza_diccionario=[]
for i in range(30):
    medias_diccionario.append(np.mean(auxiliar[i+1]))
    varianza_diccionario.append(np.std(auxiliar[i+1]))
dicrio['Tree_ Min Degree']['medias_repeticiones']=medias_diccionario
dicrio['Tree_ Min Degree']['std_repeticiones']=varianza_diccionario
    
dicrio['Tree_ Min Degree']['media_algoritmo']=np.mean(dicrio['Tree_ Min Degree']['medias_repeticiones'])
dicrio['Tree_ Min Degree']['std_algoritmo']=np.std(dicrio['Tree_ Min Degree']['medias_repeticiones'])
#------------------------------------------------Shortest Path---------------------------------------------------
dicrio['Shortest Path']={}

F=nx.Graph()
G=nx.Graph()
H=nx.Graph()
I=nx.Graph()
J=nx.Graph()

for n in (F,G,H,I,J):
    cota=randint(20,40)
    restantes=randint(0,40-cota)
    
    for i in range(cota):
        for j in range(cota):
            if adyacencia[i+restantes,j+restantes]!=0 and j!=cota-1:
                n.add_weighted_edges_from([(i,j,randint(20,40))])
                  
auxiliar={}
for i in range(30):
    auxiliar[i+1]=[]
    tiempo_acumulado=0
    while True:
        tiempo_inicial=time()
        for n in (F,G,H,I,J):
            ([p for p in nx.all_shortest_paths(n,source=0,target=len(n.nodes)-1)])                        
        tiempo_final=time()
        tiempo_ejecucion=tiempo_final-tiempo_inicial
        tiempo_acumulado+=tiempo_ejecucion
        auxiliar[i+1].append(10000*tiempo_ejecucion)
        if(tiempo_acumulado>=5):
            break

k=1
for n in (F,G,H,I,J):
    dicrio['Shortest Path'][k]={}
    dicrio['Shortest Path'][k]['nodos']=len(n.nodes)
    dicrio['Shortest Path'][k]['aristas']=len(n.edges)
    k+=1
    
medias_diccionario=[]
varianza_diccionario=[]
for i in range(30):
    medias_diccionario.append(np.mean(auxiliar[i+1]))
    varianza_diccionario.append(np.std(auxiliar[i+1]))
dicrio['Shortest Path']['medias_repeticiones']=medias_diccionario
dicrio['Shortest Path']['std_repeticiones']=varianza_diccionario
    
dicrio['Shortest Path']['media_algoritmo']=np.mean(dicrio['Shortest Path']['medias_repeticiones'])
dicrio['Shortest Path']['std_algoritmo']=np.std(dicrio['Shortest Path']['medias_repeticiones'])
      
#---------------------------------DFS tree-------------------------------------
dicrio['DFS Tree']={}

F=nx.Graph()
G=nx.Graph()
H=nx.Graph()
I=nx.Graph()
J=nx.Graph()

for n in (F,G,H,I,J):
    cota=randint(20,40)
    restantes=randint(0,40-cota)
    
    for i in range(cota):
        for j in range(cota):
            if adyacencia[i+restantes,j+restantes]!=0 and j!=cota-1:
                n.add_weighted_edges_from([(i,j,randint(20,40))])
                   
auxiliar={}
for i in range(30):
    auxiliar[i+1]=[]
    tiempo_acumulado=0
    while True:
        tiempo_inicial=time()
        for n in (F,G,H,I,J):
            nx.dfs_tree(n,source=6) 
        tiempo_final=time()
        tiempo_ejecucion=tiempo_final-tiempo_inicial
        tiempo_acumulado+=tiempo_ejecucion
        auxiliar[i+1].append(10000*tiempo_ejecucion)
        if(tiempo_acumulado>=5):
            break

k=1
for n in (F,G,H,I,J):
    dicrio['DFS Tree'][k]={}
    dicrio['DFS Tree'][k]['nodos']=len(n.nodes)
    dicrio['DFS Tree'][k]['aristas']=len(n.edges)
    k+=1
    
medias_diccionario=[]
varianza_diccionario=[]
for i in range(30):
    medias_diccionario.append(np.mean(auxiliar[i+1]))
    varianza_diccionario.append(np.std(auxiliar[i+1]))
dicrio['DFS Tree']['medias_repeticiones']=medias_diccionario
dicrio['DFS Tree']['std_repeticiones']=varianza_diccionario
    
dicrio['DFS Tree']['media_algoritmo']=np.mean(dicrio['DFS Tree']['medias_repeticiones'])
dicrio['DFS Tree']['std_algoritmo']=np.std(dicrio['DFS Tree']['medias_repeticiones'])

#--------------------------------------Strongly Connected Components-----------------------------------------
dicrio['Strongly_CC']={}

F=nx.DiGraph()
G=nx.DiGraph()
H=nx.DiGraph()
I=nx.DiGraph()
J=nx.DiGraph()

for n in (F,G,H,I,J):
    cota=randint(20,40)
    restantes=randint(0,40-cota)
    
    for i in range(cota):
        for j in range(cota):
            if adyacencia[i+restantes,j+restantes]!=0 and j!=cota-1:
                n.add_weighted_edges_from([(i,j,randint(20,40))])
                   
auxiliar={}
for i in range(30):
    auxiliar[i+1]=[]
    tiempo_acumulado=0
    while True:
        tiempo_inicial=time()
        for n in (F,G,H,I,J):    
             [len(c) for c in sorted(nx.strongly_connected_components(n), key=len, reverse=True)]
        tiempo_final=time()
        tiempo_ejecucion=tiempo_final-tiempo_inicial
        tiempo_acumulado+=tiempo_ejecucion
        auxiliar[i+1].append(10000*tiempo_ejecucion)
        if(tiempo_acumulado>=5):
            break

k=1
for n in (F,G,H,I,J):
    dicrio['Strongly_CC'][k]={}
    dicrio['Strongly_CC'][k]['nodos']=len(n.nodes)
    dicrio['Strongly_CC'][k]['aristas']=len(n.edges)
    k+=1
    
medias_diccionario=[]
varianza_diccionario=[]
for i in range(30):
    medias_diccionario.append(np.mean(auxiliar[i+1]))
    varianza_diccionario.append(np.std(auxiliar[i+1]))
dicrio['Strongly_CC']['medias_repeticiones']=medias_diccionario
dicrio['Strongly_CC']['std_repeticiones']=varianza_diccionario
    
dicrio['Strongly_CC']['media_algoritmo']=np.mean(dicrio['Strongly_CC']['medias_repeticiones'])
dicrio['Strongly_CC']['std_algoritmo']=np.std(dicrio['Strongly_CC']['medias_repeticiones'])

######################Gráficas--------------------------

tiempo_global_f=time()
tiempo_global_total=tiempo_global_i-tiempo_global_f




y=[]
for i in ('DFS Tree','Max Flow','Shortest Path','Strongly_CC','Tree_ Min Degree'):
    for j in (1,2,3,4,5):
        y.append(dicrio[i][j]['nodos'])
for i in (1,2,3,4,5):
    if i==1:
        x=[dicrio['DFS Tree']['media_algoritmo']]
        xe=[dicrio['DFS Tree']['std_algoritmo']]
        plt.errorbar(5*x,y[0:5], xerr=5*xe, fmt='<',color='g',alpha=0.5)
    if i==2:
        x=[dicrio['Max Flow']['media_algoritmo']]
        xe=[dicrio['Max Flow']['std_algoritmo']]
        plt.errorbar(5*x,y[5:10], xerr=5*xe, fmt='^',color='violet',alpha=0.5)
    if i==3:
        x=[dicrio['Shortest Path']['media_algoritmo']]
        xe=[dicrio['Shortest Path']['std_algoritmo']]
        plt.errorbar(5*x,y[10:15], xerr=5*xe, fmt='>',color='aqua',alpha=0.5)
    if i==4:
        x=[dicrio['Strongly_CC']['media_algoritmo']]
        xe=[dicrio['Strongly_CC']['std_algoritmo']]
        plt.errorbar(5*x,y[15:20], xerr=5*xe, fmt='o',color='orange',alpha=0.5)
    if i==5:
        x=[dicrio['Tree_ Min Degree']['media_algoritmo']]
        xe=[dicrio['Tree_ Min Degree']['std_algoritmo']]
        plt.errorbar(5*x, y[20:25],xerr=5*xe, fmt='+',color='blue',alpha=0.5)
plt.xlabel('Tiempo (segundos)', size=14)
plt.ylabel('Cantidad de nodos', size=14)
plt.title('Cantidad de nodos contra tiempo',size=18)
plt.text(21.5, 40, r'DFS Tree',color='g',size=14)
plt.text(21.5, 38, r'Strongly C C',color='orange',size=14)
plt.text(21.5, 36, r'Max Flow',color='violet',size=14)
plt.text(21.5, 34, r'Shortest Path',color='aqua',size=14)
plt.text(21.5, 32, r'Tree Min Deg',color='blue',size=14)
plt.text(21.5, 25, r'Algoritmo',color='black',size=20)
plt.savefig('scater1.eps', format='eps', dpi=1000, bbox_inches='tight')

'Tiempo (segundos)'

y=[]
for i in ('DFS Tree','Max Flow','Shortest Path','Strongly_CC','Tree_ Min Degree'):
    for j in (1,2,3,4,5):
        y.append(dicrio[i][j]['aristas'])     
for i in (1,2,3,4,5):
    if i==1:
        x=[dicrio['DFS Tree']['media_algoritmo']]
        xe=[dicrio['DFS Tree']['std_algoritmo']]
        plt.errorbar(5*x,y[0:5], xerr=5*xe, fmt='+',color='g',alpha=0.5)
    if i==2:
        x=[dicrio['Max Flow']['media_algoritmo']]
        xe=[dicrio['Max Flow']['std_algoritmo']]
        plt.errorbar(5*x,y[5:10], xerr=5*xe, fmt='^',color='violet',alpha=0.5)
    if i==3:
        x=[dicrio['Shortest Path']['media_algoritmo']]
        xe=[dicrio['Shortest Path']['std_algoritmo']]
        plt.errorbar(5*x, y[10:15],xerr=5*xe, fmt='>',color='aqua',alpha=0.5)
    if i==4:
        x=[dicrio['Strongly_CC']['media_algoritmo']]
        xe=[dicrio['Strongly_CC']['std_algoritmo']]
        plt.errorbar(5*x,y[15:20], xerr=5*xe, fmt='o',color='orange',alpha=0.5)
    if i==5:
        x=[dicrio['Tree_ Min Degree']['media_algoritmo']]
        xe=[dicrio['Tree_ Min Degree']['std_algoritmo']]
        plt.errorbar(5*x,y[20:25], xerr=5*xe, fmt='<',color='blue',alpha=0.5)
plt.xlabel('Tiempo (segundos)', size=14)
plt.ylabel('Cantidad de aristas', size=14)
plt.title('Cantidad de aristas contra tiempo',size=18)
plt.text(21.5, 700, r'DFS Tree',color='g',size=14)
plt.text(21.5, 650, r'Strongly C C',color='orange',size=14)
plt.text(21.5, 600, r'Max Flow',color='violet',size=14)
plt.text(21.5, 550, r'Shortest Path',color='aqua',size=14)
plt.text(21.5, 500, r'Tree Min Deg',color='blue',size=14)
plt.text(21.5, 300, r'Algoritmo',color='black',size=20)
plt.savefig('scater2.eps', format='eps', bbox_inches='tight', dpi=1000)

for i in ('DFS Tree','Max Flow','Shortest Path','Strongly_CC','Tree_ Min Degree'):
    plt.xlabel('Tiempo (segundos)', size=14)
    plt.ylabel('Repeticiones', size=14)
    plt.title('Histograma del método '+i,size=18)
    plt.hist(dicrio[i]['medias_repeticiones'])
    plt.savefig(i+'.eps', format='eps',  dpi=1000)
    plt.show()
   























#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:56:16 2019

@author: hectorgarcia
"""

import numpy as np
import networkx as nx
from time import time
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from statsmodels.formula.api import ols
import statsmodels.api as sm

import plotly.plotly as py

def RandNodes(maxi,H): 
    li=[]
    for i in range(5):
        a=random.randint(1,maxi) 
        b=random.randint(1,maxi)  
        while a==b:
            b=random.randint(1,maxi)
        li.append( [a,b])
    return li

def AddEdges(S):
    lista=[]
    lista[:]=S.edges
    capacidad=np.random.normal(loc=15, scale=3,size=None)
    H=nx.Graph(S)
    for i in range(len(S.edges)):
        H.add_edge(lista[i][0], lista[i][1], capacity=capacidad)
    return H

def correr(generator, algorithm, cardinalidad, l):
    if generator==0:
        G=AddEdges(nx.dense_gnm_random_graph(cardinalidad, int((0.5*size*(size-1))-(1-((densidad+1)/10))*(0.5*size*(size-1)))))
        nodes=RandNodes(cardinalidad-1,G)
        data[contador,3]=nx.density(G)
        if algorithm==0:
            nx.maximum_flow_value(G,nodes[l][0],nodes[l][1])
        elif algorithm==1:
            nx.algorithms.flow.edmonds_karp(G,nodes[l][0],nodes[l][1])
        elif algorithm==2:
            nx.algorithms.flow.boykov_kolmogorov(G,nodes[l][0],nodes[l][1])
    elif generator==1:
        F=AddEdges(nx.generators.random_graphs.barabasi_albert_graph(cardinalidad, random.randint(1,cardinalidad-1)))
        nodes=RandNodes(cardinalidad-1,F)
        data[contador,3]=nx.density(F)
        if algorithm==0:
            nx.maximum_flow_value(F,nodes[l][0],nodes[l][1])
        elif algorithm==1:
            nx.algorithms.flow.edmonds_karp(F,nodes[l][0],nodes[l][1])
        elif algorithm==2:
            nx.algorithms.flow.boykov_kolmogorov(F,nodes[l][0],nodes[l][1])
    elif generator==2:
        J=AddEdges(nx.generators.duplication.duplication_divergence_graph(cardinalidad, random.random(), seed=None))
        nodes=RandNodes(cardinalidad-1,J)
        data[contador,3]=nx.density(J)
        if algorithm==0:
            nx.maximum_flow_value(J,nodes[l][0],nodes[l][1])
        elif algorithm==1:
            nx.algorithms.flow.edmonds_karp(J,nodes[l][0],nodes[l][1])
        elif algorithm==2:
            nx.algorithms.flow.boykov_kolmogorov(J,nodes[l][0],nodes[l][1])
 
data=np.arange(1800*5, dtype=float).reshape(1800,5)
contador=0           
for generador in range(3):
    for algoritmo in range(3):
        for orden in (16,64,256,1024):
            size=orden
            for densidad in range(10):
                for repeticion in range(5):
                    tiempo_inicial=time()
                    correr(generador, algoritmo, size, repeticion)
                    tiempo_final=time()
                    tiempo_ejecucion=tiempo_final-tiempo_inicial
                    data[contador,0]=generador
                    data[contador,1]=algoritmo
                    data[contador,2]=orden
                    data[contador,4]=tiempo_ejecucion
                    contador+=1
                

data1=data.copy()
data2=pd.DataFrame(data1)
data3=data2[['Orden','Densidad','Tiempo']].copy()

corr = data3.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='RdBu', vmin=-1, vmax=1, aspect='equal', origin='lower')
fig.colorbar(cax)
ticks = np.arange(0,len(data3.columns),1)
ax.set_xticks(ticks)
#plt.xticks(rotation=0)
ax.set_yticks(ticks)
ax.set_xticklabels(('Genera.', 'Algorit.', 'Orden', 'Densidad','Tiempo'))
ax.set_yticklabels(('Generador', 'Algoritmo', 'Orden', 'Densidad','Tiempo'))
plt.title('Matriz Correlaciones',pad=16.0, size=14)
plt.savefig('Correlaciones1.eps', format='eps', dpi=1000,bbox_inches='tight')



data1=data.copy()

for color in ('blue', 'green', 'red'):
    for marker in ('s','^','o'):
        if color=='blue':
            aux1=0
        elif color=='green':
            aux1=1
        elif color=='red':
            aux1=2
        else:
            print('error')
        if marker=='s':
            aux2=0
        elif marker=='^':
            aux2=1
        elif marker=='o':
            aux2=2
        else:
            print('error')
        x=[i for i in range(len(data1)) if (data1[i,0]==aux1 and data1[i,1]==aux2)]
        y=[data1[i,4] for i in range(len(data1)) if (data1[i,0]==aux1 and data1[i,1]==aux2)]
        plt.scatter(x, y, marker=marker, c=color)
        plt.ylim(0,1)
plt.xlabel('Observaciones', size=14)
plt.ylabel('Tiempo', size=14)
plt.title('Algoritmo y Generador contra tiempo',size=18)
blue_patch = mpatches.Patch(color='blue', label='Random Graph')
green_patch = mpatches.Patch(color='green', label='Barabasi-Albert')
red_patch = mpatches.Patch(color='red', label='Duplicacion-divergencia')
cuadrado_line = mlines.Line2D([], [],color='black', marker='s', markersize=10, label='Flujo Maximo')
triangulo_line = mlines.Line2D([], [],color='black', marker='^', markersize=10, label='Edmund Karp')
circulo_line = mlines.Line2D([], [],color='black', marker='o', markersize=10, label='Boykov_Kolmogorov')
plt.legend(handles=[blue_patch,green_patch,red_patch,cuadrado_line,triangulo_line,circulo_line],bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
plt.savefig('scater1.eps', format='eps', dpi=1000, bbox_inches='tight')

"""
data2[0].describe() 

corrmat = data2.corr() 
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 

cg = sns.clustermap(corrmat, cmap ="YlGnBu", linewidths = 0.1); 
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0) 

np.corrcoef(data2, rowvar=False)
"""


data2.columns=['Generador','Algoritmo','Orden','Densidad','Tiempo'] 
model_name = ols('Tiempo ~ Generador+Algoritmo+Orden+Densidad+Generador*Algoritmo+Algoritmo*Orden+Orden*Densidad+Generador*Orden+Generador*Densidad+Algoritmo*Densidad+Generador*Algoritmo*Orden+Generador*Algoritmo*Densidad*Algoritmo*Orden*Densidad+Generador*Orden*Densidad', data=data2).fit()
f = open('Ols1.txt', "w")
f.write('%s \t' %model_name.summary())
f.close()

aov_table = sm.stats.anova_lm(model_name, typ=2)
f = open('Anova1.txt', "w")
f.write('%s \t' %aov_table)
f.close()

"""   
set(data1[:,0])
data2[data2['Orden']]
data2['Orden'].unique()[1]
        if j=='Generador':
        formato='o'
        color='blue'
    elif j=='Algoritmo':
        formato=''
        color='blue'
    elif j=='Orden':
        formato='o'
        color='blue'  
        label=j+' '+str(int(i))
nombres.append(j+' '+str(int(i)))
        plt.xticks(nombres)  
        


x1=[1,2,3]
"""
"""
for j in ('Generador', 'Algoritmo','Orden'):  
    nombres=[]
    contador=0
    x1=[]
    fig, ax = plt.subplots()
    for i in data2[j].unique():
        data_filter= data2[data2[j] == i]
        plt.errorbar(i,np.mean(data_filter['Tiempo']), yerr=np.std(data_filter['Tiempo']),fmt='o',color='black',alpha=10) 
        #contador+=1
        nombres.append(str(int(i)))
        #print('2aqui')
        x1.append(int(i))
    plt.ylim(0,)
    plt.xlabel(j, size=14)
    plt.ylabel('Tiempo', size=14)
    plt.title('Errorbars',size=18)
    labels = nombres
    ax.set_xticks(x1)
    ax.set_xticklabels(labels, minor=False) 
    plt.show()
    
    plt.savefig('errorGen.eps', format='eps', dpi=1000)
    
    plt.show()

"""('Algoritmo','Generador 2'), ('Algoritmo','Generador 3'),

H=nx.DiGraph()
H.add_edges_from([('Algoritmo','Generador'),('Generador','Orden'),
                  ('Orden','10 repeticiones'),('10 repeticiones','fuente-sumidero'), 
                  ('fuente-sumidero','5 repeticiones')])
pos=nx.kamada_kawai_layout(H)
nx.draw(H, pos, node_color="white", node_size=800, with_labels=True, 
        font_weight="bold", edgecolors="black")
plt.savefig('Quinto.eps', format='eps', dpi=1000, bbox_inches='tight')





















#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:20:09 2019

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
import math


orden=[20,30,40,50,60]
for i in range(len(orden)):
#----------------------------------Gráficos------------------------------------
                  #-----Visualizar Fuente y Sumidero-------
    H=nx.watts_strogatz_graph(orden[i], int(orden[i]/2) , 0.33 , seed=None)
    
    lista=[]
    lista[:]=H.edges
    width=np.arange(len(lista)*1,dtype=float).reshape(len(lista),1)
    for r in range(len(lista)):
        R=np.random.normal(loc=20, scale=5.0, size=None)
        width[r]=R
        H.add_edge(lista[r][0],lista[r][1],capacity=R)
    #h=plt.figure()
    initial=random.randint(0,round(len(H.nodes)/2))
    final=random.randint(initial,len(H.nodes)-2)
    while initial==final:
        initial=random.randint(0,round(len(H.nodes)/2))
        final=random.randint(initial,len(H.nodes)-2)
    T=nx.maximum_flow(H, initial, final)
                        #-------Flujo------
    lista_flujo=[]
    for t in T[1].keys():
        for m in T[1][t].keys():
            #print(T[1][t][m])
            if T[1][t][m]!=0:
                lista_flujo.append((t,m))
    edges = H.edges()
    weights = [(H[u][v]['capacity']/15) for u,v in edges]     
    pos1=[initial,final]
    pos=nx.circular_layout(H)
    h=nx.draw(H, pos, node_color="white", node_size=800, with_labels=True, font_weight="bold", edgecolors="black")
    h=nx.draw_networkx_edges(H, pos, edgelist=lista_flujo, width=weights, edge_color='blue', style='solid')
    h=nx.draw_networkx_nodes(H, pos, pos1, node_color="red", node_size=800, with_labels=True, font_weight="bold", edgecolors="black")
    plt.savefig('muestra'+str(i)+'.eps', format='eps', dpi=1000,bbox_inches='tight')
    plt.show()
    
    """
    H.clear()
    plt.clf()
    plt.close(h)
    """
#-------------------Experimentacion-----------------------
datos=np.arange(sum(orden)*15, dtype=float).reshape(sum(orden),15)
fila=0
for i in range(len(orden)):
    H=nx.watts_strogatz_graph(orden[i], int(orden[i]/2) , 0.33 , seed=None)
    initial=0
    final=0
    lista=[]
    lista[:]=H.edges
    width=np.arange(len(lista)*1,dtype=float).reshape(len(lista),1)
    for r in range(len(lista)):
        R=np.random.normal(loc=20, scale=5.0, size=None)
        width[r]=R
        H.add_edge(lista[r][0],lista[r][1],capacity=R)
    for w in range(orden[i]): 
        initial=random.randint(0,round(len(H.nodes)/2))
        final=random.randint(initial,len(H.nodes)-2)
        while initial==final:
            initial=random.randint(0,round(len(H.nodes)/2))
            final=random.randint(initial,len(H.nodes)-2)
        tiempo_inicial=time()
        T=nx.maximum_flow(H, initial, final)
        tiempo_final=time()
        datos[fila,0]=T[0]
        datos[fila,1]=tiempo_final-tiempo_inicial
#------------------Info Fuente---------------------------
        datos[fila,2]=nx.degree_centrality(H)[initial]
        datos[fila,3]=nx.clustering(H, nodes=initial)
        datos[fila,4]=nx.closeness_centrality(H, u=initial)
        datos[fila,5]=nx.load_centrality(H, v=initial)
        datos[fila,6]=nx.eccentricity(H, v=initial)
        datos[fila,7]=nx.pagerank(H, alpha=0.9, weight='weight')[initial]
#-----------------Info Sumidero---------------------------
        datos[fila,8]=nx.degree_centrality(H)[final]
        datos[fila,9]=nx.clustering(H, nodes=final)
        datos[fila,10]=nx.closeness_centrality(H, u=final)
        datos[fila,11]=nx.load_centrality(H, v=final)
        datos[fila,12]=nx.eccentricity(H, v=final)
        datos[fila,13]=nx.pagerank(H, alpha=0.9, weight='weight')[final]
        datos[fila,14]=orden[i]
        fila+=1
        """
        nx.draw(H, pos, node_color="white", node_size=800, with_labels=True, font_weight="bold", edgecolors="black")
        plt.show(block=False)
        nx.draw(H, pos, node_color="white", node_size=800, with_labels=True, font_weight="bold", edgecolors="black")
pos=nx.circular_layout(H)
        #plt.close()
        #pos.clear()
    
        edges = H.edges()
        weights = [math.log(H[u][v]['capacity']) for u,v in edges]     
        pos1=[initial,final]
        pos=nx.circular_layout(H)
        nx.draw(H, pos, node_color="white", node_size=800, with_labels=True, font_weight="bold", edgecolors="black")
        nx.draw_networkx_edges(H, pos, edgelist=lista_flujo, width=weights, edge_color='blue', style='solid')
        nx.draw_networkx_nodes(H, pos, pos1, node_color="red", node_size=800, with_labels=True, font_weight="bold", edgecolors="black")
        plt.show()
        #print(fila,1)
        """

data=datos.copy()
data=pd.DataFrame(data)
data.columns=['Z','Tiempo','Deg_fuente','Clstr_fuente','Clsns_fuente','Load_fuente','Ex_fuente','Prank_fuente','Deg_fuente','Clstr_sumidero','Clsns_sumidero','Load_sumidero','Ex_sumidero','Prank_sumidero','Orden'] 

#----------------------------------Correlación---------------------------------

corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='RdBu', vmin=-1, vmax=1, aspect='equal', origin='upper')
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
#plt.xticks(rotation=0)
#ax.set_yticks(ticks)
#ax.set_xticklabels(('Genera.', 'Algorit.', 'Orden', 'Densidad','Tiempo'))
#ax.set_yticklabels(('Generador', 'Algoritmo', 'Orden', 'Densidad','Tiempo'))
plt.title('Matriz Correlaciones',pad=16.0, size=14)
plt.savefig('Correlaciones.eps', format='eps', dpi=1000,bbox_inches='tight')

#----------------------------------Box plot---------------------------------
                   #----------------primero---------------
filtro20=data[data['Orden']==20]
filtro30=data[data['Orden']==30]
filtro40=data[data['Orden']==40]
filtro50=data[data['Orden']==50]
filtro60=data[data['Orden']==60]

data_aux20=filtro20['Tiempo']
data_aux30=filtro30['Tiempo']
data_aux40=filtro40['Tiempo']
data_aux50=filtro50['Tiempo']
data_aux60=filtro60['Tiempo']

to_plot=[data_aux20,data_aux30,data_aux40,data_aux50,data_aux60]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
#plt.ylim(-1,20)
plt.xlabel('Orden de grafo', size=14)
plt.ylabel('Tiempo (segundos)', size=14)
plt.savefig('boxtiempo.eps', format='eps', dpi=1000,bbox_inches='tight')
plt.show()

                     #----------------segundo---------------
data_aux20=filtro20['Clstr_fuente']
data_aux30=filtro30['Clstr_fuente']
data_aux40=filtro40['Clstr_fuente']
data_aux50=filtro50['Clstr_fuente']
data_aux60=filtro60['Clstr_fuente']
"""
data_aux=data['Clstr_fuente']
data_aux1=data['Clstr_sumidero']
"""

to_plot=[data_aux20,data_aux30,data_aux40,data_aux50,data_aux60]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
#plt.ylim(-1,20)
plt.xlabel('Orden de grafo', size=14)
plt.ylabel('Valores', size=14)
plt.savefig('box_clstrfuente.eps', format='eps', dpi=1000,bbox_inches='tight')
plt.show()


data_aux120=filtro20['Clstr_sumidero']
data_aux130=filtro30['Clstr_sumidero']
data_aux140=filtro40['Clstr_sumidero']
data_aux150=filtro50['Clstr_sumidero']
data_aux160=filtro60['Clstr_sumidero']
"""
data_aux=data['Clstr_fuente']
data_aux1=data['Clstr_sumidero']
"""

to_plot=[data_aux120,data_aux130,data_aux140,data_aux150,data_aux160]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
#plt.ylim(-1,20)
plt.xlabel('Orden de grafo', size=14)
plt.ylabel('Valores', size=14)
plt.savefig('box_clstrsumidero.eps', format='eps', dpi=1000,bbox_inches='tight')
plt.show()
                     #----------------tercero---------------

data_aux20=filtro20['Clsns_fuente']
data_aux30=filtro30['Clsns_fuente']
data_aux40=filtro40['Clsns_fuente']
data_aux50=filtro50['Clsns_fuente']
data_aux60=filtro60['Clsns_fuente']
"""
data_aux=data['Clstr_fuente']
data_aux1=data['Clstr_sumidero']
"""

to_plot=[data_aux20,data_aux30,data_aux40,data_aux50,data_aux60]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
#plt.ylim(-1,20)
plt.xlabel('Orden de grafo', size=14)
plt.ylabel('Valores', size=14)
plt.savefig('box_clsnsfuente.eps', format='eps', dpi=1000,bbox_inches='tight')
plt.show()


data_aux120=filtro20['Clsns_sumidero']
data_aux130=filtro30['Clsns_sumidero']
data_aux140=filtro40['Clsns_sumidero']
data_aux150=filtro50['Clsns_sumidero']
data_aux160=filtro60['Clsns_sumidero']
"""
data_aux=data['Clstr_fuente']
data_aux1=data['Clstr_sumidero']
"""

to_plot=[data_aux120,data_aux130,data_aux140,data_aux150,data_aux160]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
#plt.ylim(-1,20)
plt.xlabel('Orden de grafo', size=14)
plt.ylabel('Valores', size=14)
plt.savefig('box_clsnssumidero.eps', format='eps', dpi=1000,bbox_inches='tight')
plt.show()

                      #-------------cuarto----------------
                     
data_aux20=filtro20['Load_fuente']
data_aux30=filtro30['Load_fuente']
data_aux40=filtro40['Load_fuente']
data_aux50=filtro50['Load_fuente']
data_aux60=filtro60['Load_fuente']
"""
data_aux=data['Clstr_fuente']
data_aux1=data['Clstr_sumidero']
"""

to_plot=[data_aux20,data_aux30,data_aux40,data_aux50,data_aux60]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
#plt.ylim(-1,20)
plt.xlabel('Orden de grafo', size=14)
plt.ylabel('Valores', size=14)
plt.savefig('box_loadfuente.eps', format='eps', dpi=1000,bbox_inches='tight')
plt.show()


data_aux120=filtro20['Load_sumidero']
data_aux130=filtro30['Load_sumidero']
data_aux140=filtro40['Load_sumidero']
data_aux150=filtro50['Load_sumidero']
data_aux160=filtro60['Load_sumidero']
"""
data_aux=data['Clstr_fuente']
data_aux1=data['Clstr_sumidero']
"""

to_plot=[data_aux120,data_aux130,data_aux140,data_aux150,data_aux160]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
#plt.ylim(-1,20)
plt.xlabel('Orden de grafo', size=14)
plt.ylabel('Valores', size=14)
plt.savefig('box_loadsumidero.eps', format='eps', dpi=1000,bbox_inches='tight')
plt.show()

                    #-------------quinto-----------------

data_aux20=filtro20['Prank_fuente']
data_aux30=filtro30['Prank_fuente']
data_aux40=filtro40['Prank_fuente']
data_aux50=filtro50['Prank_fuente']
data_aux60=filtro60['Prank_fuente']
"""
data_aux=data['Clstr_fuente']
data_aux1=data['Clstr_sumidero']
"""

to_plot=[data_aux20,data_aux30,data_aux40,data_aux50,data_aux60]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
#plt.ylim(-1,20)
plt.xlabel('Orden de grafo', size=14)
plt.ylabel('Valores', size=14)
plt.savefig('box_pagefuente.eps', format='eps', dpi=1000,bbox_inches='tight')
plt.show()


data_aux120=filtro20['Prank_sumidero']
data_aux130=filtro30['Prank_sumidero']
data_aux140=filtro40['Prank_sumidero']
data_aux150=filtro50['Prank_sumidero']
data_aux160=filtro60['Prank_sumidero']
"""
data_aux=data['Clstr_fuente']
data_aux1=data['Clstr_sumidero']
"""

to_plot=[data_aux120,data_aux130,data_aux140,data_aux150,data_aux160]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
#plt.ylim(-1,20)
plt.xlabel('Orden de grafo', size=14)
plt.ylabel('Valores', size=14)
plt.savefig('box_pagesumidero.eps', format='eps', dpi=1000,bbox_inches='tight')
plt.show()


"""
                           #-------------sexto-----------------

data_aux20=filtro20['Prank_fuente']
data_aux30=filtro30['Prank_fuente']
data_aux40=filtro40['Prank_fuente']
data_aux50=filtro50['Prank_fuente']
data_aux60=filtro60['Prank_fuente']
"""
data_aux=data['Clstr_fuente']
data_aux1=data['Clstr_sumidero']
"""

to_plot=[data_aux20,data_aux30,data_aux40,data_aux50,data_aux60]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
#plt.ylim(-1,20)
plt.xlabel('Generador de grafo', size=14)
plt.ylabel('Tiempo (segundos)', size=14)
plt.title('Generador contra tiempo',size=18)


data_aux120=filtro20['Prank_sumidero']
data_aux130=filtro30['Prank_sumidero']
data_aux140=filtro40['Prank_sumidero']
data_aux150=filtro50['Prank_sumidero']
data_aux160=filtro60['Prank_sumidero']
"""
data_aux=data['Clstr_fuente']
data_aux1=data['Clstr_sumidero']
"""

to_plot=[data_aux120,data_aux130,data_aux140,data_aux150,data_aux160]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
#plt.ylim(-1,20)
plt.xlabel('Generador de grafo', size=14)
plt.ylabel('Tiempo (segundos)', size=14)
plt.title('Generador contra tiempo',size=18)            

"""

#---------------------------------Fin boxplot----------------------------------

#---------------------------------ols y anova----------------------------------

model_name = ols('Z~Deg_fuente+Clstr_fuente+Clsns_fuente+Load_fuente+Ex_fuente+Prank_fuente+Deg_fuente+Clstr_sumidero+Clsns_sumidero+Load_sumidero+Ex_sumidero+Prank_sumidero', data=data).fit()
f = open('Ols.txt', "w")
f.write('%s \t' %model_name.summary())
f.close()

aov_table = sm.stats.anova_lm(model_name, typ=2)
f = open('Anova1.txt', "w")
f.write('%s \t' %aov_table)
f.close()

#---------------------------------otros----------------------------------



nx.draw(T[1])

plt.savefig('boxplotgenerador.eps', format='eps', dpi=1000)

data1= data[data['Generador'] == 0]
data2= data2[data2['Generador'] == 1]
data3= data2[data2['Generador'] == 2]

tiempos1= data4['Tiempo']
tiempos2= data5['Tiempo']
tiempos3= data6['Tiempo']


to_plot=[tiempos1, tiempos2, tiempos3]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
plt.ylim(-1,20)
plt.xlabel('Generador de grafo', size=14)
plt.ylabel('Tiempo (segundos)', size=14)
plt.title('Generador contra tiempo',size=18)
plt.savefig('boxplotgenerador.eps', format='eps', dpi=1000)


data=pd.DataFrame(datos)

f = open('data.txt', "w")
f.write('%s \t' %data.to_string)
f.close()

np.savetxt('data1.txt', data.values, fmt=['%1.3f','%1.8f','%1.8f','%1.8f','%1.8f','%d','%1.8f','%1.8f','%1.8f','%1.8f','%d','%1.8f'])






#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:14:48 2019

@author: hectorgarcia
"""

import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.flow import shortest_augmenting_path
import pandas as pd

lineas=pd.read_excel('BD.xlsx', sheet_name='topologia',index_col=0)

aaristas=[]
for i in lineas.keys():
    for j in lineas.keys():
        try:
            if pd.isna(lineas[i][j])==False:
                aaristas.append((i,j))
        except:
            a=0

Grafo_prueba=nx.Graph()
Grafo_prueba.add_edges_from(aaristas)
pos = planar_layout(Grafo_prueba)
nx.draw_networkx_nodes(Grafo_prueba, pos_disp, nodelist=abiertos, node_color='green',node_shape='s')
nx.draw_networkx_nodes(Grafo_prueba, pos_disp, nodelist=cerrados, node_color='red',node_shape='s')
nx.draw_networkx_nodes(Grafo_prueba, pos_disp, nodelist=aux, alpha=0)
nx.draw_networkx_edges(Grafo_prueba, pos_disp, edge_color='white')
plt.axis('off')
plt.style.use('dark_background')
plt.show()

labels1={}
for i in lineas.keys():
    for j in lineas.keys():
        try:
            if pd.isna(lineas[i][j])==False:
                
        except:
            a=0

labels={('Fuente','H1'):'4120',('H1','H3'):'C-2057',('V1','V2'):'R-C508',
        ('H2','Fuente'):'A-2070', ('V3','Fuente'):'F-0434',('V4','Fuente'):'A-B876',
        ('V5','Fuente'):'A-0433'}
#----------------------------------Inicializar---------------------------------

#a_fuente=['4120', 'F-0434', 'A-B876', 'A-0433', 'A-2070']

cerrados=['4120', 'R-c508', 'C-2057']
abiertos=['A-2070',  'A-B876', 'F-0434', 'A-0433']
aux=['Aux','Aux1', 'Aux2', 'Aux3', 'Aux4']


Grafo_disp=nx.Graph()
Grafo_disp.add_edges_from([('4120','Aux'),('Aux','R-c508'),('R-c508','Aux1'),
                           ('Aux1','A-2070'),('Aux','C-2057'),
                           ('C-2057','Aux2'),('Aux2','F-0434'),
                           ('Aux2','Aux3'),('Aux3','A-B876'),
                           ('Aux3','Aux4'),('Aux4','A-0433')])

labels_disp={('4120','Aux'): 'H1', ('Aux','R-c508'): 'V1', 
             ('R-c508','Aux1'):'V2',('Aux1','A-2070'):'H3', 
             ('Aux','C-2057'): 'H2', ('C-2057','Aux2'): 'H4', 
             ('Aux2','F-0434'): 'V3',('Aux2','Aux3'): 'H5', 
             ('Aux3','A-B876'): 'V4', ('Aux3','Aux4'): 'H6', 
             ('Aux4','A-0433'): 'V5'}

labels={('H1','H2'): 'Aux', ('H1','V1'): 'Aux', ('H2','H4'):'C-2057',
        ('V1','V2'):'R-C508', ('V2','H3'): 'Aux1', ('H4','H5'): 'Aux2', 
        ('H4','V3'): 'Aux2',('H5','H6'): 'Aux3', ('H5','V4'): 'Aux3',
        ('H6','V5'): 'Aux4',('H3','Sumidero'): 'A-2070',
        ('V3','Sumidero'): 'F-0434', ('V4','Sumidero'): 'A-B876', 
        ('V5','Sumidero'): 'A-0433',('Fuente', 'H1'): '4120', 
        ('Fuente', 'H3'): 'A-2070', ('Fuente', 'V3'): 'F-0434',
        ('Fuente','V4'): 'A-B876', ('Fuente','V5'): 'A-0433'}

Grafo_lineas=nx.Graph()
Grafo_lineas.add_edges_from(list(labels.keys()))

                                #---------Graficar----------
nx.draw_networkx_nodes(Grafo_disp, pos_disp, nodelist=abiertos, node_color='green',node_shape='s')
nx.draw_networkx_nodes(Grafo_disp, pos_disp, nodelist=cerrados, node_color='red',node_shape='s')
nx.draw_networkx_nodes(Grafo_disp, pos_disp, nodelist=aux, alpha=0)
nx.draw_networkx_edges(Grafo_disp, pos_disp, edge_color='white')
plt.axis('off')
plt.style.use('dark_background')
plt.show()


                                 #--------La otra-----------
pos=nx.kamada_kawai_layout(Grafo_lineas)
nx.draw_networkx_edge_labels(Grafo_lineas,pos,edge_labels=labels,font_color='red',font_size=8)
nx.draw(Grafo_lineas, pos,node_color="white", node_size=300, with_labels=True, font_size=8, edgecolors="black")
plt.style.use('default')
plt.show()



#------------------------------------Lo bueno----------------------------------
demanda=250

for arista in labels.keys():
    print(arista)
    for j in abiertos:
        print(j)
        if labels[arista]==j:
            Grafo_lineas.add_edge('Fuente', arista[0], capacity=0)
            Grafo_lineas.add_edge(arista[0], arista[1], capacity=0)
            Grafo_lineas.add_edge(arista[0], 'Sumidero')
Grafo_lineas.remove_edge('Fuente','Sumidero')
pos=pos=nx.kamada_kawai_layout(Grafo_lineas)
nx.draw(Grafo_lineas, pos, node_color="white", node_size=800, with_labels=True, font_weight="bold", edgecolors="black")
plt.show()


llegadas=len(Grafo_lineas['Sumidero'])
for nodo in Grafo_lineas['Sumidero']:
    Grafo_lineas.add_edge('Sumidero', nodo, capacity=int(demanda/llegadas))

T=nx.maximum_flow(Grafo_lineas, 'Fuente', 'Sumidero', flow_func=shortest_augmenting_path)
        
lista=[]
for t in T[1].keys():
    for m in T[1][t].keys():
        if T[1][t][m]!=0:
            lista.append((m,T[1][t][m]))

lista_flujo=[]
pesos=[]
for i in range(len(lista)):
    for j in labels_disp.keys():
        if lista[i][0]==labels_disp[j]:
            lista_flujo.append(j)
            pesos.append(lista[i][1]/50)


Grafo_disp=nx.Graph()
Grafo_disp.add_edges_from([('4120','Aux'),('Aux','R-c508'),('R-c508','Aux1'),('Aux1','A-2070'),
                  ('Aux','C-2057'),('C-2057','Aux2'),('Aux2','F-0434'),('Aux2','Aux3'),('Aux3','A-B876'),
                  ('Aux3','Aux4'),('Aux4','A-0433')])

labels_disp={('4120','Aux'): 'H1', ('Aux','R-c508'): 'V1', ('R-c508','Aux1'):'V2',
        ('Aux1','A-2070'):'H3', ('Aux','C-2057'): 'H2', ('C-2057','Aux2'): 'H4', ('Aux2','F-0434'): 'V3',
        ('Aux2','Aux3'): 'H5', ('Aux3','A-B876'): 'V4', ('Aux3','Aux4'): 'H6', ('Aux4','A-0433'): 'V5'}
nx.draw_networkx_nodes(Grafo_disp, pos_disp, nodelist=abiertos, node_color='green',node_shape='s')
nx.draw_networkx_nodes(Grafo_disp, pos_disp, nodelist=cerrados, node_color='red',node_shape='s')
nx.draw_networkx_nodes(Grafo_disp, pos_disp, nodelist=aux, alpha=0)
nx.draw_networkx_edges(Grafo_disp, pos_disp, edgelist=lista_flujo, width=pesos, edge_color='yellow', style='solid')
#nx.draw_networkx_edges(Grafo_disp, pos_disp, edge_color='white')
plt.axis('off')
plt.style.use('dark_background')
plt.show()  


plt.savefig('circuito4.eps', format='eps', dpi=1000,bbox_inches='tight')



         



#-------------------------------Hasta aqui-------------------------------------

while True:

    demanda=str(input('Dispositivo(s) conectado(s) a fuente: '))
    
    


cerrados=['A-2070', '4120', 'R-c508', 'C-2057']
abiertos=['A-B876', 'F-0434', 'A-0433']
aux=['Aux','Aux1', 'Aux2', 'Aux3', 'Aux4']

cerrados=['4120', 'R-c508']
abiertos=['A-2070',  'A-B876', 'F-0434', 'A-0433', 'C-2057']
aux=['Aux','Aux1', 'Aux2', 'Aux3', 'Aux4']






#for camino in nx.all_simple_paths(Grafo_lineas, source='Sumidero', target='Fuente'):
 #   print(camino)







    salida=int(input('Desea actualizar el grafo? 1:Si, 2:No'))
    
    demanda=float(input('Cual es la demanda? :'))

    if salida==2:
        break






pos_lineas,

,('Fuente','6'),('Fuente','7'),('Sumidero','3'),
                  ('Sumidero','4'),('Sumidero','5'),('Sumidero','6'),('Sumidero','7'),
                  ('3','5'),('4','5'),('3','4'),('6','7')

nx.draw_networkx_nodes(Grafo_disp, pos_disp, nodelist=fijo, alpha=1)
('Fuente','4120'),,('Fuente','H2'),('Fuente','V3'),
                  ('Fuente','V4'),('Fuente','V5')
                  
                  ,('V5','Sumidero'),
                  ('V4','Sumidero'),('V3','Sumidero'),('H2','Sumidero'),('H2','Sumidero'),









archivo1=pd.read_excel('BD.xlsx', sheet_name='Hoja3', header=None)
archivo=archivo1.values

dispositivos=7

H=nx.Graph()
H.add_edge('Fuente',archivo[0,1])
label={}
label[('Fuente',archivo[0,1])]=archivo[0,0]
for i in range(len(archivo[0])-2):
    












G=nx.Graph()
G.add_edges_from([('Fuente','H1'),('H1','V1'),('V1','V2'),('V2','H2'),('H2','Sumidero'),
                  ('H1','H3'),('H3','V4'),('V4','V3'),('V3','V5'),('V5','Sumidero'),
                  ('V4','Sumidero'),('V3','Sumidero'),('H2','Sumidero')])
pos=nx.kamada_kawai_layout(G)
nx.draw(G, pos, node_color="white", node_size=800, with_labels=True, font_weight="bold", edgecolors="black")

    
g=nx.convert.to_dict_of_dicts(G)

plt.savefig('redes2.eps', format='eps', dpi=1000,bbox_inches='tight')











nx.write_edgelist(G, 'txt.txt', comments='#', delimiter=' ', data=True, encoding='utf-8')               
plt.savefig('redes1.eps', format='eps', dpi=1000,bbox_inches='tight')




















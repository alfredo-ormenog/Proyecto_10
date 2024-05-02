#!/usr/bin/env python
# coding: utf-8

# # PROYECTO 10 - TRIPLETEN
# 
# El proyecto se basará en la evaluación del estudio de mercado para un restaurante que tiene como diferenciación que los comensales serán atendidos por robots. El objetivo es analizar el estado actual del mercado y comprobar si se podría mantener el éxito en el competitivo mundo de los restaurantes ubicados en Los Ángeles.

# ## Descarga los datos

# In[54]:


# Importación de librerías necesarias para trabajar:
import pandas as pd
import numpy as np 
import scipy as sp
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import datetime as dt
from scipy import stats
import plotly.express as px


# #Descrpición de datos:
# 
# + Tabla rest_data:
# + id — Identificación de restaurante única
# + object_name — Nombre del establecimiento
# + chain — Establecimiento que pertenece a una cadena (TRUE/FALSE)
# + object_type — Tipo de establecimiento
# + address — Dirección
# + number — Número de asientos
# 

# In[55]:


#Lectura e información principal del DF data:

data = pd.read_csv('/datasets/rest_data_us_upd.csv')
print(data.head())
print("Cantidad de filas es: ", len(data))
print("La estructura del dataset es: ", data.shape)
print("Tipo de datos:")
data.dtypes
print()
print("Información estadística del dataset:")
data.info(memory_usage="deep")


# El DF data contiene información acerca de los restaurantes en la zona de LA. Nos brinda datos de ubicación, nombre, si pertenece a una cadena , el tipo , y la cantidad de comensales que puede permitirse atender por número de sillas. 

# ### Prepáralos para el análisis

# In[56]:


# Recopilar información acerca de valores ausentes y/o nulos.
data.info()

print()

print(data.isnull().sum())


# Se obtiene que la columna CHAIN tiene 3 valores ausentes.

# In[57]:


#Procedemos con la eliminación de los valores ausentes:
data = data.dropna()
data.info()
print()
print(data.isnull().sum())


# Se procede con la eliminación de esas tres filas ya que no representan un porcentaje que pueda afectar de manera concreta nuestro estudio de mercado frente a los restaurantes de la competencia. Adicional, se ha evaluado el tipo de variable que contiene cada una de las filas, se considera correcto para proceder.

# In[58]:


#Evaluación de valores duplicados:

print(data.duplicated().sum())


# ## Análisis de datos
# 

# ### Investiga las proporciones de los distintos tipos de establecimientos. Traza un gráfico.

# In[59]:


# Tipos de establecimientos:
data['object_type'].unique()


# In[60]:


#Cantidad de filas:
total_objtype = data['object_type'].count()
total_objtype


# Calcular la proporción referente al total:

# In[61]:


obj_cafe  = (data['object_type'] == 'Cafe').sum()
prop_cafe = obj_cafe / total_objtype


# In[62]:


obj_rest  = (data['object_type'] == 'Restaurant').sum()
prop_rest = obj_rest / total_objtype


# In[63]:


obj_fastfood  = (data['object_type'] == 'Fast Food').sum()
prop_fastfood = obj_fastfood / total_objtype


# In[64]:


obj_bakery  = (data['object_type'] == 'Bakery').sum()
prop_bakery = obj_bakery / total_objtype


# In[65]:


obj_bar  = (data['object_type'] == 'Bar').sum()
prop_bar = obj_bar / total_objtype


# In[66]:


obj_pizza  = (data['object_type'] == 'Pizza').sum()
prop_pizza = obj_pizza / total_objtype


# In[67]:


#Gráfico:

keys = ['Cafe', 'Restaurant', 'Fast Food', 'Bakery', 'Bar', 'Pizza']
vals = [prop_cafe, prop_rest,prop_fastfood, prop_bakery, prop_bar, prop_pizza]

with plt.style.context('seaborn-pastel'):
    plt.bar(keys,vals)
    plt.title('Proporción de tipos de establecimientos en LA')
    plt.show()


# Al analizar el gráfico se tiene que el tipo de establecimiento mayoritario en este territorio corresponde al Restaurant con más del 70%, en segundo lugar estaría Fast Food con a penas un poco más del 10%.

# ###  Investiga las proporciones de los establecimientos que pertenecen a una cadena y de los que no. Traza un gráfico.

# In[68]:


data


# In[69]:


total_chain = data['chain'].count()
total_chain


# In[70]:


chain_false = (data['chain'] == False).sum()
chain_false


# In[71]:


prop_false = chain_false / total_chain
prop_false


# In[72]:


chain_true = (data['chain'] == True).sum()
chain_true


# In[73]:


prop_true = chain_true / total_chain
prop_true


# In[74]:


#Gráfico:

keys = ['Pertenecen' , 'No pertenecen']
vals = [prop_true , prop_false]

with plt.style.context('seaborn-pastel'):
    plt.bar(keys,vals)
    plt.title('Proporción de restaurantes que pertenecen a una cadena en LA')
    plt.show()


# Según el gráfico tenemos que la mayoría de restaurantes indpendientemente del tipo por evaluar, no pertenecen a una cadena, es decir, son originales/propios. Con este dato podemos revisar que la competencia más fuerte proviene de emprendedores.

# ### ¿Qué tipo de establecimiento es habitualmente una cadena?

# In[75]:


true_chain = (data[data['chain'] == True])
true_chain


# In[76]:


count_type = true_chain['object_type'].value_counts().to_frame()
count_type


# In[77]:


# Gráfico:

count_type.plot(kind='bar', title= 'Cantidad de restaurantes que pertenecen a cadenas en LA', rot = 45)


# Por orden ,se puede observar que nuevamente, los que son tipo restaurante en su mayoría son los que pertenecen a una cadena de negocio.

# ### ¿Qué caracteriza a las cadenas: muchos establecimientos con un pequeño número de asientos o unos pocos establecimientos con un montón de asientos?

# In[78]:


true_chain


# In[79]:


print(data['number'].min())
print(data['number'].max())


# In[80]:


data['number'].describe()


# In[81]:


group_1 = (data['number'] <= 50).sum()
group_1


# In[82]:


group_2 = (data['number'] >= 51).sum()
group_2


# In[83]:


#Gráfico:

keys = ['Rest con menos de 50 asientos' , 'Rest con más de 50 asientos']
vals = [group_1 , group_2]

with plt.style.context('seaborn-pastel'):
    plt.bar(keys,vals)
    plt.title('Relación de restaurantes respecto a número de asientos')
    plt.show()


# Lo que caracteriza a las cadenas por lo que hemos obtenido es que en su mayoría hay muchos establecimientos con pocos asientos.
# 
# 

# A continuación, vamos a encontrar por grupo de 25, qué tantos restaurantes participan de esa distribución:

# In[84]:


group_25 = (data['number'] <= 25).sum()
group_25


# In[85]:


group_50 = ((data['number'] >= 26) & (data['number'] <= 50)).sum()
group_50


# In[86]:


group_75 = ((data['number'] >= 51) & (data['number'] <= 75)).sum()
group_75


# In[87]:


group_100 = ((data['number'] >= 76) & (data['number'] <= 100)).sum()
group_100


# In[88]:


group_125 = ((data['number'] >= 101) & (data['number'] <= 125)).sum()
group_125


# In[89]:


group_150 = ((data['number'] >= 126) & (data['number'] <= 150)).sum()
group_150


# In[90]:


group_175 = ((data['number'] >= 151) & (data['number'] <= 175)).sum()
group_175


# In[91]:


group_200 = ((data['number'] >= 176) & (data['number'] <= 200)).sum()
group_200


# In[92]:


group_229 = ((data['number'] >= 201) & (data['number'] <= 230)).sum()
group_229


# In[93]:


#Gráfico:

keys = ['de 1 a 26', 'de 27 a 50', 'de 51 a 75','de 76 a 100', 'de 101 a 125', 'de 126 a 150', 'de 151 a 175', 'de 176 a 200', 'de 201 a 229']
vals = [group_25 , group_50, group_75, group_100, group_125, group_150, group_175, group_200, group_229]

with plt.style.context('seaborn-pastel'):
    plt.figure(figsize=(16, 8)) 
    plt.bar(keys,vals)
    plt.title('Relación de restaurantes respecto a número de asientos')
    plt.show()


# En conclusión, para esta pregunta, se puede observar que las agrupaciones agregando asientos de 25 en 25, nos brinda que la respuesta se mantiene. La mayoría de establecimientos que pertencen a una cadena presentan menor cantidad de asientos

# ### Determina el promedio de número de asientos para cada tipo de restaurante. De promedio, ¿qué tipo de restaurante tiene el mayor número de asientos? Traza gráficos.

# In[94]:


prom_type = data[['object_type', 'number']]
prom_type


# In[95]:


graph_prom_type = prom_type.groupby('object_type').agg({'number':'mean'}).sort_values(by='number').round()
graph_prom_type


# In[96]:


# Gráfico:
import plotly.graph_objects as go
from plotly import graph_objects as go

graph_prom_type = dict( 
    promedio_asientos = [22, 25, 29, 32, 45 , 48] ,
    tipo_restaurante = ['Bakery', 'Cafe', 'Pizza', 'Fast Food', 'Bar', 'Restaurant'])

fig = px.funnel(graph_prom_type, x='promedio_asientos', y='tipo_restaurante',
                title='Promedio de número de asientos por tipo de restaurante')

fig.show()


# ### Coloca los datos de los nombres de las calles de la columna address en una columna separada.

# In[97]:


data['address'] = data['address'].astype(str)
data['address'] = data['address'].str.upper()


# In[98]:


data['name_street'] = data['address'].str.extract(r'\d{2,5}(.*)', expand=False).str.strip()
#ata['name_street'] = data['address'].str.extract(r'\d{2,4}(\d\w\s)\d{1,4}', expand=False).str.strip()

data.head(10)


# ARREGLAR ESTO


# ### Traza un gráfico de las diez mejores calles por número de restaurantes.

# In[99]:


best_streets = (data.groupby('name_street').agg({'object_name':'count'})).sort_values(by='object_name',ascending=False)

best_streets.head(10)


# In[100]:


best_streets_10 = best_streets.head(10)


# In[101]:


best_streets_10.plot(kind='bar', title= 'Las 10 mejores calles por número de restaurantes', rot = 45,figsize=[12,8])


# ### Encuentra el número de calles que solo tienen un restaurante.

# In[102]:


best_streets_min = (best_streets['object_name'] <= 1).sum()
best_streets_min


# Con esta información obtenemos que hay 2433 calles que presentan solo un restaurante en la misma.

# In[103]:


best_streets_max = (best_streets['object_name'] > 1).sum()
best_streets_max


# La diferencia, 628 calles tienen más de un restaurante en la misma

# ###  Para las calles con muchos restaurantes, analiza la distribución del número de asientos. ¿Qué tendencias puedes ver?

# In[104]:


distribucion = data.drop(['id','address','chain','object_type'] ,axis =1)
distribucion


# In[105]:


df_asientos = (distribucion.groupby(['name_street']).agg({'object_name':'count', 'number':'mean'})).sort_values(by='object_name',ascending=False)
df_asientos['number'] = df_asientos['number'].round(0)
df_asientos = df_asientos.reset_index()

more_50 = df_asientos['object_name'] > 50 

df_asientos_50 = df_asientos[more_50]
df_asientos_50.head()


# In[106]:


df_asientos_50['number'] = df_asientos_50['number'].astype(str).astype(float)
df_asientos_50['number'] = df_asientos_50['number'].astype(float).astype(int)


# In[107]:


df_asientos_50 = df_asientos_50.sort_values(by='number')


# In[108]:


# Generar listas para los gráficos:
number      = df_asientos_50['number']
number      = number.tolist()
name_street = df_asientos_50['name_street']
name_street = name_street.tolist()


# In[109]:


df_asientos_50.plot(kind='bar', rot = 45,figsize=[16,8] , x='name_street',y ='number' , color = 'brown' ,
                   title = 'Relación entre la cantidad de asientos y las calles con muchos restaurantes')


# In[110]:


sns.pairplot(df_asientos_50)


# Esta distribución ordenada de menor a mayor, podemos observar que la tendencia o mayoría de calles presentan un alto número de asientos. Posiblemente estas calles son altamente transitadas y según histórico de consumo se podría revelar uinformación que los comensales prefieren comer y estar en el local en vez de otras alternativas.

# Como conclusión de este proyecto, se puede decir que la atención realizada por los robots será diferenciada si pueden mantener y atender durante una jornada laboral constante a un alto número de asientos que es lo que existe actualmente en la zona estudiada de LA. Adicional, se puede observar que la tendencia de los restaurantes en las calles cuentan como obligatoriedad un alto nivel de asientos para atención al público en general por más que este represente aproximadamente el 21%.
# Esta propuesta de local y atención podrá ser beneficiosa si el recorrido de los robots y su tiempo de atención es eficiente o similar al realizado por un humano, de esta manera se podría evaluar también el nivel de servicio y proponer una diferenciación aún más marcada frente a los competidores.
# Se podría recomendar que el restaurante propuesto debe tener una mínima de 50 asientos que entraría entre los más destacados y en los que se encuentran en mas calles.
# Para el desarrollo de una cadena, se debe comprender que la mayoría de la competencia es un Restaurante, por lo que la propuesta podría tomar ese camino para respetar la idea de negocio original, y vendría con soporte porque es la tendencia actual de las cadenas.

# ## PRESENTACIÓN:
#     

# Presentation: <https://jupyterhub.tripleten-services.com/user/user-3-1b6adfc1-84b4-4cb5-b39d-70f194c321a6/files/Proyecto%2010%20-%20TripleTen_presentacion.pdf>

# 

# <a id="some_id"></a>

#!/usr/bin/env python
# coding: utf-8

# >[Tratamiento del dataset y análisis](#scrollTo=O9fwN1bdANni)
# 
# >>[Los datos representan lo siguiente](#scrollTo=Dw8euZjbsKuw)
# 
# >>[Hay valores que dan 0 y que no respresentan nada.](#scrollTo=AX00p7RJCKw9)
# 
# >>[Porcentaje de diabéticos](#scrollTo=PeXvKVOAfn7C)
# 
# >>[¿La glucosa tiene una distribución normal?](#scrollTo=SnxJYmgEK0mK)
# 
# >>[¿Cuál es el máximo número de embarazos?](#scrollTo=cJpXJC-dK5jY)
# 
# >>[¿La presión arterial tiene una distribución normal?](#scrollTo=qRQRhacZLMLN)
# 
# >>[Dispersión de los datos para descubrir patrones](#scrollTo=NMXZzF9mJQ6K)
# 
# >[Última etapa del preprocesamiento](#scrollTo=_tkWlzNUn7kE)
# 
# >>[Método Mice al conjunto de entrenamiento](#scrollTo=XYUvIDnYKHdk)
# 
# >>[Oversampling al conjunto de entrenamiento](#scrollTo=KhlMYxNzKehQ)
# 
# >>[Disminución de la dimensionalidad en busca de patrones (sólo para visualizar)](#scrollTo=KPy1OVypKlBZ)
# 
# >[Usando redes neuronales](#scrollTo=ax4oTQo1jxfL)
# 
# >>[Reescalado de valores](#scrollTo=-CqTSPGxK0dK)
# 
# >>[Búsqueda en grilla para hallar el mejor modelo de red neuronal](#scrollTo=ELc82p-cpLGl)
# 
# >>[Entreno la mejor red](#scrollTo=7flK7kEu8--p)
# 
# >>[Elección del threshold](#scrollTo=wBRNuV-gzwER)
# 
# >>[Curva ROC](#scrollTo=baJHbjo98M--)
# 
# >>[Evaluación del modelo en el conjunto de testeo](#scrollTo=99oq9blxaObX)
# 
# >>[Veamos la matriz de confusión](#scrollTo=TM0t_SxsJhB4)
# 
# >[Usando Boosting](#scrollTo=VlXtwPzP-60U)
# 
# >>[Evaluación de overfitting o underfitting](#scrollTo=HjD7AGsFgtVc)
# 
# >>[Veamos la influencia de cada feature (columna).](#scrollTo=AOgJB_xxAOgn)
# 
# >>[Elección del threshold](#scrollTo=8llI2fziU7UF)
# 
# >>[Evaluación del modelo en el conjunto de testeo](#scrollTo=Vil5JPNGU_XH)
# 
# >>[Matriz de Confusión](#scrollTo=LmLhQAcjxIe7)
# 
# >>[Visualización de un árbol](#scrollTo=WAowH8SM9kgh)
# 
# >>>[Otra forma de visualizar un árbol en específico](#scrollTo=2B4YbGVydLRP)
# 
# >[Usando SVM](#scrollTo=goCa75qIPYPk)
# 
# >>[Con LinearSVC](#scrollTo=jq0h5mkXzwCU)
# 
# >>>[Evaluación de overfitting o underfitting](#scrollTo=Zxup_MXcZFcU)
# 
# >>>[Veamos qué threshold elegir](#scrollTo=58copGUmZMDt)
# 
# >>>[Curva ROC](#scrollTo=XYdGPjB_apsG)
# 
# >>>[Evaluación del modelo en el conjunto de testeo](#scrollTo=X0ZIPsjTzZ4d)
# 
# >>[Con kernel RBF](#scrollTo=oCzyDrrkzqMk)
# 
# >>>[Evaluación de overfitting o underfitting](#scrollTo=jD5naW-XbKtn)
# 
# >>>[Veamos qué threshold elegir](#scrollTo=_Zj2c-K0bKto)
# 
# >>>[Curva ROC](#scrollTo=3g0X6QWRaYyk)
# 
# >>>[Evaluación del modelo en el conjunto de testeo](#scrollTo=_nNAqnxEbKto)
# 
# >[Con regresión logística](#scrollTo=F-gKFUsI560H)
# 
# >>[Ver si hay underfitting o overfitting](#scrollTo=VqxhtTtEEEzi)
# 
# >>[Hagamos una k-fold Cross Validation para ver la performance del modelo](#scrollTo=DYQcPSzZBDXa)
# 
# >>[Veamos qué threshold elegir](#scrollTo=D1hctrLNeIoE)
# 
# >>[Curva ROC](#scrollTo=M7BPbOENFKrh)
# 
# >>[Evaluación del modelo en el conjunto de testeo](#scrollTo=xN3dlfafeIoO)
# 
# >[Con Ensamble (Votting Classifier)](#scrollTo=aVvQ9IHV58gg)
# 
# >>[Evaluación si hay overfitting o underfitting](#scrollTo=IafyR75a4byE)
# 
# >>[Evaluación de métricas y elección de threshold](#scrollTo=BsnR2tGQ4imy)
# 
# >>[Curva ROC](#scrollTo=jLEFZd8942Rm)
# 
# >>[Evaluación en el conjunto de testeo](#scrollTo=XjqJ1iwq4x1z)
# 
# >[Conclusión](#scrollTo=DonlzNlpShPb)
# 
# 

# In[1]:

# Asegúrate de que keras-tuner esté instalado antes de ejecutar el script
try:
    import keras_tuner as kt
except ImportError:
    import os
    os.system('pip install keras-tuner')

# Resto de tu código

# Librerías
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
import os

# Para estadística
import statsmodels.api as sm
from scipy import stats
from scipy.stats import shapiro, norm

# Para aplicar MICE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Para separar el dataset
from sklearn.model_selection import train_test_split

# Para oversamplear datasets desbalanceados
from imblearn.over_sampling import RandomOverSampler

# Para visualización en 2d con reducción de dimensionalidad
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Para el entrenamiento de los modelos
import tensorflow as tf
from tensorflow import keras
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

# Para no sobreentrenar
from tensorflow.keras.callbacks import EarlyStopping

# Para usar búsqueda de hiperparámetros con TensorFlow
get_ipython().system('pip install -q keras_tuner')

import keras_tuner as kt

# Métricas para evaluar
from sklearn.metrics import (classification_report, precision_recall_curve, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay,balanced_accuracy_score)
from sklearn.model_selection import learning_curve, cross_validate

# Visualización de los árboles de decisión
get_ipython().system('pip install -q -U dtreeviz')

import dtreeviz
from xgboost import plot_tree
import logging

# Ignorar los warning
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Fijar la semilla para garantizar reproducibilidad
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
seed = 42

os.environ['TF_DETERMINISTIC_OPS'] = '1'


# # Tratamiento del dataset y análisis

# In[3]:


# Cargando el dataset
get_ipython().system('wget https://raw.githubusercontent.com/plotly/datasets/502daad486f112310367a20a74a00a9cc6e75240/diabetes.csv')

df = pd.read_csv('diabetes.csv')


# ## Los datos representan lo siguiente
# <table>
# <thead>
# <tr>
# <th>Columna</th>
# <th>Descripción</th>
# <th>Unidad</th>
# <th>Aclaración</th>
# </tr>
# </thead>
# <tbody>
# <tr>
# <td>Pregnancies</td>
# <td>Expresa el número de embarazos</td>
# <td>Número natural</td>
# </tr>
# <tr>
# <td>Glucose</td>
# <td>Expresa el nivel de glucosa en sangre</td>
# <td>$mg/dL$</td>
# <td>Plasma glucose concentration at 2 hours in an oral glucose tolerance test</td>
# </tr>
# <tr>
# <td>BloodPressure</td>
# <td>Expresa la medición de la presión arterial</td>
# <td>$mm$ $Hg$</td>
# <td>Diastolic blood pressure</td>
# </tr>
# <tr>
# <td>SkinThickness</td>
# <td>Expresa el grosor de la piel</td>
# <td>$mm$</td>
# <td>Triceps skin fold thickness</td>
# </tr>
# <tr>
# <td>Insulin</td>
# <td>Exoresa el nivel de insulina en sangre</td>
# <td>$\mu U/ml$</td>
# <td>2-Hour serum insulin</td>
# </tr>
# <tr>
# <td>BMI</td>
# <td>Expresa el índice de masa corporal</td>
# <td>$kg/m^2$</td>
# <td>$weight/height^2$</td>
# </tr>
# <tr>
# <td>DiabetesPedigreeFunction</td>
# <td>Expresa el porcentaje de riesgo para la diabetes (basada en historial familiar)</td>
# </tr>
# <tr>
# <td>Age</td>
# <td>Expresa la edad</td>
# <td>Número natural</td>
# </tr>
# <tr>
# <td>Outcome</td>
# <td>Expresa la presencia o no de diabetes</td>
# <td>Únicos valores: 0 o 1</td>
# <td>1 si el paciente tiene diabetes, 0 sino</td>
# </tr>
# </tbody>
# </table>

# Todos los pacientes son mujeres de al menos 21 años de herencia Pima Indian.

# In[4]:


# Primeras 5 filas
df.head(5)


# Hay valores muy grandes y muy chicos, se va a necesitar un reescalado de los mismos para algunos modelos.

# In[5]:


# Summary de los datos
df.describe()


# ## Hay valores que dan 0 y que no respresentan nada.

# In[6]:


# Porcentaje de la cantidad de valores 0 por columna
df.isin([0]).sum()/len(df)*100


# Bastantes ceros en algunas columnas que no significa nada tenerlos. Se evaluarán más adelante medidas a tomar, por ahora se los categorizará como NaN.

# In[7]:


df[['SkinThickness','Insulin', 'BloodPressure', 'Glucose', 'BMI']] = df[['SkinThickness','Insulin',
                                                                         'BloodPressure', 'Glucose', 'BMI']].replace(0, np.nan)


# In[8]:


# Matriz de correlación
sns.heatmap(df.corr(), annot=True)
plt.show()


# Como es de esperarse, la insulina, la glucosa y el BMI son relevantes en el resultado de la diabetes. Además se perciben correlaciones considerables entre la insulina y la glucosa, SkinThickness y BMI; y edad y embarazos.

# El espesor de la piel y la insulina presentan muchos valores iguales a 0, ahora categorizados como NaN, (29% y 48%, respectivamente). Considerando la importancia de estos datos se decide aplicar el método de MICE para "parchear" esos datos (más adelante se aplica). El resto de columnas se decide hacer un análisis para reemplazar estos valores por una medida de centralidad.

# In[9]:


# Reemplazo de valores NaN de la Glucosa, BloodPressure y BMI
columna = ['Glucose', 'BloodPressure', 'BMI']
medias, medianas = [[] for _ in range(2)]

fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(wspace=0.5)

# Histogramas para decidir si la mediana, la media o la moda representa a la muestra
for j in range(3):
  plt.subplot(1, 3, j+1)
  plt.hist(df[columna[j]], bins='auto', color='indigo')

  # Medidas de centralidad
  media = df[columna[j]].mean()
  mediana = df[columna[j]].median()
  # Se agregan a sus respectivas listas
  medias.append(media)
  medianas.append(mediana)

  # Ploteo
  plt.xlabel(columna[j])
  plt.ylabel('Frecuencia')
  plt.axvline(x = media, color = 'gold', linestyle='--', linewidth=2)
  plt.axvline(x = mediana, color = 'r', linestyle='--', linewidth=2)

# Leyenda
fig.legend(bbox_to_anchor=(1.1, 1.05), labels=['Media','Mediana'])
plt.tight_layout()
plt.show()


# En base a esto se propone cambiar los valores NaN por:
# <table>
# <thead>
# <tr>
# <th>Columna</th>
# <th>Reemplazo por</th>
# </tr>
# </thead>
# <tbody>
# <tr>
# <td>Glucose</td>
# <td>Mediana</td>
# </tr>
# <tr>
# <td>BloodPressure</td>
# <td>Mediana</td>
# </tr>
# <tr>
# <td>BMI</td>
# <td>Mediana</td>
# </tr>
# </tbody>
# </table>

# In[10]:


# Reemplazo de dichos valores
reemplazos = {'Glucose': medianas[0], 'BloodPressure': medianas[1],'BMI': medianas[2]}
df.replace(np.nan, reemplazos, inplace=True)


# In[11]:


df.describe()


# Veamos los boxplots en búsqueda de ver si no hay cosas raras.

# In[12]:


plt.figure(figsize=(10, 8))
for i, column in enumerate(df.columns[:-1]):
    plt.subplot(4, 2, i + 1)
    sns.boxplot(data=df, x=column, hue='Outcome')
    plt.title(f'Boxplot de {column}')
plt.tight_layout()
plt.show()


# Se considera que una SkinThickness de casi 100mm no es posible y valores de insulina arriba de 600 son poco probables. Por ello se decide eliminar estos datos.

# In[13]:


df.drop(df[(df['Insulin'] > 600) | (df['SkinThickness'] > 90)].index, inplace=True)


# La decisión de no seguir sacando posibles outliers viene dada por considerar que hace falta información del tipo y técnicas del muestro. Así mismo como una comprensión más profunda (con un experto en el área de la medicina) de los tipos de datos para poder evaluar qué es outlier y qué no.

# ## Porcentaje de diabéticos

# In[14]:


plt.pie(df['Outcome'].value_counts(), labels=['no diabéticos','diabéticos'], autopct='%1.2f%%', colors=['orchid', 'gold'])
plt.title(label="Porcentaje de diabéticos del dataset")
plt.show()


# Queremos un modelo que supere ampliamente el 65% de accuracy, sino sería un modelo que no aporta nada.

# ## ¿La glucosa tiene una distribución normal?

# In[15]:


# Veamos un histograma y boxplot
fig = plt.figure()
fig.subplots_adjust(hspace=0.3, wspace=0.5)

plt.subplot(1, 2, 1)
plt.hist(df['Glucose'], bins='auto', density=True)
x_axis = np.arange(df.Glucose.min(), df.Glucose.max(), 0.1)
plt.plot(x_axis, norm.pdf(x_axis, df.Glucose.mean(), df.Glucose.std()), linewidth=2, color='r')
plt.xlabel('Glucosa')
plt.ylabel('Frecuencia')

plt.subplot(1, 2, 2)
plt.boxplot(df['Glucose'])
plt.ylabel('Glucosa')

plt.show()


# El histograma da con una cola más pesada a izquierda (cae más rápidamente). <br>
# 
# Por otro lado, el boxplot da un resultado un poco asimétrico, la mediana no queda tan centrado como debería.
# Pros: los brazos son parecidos.

# In[16]:


# Haré un qqplot
print(sm.qqplot(df['Glucose'], line='q'))


# Efectivamente, se tiene una cola pesada a izquierda y liviana a derecha (aunque abrupta). Podríamos concluir que la glucosa no tiene una distribución normal.

# Veamos un test de hipótesis para corroborar que no tiene una distribución normal.

# In[17]:


# Shapiro-Wilk test para ver si corresponde a una distribución normal
shapiro(df['Glucose'])


# El p-valor es muy chiquito, para cualquier nivel de significación aceptable se rechazaría la hipótesis nula de que la distrubición de la glucosa es normal.

# Sin embargo, separando por diabéticos y no diabéticos se tiene una distribución más normal:

# In[18]:


sns.histplot(data=df,x='Glucose',hue='Outcome')
plt.show()


# In[19]:


# Test de normalidad
print('test de Shapiro en las personas NO diabéticas: \n p-valor =', shapiro(df.loc[df["Outcome"] == 0, "Glucose"])[1])
print('test de Shapiro en las personas diabéticas: \n p-valor =', shapiro(df.loc[df["Outcome"] == 1, "Glucose"])[1])


# Volviendo a hacer el Shapiro-Wilk test obtenemos un resultado mejor pero aún así no se tiene normalidad para un nivel de significación razonable.

# ##¿Cuál es el máximo número de embarazos?

# In[20]:


df['Pregnancies'].max()


# ¡Una mujer tuvo 17 embarazos! ¿Es posible? Veamos su edad:

# In[21]:


df[df['Pregnancies']==df['Pregnancies'].max()]


# La edad no nos descarta la posibilidad de 17 embarazos. Se decide mantener el dato.

# ## ¿La presión arterial tiene una distribución normal?

# In[22]:


# Veamos un histograma
plt.subplot(1, 2, 1)
plt.hist(df['BloodPressure'], bins='auto', density=True)

# Ploteo de una normal usando la media y desvío de la BloodPressure
x_axis = np.arange(df.BloodPressure.min(), df.BloodPressure.max(), 0.1)
plt.plot(x_axis, norm.pdf(x_axis, df.BloodPressure.mean(), df.BloodPressure.std()), linewidth=2, color='r')

# Etiquetas
plt.xlabel('Presión arterial')
plt.ylabel('Frecuencia')

# Boxplot
plt.subplot(1, 2, 2)
plt.boxplot(df['BloodPressure'])
plt.ylabel('Presión arterial')
plt.tight_layout()
plt.show()


# Tiene un compartamiento nada parecido a la de una normal, con muchos datos atípicos.

# In[23]:


# Rango intercuartil
IQR = stats.iqr(df['BloodPressure'])
print(IQR)


# Comrpobando con el test de Shapiro-Wilk:

# In[24]:


# Test de normalidad
print(shapiro(df['BloodPressure']))


# Se concluye que los datos no vienen de una distribución normal.

# ## Dispersión de los datos para descubrir patrones

# Veamos la dispersión de los datos separados por diabéticos y no diabéticos:

# In[25]:


sns.pairplot(df, hue='Outcome')
plt.tight_layout()
plt.show()


# Se ve la tendencia clara que a mayor nivel de Glucosa más tendencia a ser diabético. Con el BMI y la Glucosa los sets de diabéticos y no diabéticos se notan un poco más definidos.

# # Última etapa del preprocesamiento

# In[26]:


train_data = df.copy()
train_label = train_data.pop('Outcome')


# Mantendremos la proporcionalidad (de diabético y no diabéticos) en los distintos conjuntos. Por eso se usa stratify.

# In[27]:


# Se separa los datos en 3 pares de sets: training, validation data y test
X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, train_size = .8, shuffle=True,
                                                    random_state=seed, stratify=train_label)

X_train, X_val, y_train, y_val = train_test_split(X_train,  y_train, train_size = .8, shuffle=True,
                                                 random_state=seed, stratify=y_train)

training_complete = pd.concat([X_train, y_train], axis=1)


# ## Método Mice al conjunto de entrenamiento

# Se lo va a aplicar sólo al conjunto de entrenamiento, en el resto de conjuntos se van a eliminar los datos NaN.

# In[28]:


# Eliminación de datos NaN en los conjuntos de testeo y validación
na_indices_test = X_test[X_test.isna().any(axis=1)].index
X_test.dropna(inplace=True)
y_test.drop(index=na_indices_test, inplace=True)

na_indices_val = X_val[X_val.isna().any(axis=1)].index
X_val.dropna(inplace=True)
y_val.drop(index=na_indices_val, inplace=True)


# In[29]:


# Aplicar IterativeImputer (MICE)
imputer = IterativeImputer(max_iter=10, random_state=seed)
imputed_data = imputer.fit_transform(training_complete)
training_complete = pd.DataFrame(imputed_data, columns=training_complete.columns)


# Chequeemos que los datos fueron asignados acorde.

# In[30]:


X_train.describe()


# In[31]:


plt.figure(figsize=(10, 8))
for i, column in enumerate(X_train.columns):
    plt.subplot(4, 2, i + 1)
    sns.boxplot(data=training_complete, x=column, hue='Outcome')
    plt.title(f'Boxplot de {column}')
plt.tight_layout()
plt.show()


# In[32]:


y_train_sin_oversampling = training_complete.pop('Outcome')
X_train_sin_oversampling = training_complete


# ## Oversampling al conjunto de entrenamiento

# Tratemos de equilibrar las clases de diabéticos y no diabéticos. Para ello haremos un bootstrap en la clase de diabéticos SOLO para el conjunto de entrenamiento. Habrá más riesgo de sobrefiteo.

# In[33]:


ros = RandomOverSampler(random_state=seed)
X_train, y_train = ros.fit_resample(X_train_sin_oversampling, y_train_sin_oversampling)


# ## Disminución de la dimensionalidad en busca de patrones (sólo para visualizar)

# In[34]:


# Aplicar PCA para reducir a 2 dimensiones
pca_2d = PCA(n_components=2)
components_2d = pca_2d.fit_transform(X_train)

# Visualización
plt.figure(figsize=(8, 6))
outcome_texto = {0: 'no diabético', 1:'diabético'}
for outcome in y_train.unique():
    subset = components_2d[y_train == outcome]
    plt.scatter(subset[:, 0], subset[:, 1], label=outcome_texto[outcome], s=50, alpha=0.4)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('PCA 2d')
plt.legend(title='Outcome')
print('La varianza explicada es de:', pca_2d.explained_variance_ratio_)
plt.show()


# Muy buena varianza explicada, arriba del 96%. Se puede llegar a ver secciones en las que predominan mayormente los no diabéticos.

# In[35]:


# Escalar los datos
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X_train)

# Aplicar t-SNE
tsne = TSNE(n_components=2, random_state=seed)
components_tsne = tsne.fit_transform(scaled_data)

# Visualización
plt.figure(figsize=(8, 6))
outcome_texto = {0: 'no diabético', 1:'diabético'}
for outcome in y_train.unique():
    subset = components_tsne[y_train == outcome]
    plt.scatter(subset[:, 0], subset[:, 1], label=outcome_texto[outcome], s=50)
plt.xlabel('Dimensión 1')
plt.ylabel('Dimensión 2')
plt.title('t-SNE 2d')
plt.legend(title='Outcome')
plt.grid()
plt.show()


# t-SNE reduce dimensionalidad mientras trata de mantener distancias cercanas entre los datos que se aproximan y distancias más largas entre los datos que no se aproximan.<br>
# Se ven los sectores de diabéticos y no diabéticos más diferenciados así, principalmente el sector no diabético.

# Aclaración: es preferible tener mayor recall (poder clasificar bien a la gente que está enferma) por ello se ajustará al threshold adecuado para cada modelo.

# # Usando redes neuronales

# ## Reescalado de valores

# In[36]:


Scaler = StandardScaler()

Scaler.fit(X_train)
X_train_escalado = Scaler.transform(X_train)
X_val_escalado = Scaler.transform(X_val)
X_test_escalado = Scaler.transform(X_test)


# ## Búsqueda en grilla para hallar el mejor modelo de red neuronal

# In[37]:


custom_threshold = 0.5
def model_builder(hp):
  # Número de neuronas en la primera y segunda capa
  hp_units_1 = hp.Int('units_1', min_value=32, max_value=152, step=32)
  hp_units_2 = hp.Int('units_2', min_value=32, max_value=152, step=32)

  model = keras.Sequential([
      keras.layers.Dense(hp_units_1, input_shape=[X_train.shape[1],], activation='relu'),
      keras.layers.Dropout(0.2),
      # Para normalizar las activaciones
      keras.layers.BatchNormalization(),
      keras.layers.Dense(hp_units_2, activation='relu'),
      keras.layers.Dropout(0.2),
      keras.layers.BatchNormalization(),
      keras.layers.Dense(1, activation='sigmoid')
  ])

  # Variación del learning rate
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss='binary_crossentropy',
                metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy", dtype=None, threshold=custom_threshold),
                         tf.keras.metrics.Precision(thresholds=custom_threshold, name='precision'),
                         tf.keras.metrics.Recall(thresholds=custom_threshold, name='recall'),
                         tf.keras.metrics.AUC(name='auc')]
              )

  return model


# In[38]:


tuner = kt.Hyperband(model_builder,
                     objective=kt.Objective("val_auc", direction="max"),
                     max_epochs=10,
                     factor=3,
                     seed=seed)

# Vamos a ponerle un stop para no sobre-entrenar, que pare cuando la val_loss no siga bajando
patience = 5
stop_early = tf.keras.callbacks.EarlyStopping(
    start_from_epoch=2, # dejar correr al menos 2 épocas
    min_delta=0.001, # mínima cantidad que cuenta como mejora
    patience=patience, # épocas antes de parar
    restore_best_weights=True # se queda con los mejores pesos antes de no decrecer más
)


# In[39]:


# Búsqueda de los mejores hiperparámetros
tuner.search(X_train_escalado, y_train, epochs=50, validation_data=(X_val_escalado, y_val), callbacks=[stop_early])

# Hiperparámetros óptimos
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
El número óptimo de neuronas en la primera capa es de {best_hps.get('units_1')}.
El número óptimo de neuronas en la segunda capa es de {best_hps.get('units_2')}.
El learning rate óptimo es {best_hps.get('learning_rate')}.
""")


# ## Entreno la mejor red

# In[40]:


# Esta clase servirá para el modelo de Ensemble, por eso se aplica ahora
class TFNeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, input_shape, threshold=0.5):
        self.input_shape = input_shape
        self.model = self._build_model()
        self.threshold = threshold
        self.history = None
        self.callbacks = [stop_early]

    def _build_model(self):
        model = tuner.hypermodel.build(best_hps)
        return model

    def fit(self, X, y, batch_size=200, epochs=80, verbose=0, *args, **kwargs):
        self.history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=self.callbacks, *args, **kwargs)
        return self.history

    def predict(self, X):
        pred_probs = self.model.predict(X, verbose=0)
        return (pred_probs > self.threshold).astype("int32")

    def predict_proba(self, X):
        pred_probs = self.model.predict(X, verbose=0)
        return np.hstack([1 - pred_probs, pred_probs])


# In[41]:


model_red_neuronal = TFNeuralNetwork(input_shape=(X_train.shape[1],), threshold=custom_threshold)


# In[42]:


# Entreno el modelo, ya tiene el callback asignado
history = model_red_neuronal.fit(X_train_escalado, y_train, batch_size=200, epochs=70,
                                 validation_data=(X_val_escalado, y_val), verbose=1)


# In[43]:


# Ploteo de la loss y el auc
history_df = pd.DataFrame(history.history)[:-patience]
history_df.loc[:, ['loss', 'val_loss']].plot()
plt.title('Loss del modelo de Redes Neuronales')
plt.xlabel('Época')
plt.ylabel('Valor')
plt.ylim((0,2))
history_df.loc[:, ['auc', 'val_auc']].plot()
plt.title('AUC del modelo de Redes Neuronales')
plt.xlabel('Época')
plt.ylabel('Valor')

print(("Última Validation Loss: {:0.4f} \n" +\
      "Última Validation Accuracy: {:0.4f} \n" +\
      "Última Validation Precision: {:0.4f} \n" +\
      "Última Validation Recall: {:0.4f}") \
      .format(history_df["val_loss"].iloc[-1],
              history_df["val_accuracy"].iloc[-1],
              history_df["val_precision"].iloc[-1],
              history_df["val_recall"].iloc[-1]))


# No parece haber overfitting ni underfitting.

# ## Elección del threshold

# In[44]:


# Ploteo de la recall y precisión en base al threshold
prob = model_red_neuronal.predict_proba(X_val_escalado)[:,1]
precision, recall, thresholds = precision_recall_curve(y_val, prob)  # Se le pueden pasar probabilidades
plt.plot(thresholds, precision[:-1], label='precisión')
plt.plot(thresholds, recall[:-1], label='recall')
plt.xlabel('Threshold')
plt.legend()
plt.grid()
plt.yticks(np.linspace(0, 1, num=21))
plt.title('Recall y Precisión del modelo de Redes Neuronales')
plt.show()


# Viendo el gráfico hay un punto que tiene una buena recall (de 0.9) y buena precision (alrededor de 0.7). Averiguemos qué punto es:

# In[45]:


# Se define una recall mínima que se quiere y de ahí se saca el threshold
recall_min = 0.9
custom_threshold = thresholds[np.argmin(recall >= recall_min)]
print(f'El threshold elegido es de: {custom_threshold:.2f} \n')


# ## Curva ROC

# In[46]:


fper, tper, thresholds = roc_curve(y_val, prob)  # Se le pueden pasar probabilidades
roc_auc = roc_auc_score(y_val, prob)
plt.plot(fper, tper, marker='.', label='Curva ROC redes neuronal',color='g')
plt.plot(fper[np.argmax(thresholds < custom_threshold)],
         tper[np.argmax(thresholds < custom_threshold)], marker='o', markersize=7,
         color='r', label='Punto aproximado del threshold elegido')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.legend()
plt.grid()
print('El área de la curva ROC es',roc_auc)
plt.title('Curva ROC del modelo de Redes Nueronales')
plt.show()


# ## Evaluación del modelo en el conjunto de testeo

# In[47]:


# Realizar predicciones en los datos de prueba
pred_probabilities = model_red_neuronal.predict_proba(X_test_escalado)[:,1]

# Convertir probabilidades en etiquetas utilizando el umbral personalizado
pred_labels_custom_threshold = (pred_probabilities >= custom_threshold).astype(int)

print('Métricas en el conjunto de testeo:\n', classification_report(y_test, pred_labels_custom_threshold))


# Se obtuvo buena recall a costa de una baja en la precisión (en la clase diabética). Buena recall y precisión en la clase no diabética.

# ## Veamos la matriz de confusión

# In[48]:


predicted = model_red_neuronal.predict_proba(X_test_escalado)[:,1]
predicted = (predicted > custom_threshold).astype('int')
cm = confusion_matrix(y_test, predicted)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])
cm_display.plot()
plt.title('Matriz de confunsión del modelo de Redes Neuronales')
plt.show()


# # Usando Boosting

# In[49]:


# Definir el grid para búsqueda por grilla
param_grid = {
    'n_estimators': [4, 10, 15],  # Número de árboles en el ensamble
    'max_depth': [2, 3],        # Profundidad máxima de cada árbol
    'learning_rate': [0.2, 0.3, 0.35, 0.4],  # Tasa de aprendizaje
    'gamma': [1, 1.5, 2, 2.5],          # Parámetro de regularización que controla la complejidad del árbol
    'reg_alpha': [0.3, 0.5, 1, 2],        # Término de regularización L1 (Lasso)
    'reg_lambda': [0.15, 0.2, 0.3, 0.5],       # Término de regularización L2 (Ridge)
}

# Configurar GridSearchCV
grid_search_boosting = GridSearchCV(estimator=XGBClassifier(random_state=seed, n_jobs=-1),
                           param_grid=param_grid, scoring='roc_auc', cv=7, n_jobs=-1)

# Entrenar GridSearchCV
grid_search_boosting.fit(X_train, y_train,
                         eval_set=[(X_val, y_val)], early_stopping_rounds=2, eval_metric='auc', verbose=False)

# Obtener los mejores hiperparámetros
best_params = grid_search_boosting.best_params_
print(f'Mejores parámetros: {best_params}')


# In[50]:


# Lista de evaluación, primero se debe poner la que monitoriará el stopping
eval_set = [(X_val, y_val), (X_train, y_train)]

# Crear el modelo
model_boosting = XGBClassifier(**best_params, random_state=seed, n_jobs=-1)
# Entrenar el mejor modelo
model_boosting.fit(X_train, y_train, eval_metric=['logloss', 'auc'], eval_set=eval_set, early_stopping_rounds=2, verbose=0)


# ## Evaluación de overfitting o underfitting

# In[51]:


# Obtener los resultados de evaluación
evals_result = model_boosting.evals_result()

# Plotear los resultados
iteraciones = len(evals_result['validation_0']['logloss'])
x_axis = range(0, iteraciones)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Plotear Log Loss
ax1.plot(x_axis, evals_result['validation_1']['logloss'], label='Entrenamiento')
ax1.plot(x_axis, evals_result['validation_0']['logloss'], label='Validación')
ax1.set_xlabel('Iteración')
ax1.set_ylabel('Log Loss')
ax1.legend()
ax1.set_xticks(x_axis)

# Plotear Error
ax2.plot(x_axis, evals_result['validation_1']['auc'], label='Entrenamiento')
ax2.plot(x_axis, evals_result['validation_0']['auc'], label='Validación')
ax2.set_xlabel('Iteración')
ax2.set_ylabel('AUC')
ax2.legend()
ax2.set_xticks(x_axis)

# Título general para la figura
fig.suptitle('Log Loss y AUC del modelo boosting')

fig.tight_layout()
# Mostrar la figura
plt.show()


# Hay un poco de sobrefiteo para el final, por eso se eligirá la iteración óptima.

# In[52]:


# Obtener el número de iteración óptima
iteracion_deseada = 7

# Restaurar el modelo a esa iteración
model_boosting.set_params(n_estimators=iteracion_deseada)


# ## Veamos la influencia de cada feature (columna).

# In[53]:


# Imprimir las importancias de las columnas
for name, score in zip(train_data.columns, model_boosting.feature_importances_):
    print(f"{name}: {score}")


# La insulina toma una importancia destacable frente a las otras, le sigue la Glucosa y la BMI.

# ## Elección del threshold

# In[54]:


y_predict_scores = model_boosting.predict_proba(X_val)[:,1]
precision, recall, thresholds = precision_recall_curve(y_val, y_predict_scores)

# Ploteo recall y precisión
plt.plot(thresholds, precision[:-1], label='precisión')
plt.plot(thresholds, recall[:-1], label='recall')
plt.xlabel('Threshold')
plt.legend()
plt.grid()
plt.yticks(np.linspace(0, 1, num=21))
plt.title('Precisión y Recall del modelo Boosting')
plt.show()


# In[55]:


threshold = thresholds[np.argmin(recall >= 0.95)]
print(f'El threshold elegido es de: {threshold:.2f} \n')


# In[56]:


# Ploteo de la Curva ROC
fper, tper, thresholds = roc_curve(y_val, y_predict_scores)
roc_auc = roc_auc_score(y_val, y_predict_scores)
plt.plot(fper, tper, marker='.', label='Curva ROC Boosting',color='orange')
plt.plot(fper[np.argmax(thresholds < threshold)],
         tper[np.argmax(thresholds < threshold)], marker='o', markersize=7,
         color='r', label='Punto aproximado del threshold elegido')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.legend()
plt.grid()
print('El área de la curva ROC es', roc_auc)
plt.title('Curva ROC del modelo Boosting')
plt.show()


# ## Evaluación del modelo en el conjunto de testeo

# In[57]:


y_predict = model_boosting.predict_proba(X_test)[:,1]

y_predict_con_scores = (y_predict > threshold).astype(int)

print('Métricas en el conjunto de testeo:\n', classification_report(y_test, y_predict_con_scores))


# ## Matriz de Confusión

# In[58]:


cm = confusion_matrix(y_test, y_predict_con_scores)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])
cm_display.plot()
plt.title('Matriz de confusión del modelo Boosting')
plt.show()


# ## Visualización de un árbol

# In[59]:


# Obtener el número total de árboles en el modelo
num_trees = len(model_boosting.get_booster().get_dump())
print(f"El modelo tiene {num_trees} árboles.")


# In[60]:


num_tree = 0

fig, ax = plt.subplots(figsize=(30, 30))
plot_tree(model_boosting, num_trees=num_tree, ax=ax, proportion=True)
plt.title('Primer árbol de decisión del modelo Boosting')
plt.show()


# Las hojas finales son una predicción logística, para pasar a una predicción entre 0 y 1 hay que pasarla por una función, en este caso:
# \begin{equation*}
# f(x) = \frac{1}{1+e^{x}}
# \end{equation*}
# El resultado dará la probabilidad de pertenecer a la clase diabética.

# ### Otra forma de visualizar un árbol en específico

# In[61]:


# Paso necesario para visualizar con dtreeviz
y_train = y_train.astype(int)

viz_model = dtreeviz.model(model_boosting,
                           tree_index=num_tree,
                           X_train=X_train,
                           y_train=y_train,
                           feature_names=Scaler.feature_names_in_.tolist(),
                           target_name='Output',
                           class_names=['No diabetes', 'Diabetes'])


# In[62]:


# Elimina "Arial font not found warnings" que aparece al hacer la view
logging.getLogger('matplotlib.font_manager').setLevel(level=logging.CRITICAL)
viz_model.view(scale=1.2)


# # Usando SVM

# ## Con LinearSVC

# In[63]:


polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures()),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(loss="hinge"))
    ])

# Definir la grilla de hiperparámetros
param_grid = {
    'svm_clf__C': [0.003, 0.005, 0.01, 0.03, 0.06],
    'poly_features__degree': [2, 3, 4, 5]
}
# Configurar GridSearchCV
grid_search = GridSearchCV(polynomial_svm_clf, param_grid=param_grid, scoring='roc_auc', cv=6)

grid_search.fit(X_train, y_train)

# Obtener los mejores hiperparámetros
best_params = grid_search.best_params_
print(f'Best parameters found: {best_params}')

# Mejor modelo
model_SVM = grid_search.best_estimator_


# ### Evaluación de overfitting o underfitting

# In[64]:


# Ploteo de la roc auc con cross validation a medida que el tamaño del conjunto de entrenamiento aumenta
def plot_learning_curve(estimator, X, y, cv=None, train_sizes=np.linspace(.1, 1.0, 5)):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv,
                                                            train_sizes=train_sizes, scoring='roc_auc',
                                                            random_state=seed)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.xlabel("Tamaño del Conjunto de Entrenamiento")
    plt.ylabel("Roc AUC")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="en Entrenamiento")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="en el conjunto de validación")

    plt.legend(loc="best")
    return plt

# Creación del modelo de nuevo para pasarlo por la learning_curve (necesario porque hace un fit)
best_model = Pipeline([
    ("poly_features", PolynomialFeatures(degree=best_params['poly_features__degree'])),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(loss="hinge", C=best_params['svm_clf__C']))
])

plot_learning_curve(best_model, X_train, y_train, cv=6)
plt.title('Curva de la ROC AUC con cross validation del modelo SVM')
plt.show()


# Se ven cómo las curvas se van acercando a medida que el tamño del conjunto de entrenamiento crece, esto es una buena señal. También los valores de AUC dan bien por lo que no habría underfitting.<br>
# El modelo no está sobrefiteando.

# ### Veamos qué threshold elegir

# In[65]:


y_predict_scores = model_SVM.decision_function(X_val)
precision, recall, thresholds = precision_recall_curve(y_val, y_predict_scores)

# Ploteo de la recall y precisión
plt.plot(thresholds, precision[:-1], label='precisión')
plt.plot(thresholds, recall[:-1], label='recall')
plt.xlabel('Threshold')
plt.legend()
plt.grid()
plt.yticks(np.linspace(0, 1, num=21))
plt.title('Precisión y Recall del modelo SVM')
plt.show()


# In[66]:


threshold = thresholds[np.argmin(recall>= 0.85)]
print(f'El threshold elegido es de: {threshold:.3f} \n')


# ### Curva ROC

# In[67]:


fper, tper, thresholds = roc_curve(y_val, y_predict_scores)
roc_auc = roc_auc_score(y_val, y_predict_scores)

# Ploteo de la Curva ROC
plt.plot(fper, tper, marker='.', label='Curva ROC',color='g')
plt.plot(fper[np.argmax(thresholds < threshold)],
         tper[np.argmax(thresholds < threshold)], marker='o', markersize=7,
         color='r', label='Punto aproximado del threshold elegido')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.legend()
plt.grid()
print('El área de la curva ROC es', roc_auc)
plt.title('Curva ROC del modelo SVM lineal')
plt.show()


# ### Evaluación del modelo en el conjunto de testeo

# In[68]:


y_predict_scores = model_SVM.decision_function(X_test)
y_predict_con_scores = (y_predict_scores > threshold).astype(int)
print('Métricas en el conjunto de testeo:\n', classification_report(y_test, y_predict_con_scores))


# Buena recall en la clase diabética. Buena recall y precisión en la clase no diabética.

# ## Con kernel RBF

# In[69]:


# tubería que incluye el escalado de características y el SVM con kernel RBF
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', probability=True))
])

# Definir un rango de hiperparámetros para búsqueda en cuadrícula
param_grid = {
    'svm__C': [0.1, 1, 5, 10, 15],
    'svm__gamma': [0, 0.1, 0.01, 1e-3, 1e-4]
}

# Configurar GridSearchCV para encontrar los mejores hiperparámetros
grid_search_svm_rbf = GridSearchCV(pipeline, param_grid, cv=6, scoring='roc_auc', n_jobs=-1)

# Ajustar el modelo
grid_search_svm_rbf.fit(X_train, y_train)

best_params = grid_search_svm_rbf.best_params_
# Imprimir los mejores hiperparámetros encontrados
print(f"Mejores parámetros: {best_params}")


# In[70]:


# Mejor modelo
model_SVM_rbf = grid_search_svm_rbf.best_estimator_


# ### Evaluación de overfitting o underfitting

# In[71]:


# Creación del modelo de nuevo para pasarlo por la learning_curve (necesario porque hace un fit)
pipeline_best = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', probability=True, C=best_params['svm__C'], gamma=best_params['svm__gamma']))
])

plot_learning_curve(pipeline_best, X_train, y_train, cv=6)
plt.title('Curva de la ROC AUC con cross validation del modelo SVM con rbf')
plt.show()


# Las curvas se acercan lo cual indica que no hay sobrefiteo. También los valores de AUC dan bien por lo que no habría underfitting. <br>
# El modelo está bien.

# ### Veamos qué threshold elegir

# In[72]:


y_predict_scores = model_SVM_rbf.predict_proba(X_val)[:,1]
precision, recall, thresholds = precision_recall_curve(y_val, y_predict_scores)

# Ploteo de la recall y precisión
plt.plot(thresholds, precision[:-1], label='precisión')
plt.plot(thresholds, recall[:-1], label='recall')
plt.xlabel('Threshold')
plt.legend()
plt.grid()
plt.yticks(np.linspace(0, 1, num=21))
plt.title('Precisión y Recall del modelo SVM con rbf')
plt.show()


# In[73]:


threshold = thresholds[np.argmin(recall>= 0.9)]
print(f'El threshold elegido es de: {threshold:.3f} \n')


# ### Curva ROC

# In[74]:


fper, tper, thresholds = roc_curve(y_val, y_predict_scores)
roc_auc = roc_auc_score(y_val, y_predict_scores)

# Ploteo de la Curva ROC
plt.plot(fper, tper, marker='.', label='Curva ROC',color='g')
plt.plot(fper[np.argmax(thresholds < threshold)],
         tper[np.argmax(thresholds < threshold)], marker='o', markersize=7,
         color='r', label='Punto aproximado del threshold elegido')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.legend()
plt.grid()
print('El área de la curva ROC es', roc_auc)
plt.title('Curva ROC del modelo SVM con rbf')
plt.show()


# ### Evaluación del modelo en el conjunto de testeo

# In[75]:


y_predict_scores = model_SVM_rbf.predict_proba(X_test)[:,1]
y_predict_con_scores = (y_predict_scores > threshold).astype(int)
print('Métricas en el conjunto de testeo:\n', classification_report(y_test, y_predict_con_scores))


# Buena recall (en la clase de diabéticas) a costo de que baja la precisión.

# # Con regresión logística

# In[76]:


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', LogisticRegression(random_state=seed))
  ])

# Definir el espacio de búsqueda
param_grid = {
    'svm__C': [0.003, 0.005, 0.01, 0.1, 1],
    'svm__penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'svm__solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
    'svm__max_iter': [4, 5, 6, 8, 10]
}

# Configurar GridSearchCV para encontrar los mejores hiperparámetros
grid_search_logistic_reg = GridSearchCV(pipeline, param_grid, cv=8, scoring='roc_auc', n_jobs=-1)

# Ajustar el modelo
grid_search_logistic_reg.fit(X_train, y_train)

best_params = grid_search_logistic_reg.best_params_

# Imprimir los mejores hiperparámetros encontrados
print(f"Mejores parámetros: {best_params}")


# In[77]:


# Mejor modelo
model_regresion_logistica = grid_search_logistic_reg.best_estimator_


# ## Ver si hay underfitting o overfitting

# In[78]:


# Creación del modelo de nuevo para pasarlo por la learning_curve (necesario porque hace un fit)
best_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', LogisticRegression(random_state=seed, C=best_params['svm__C'], penalty=best_params['svm__penalty'],
                               solver=best_params['svm__solver'], max_iter=best_params['svm__max_iter']))
  ])

plot_learning_curve(best_pipeline, X_train, y_train, cv=6)
plt.title('Curva de la ROC AUC con cross validation del modelo Regresión Logística')
plt.show()


# Las curvas se acercan y buena valor de AUC. No hay overfitting ni underfitting.

# ## Hagamos una k-fold Cross Validation para ver la performance del modelo

# In[79]:


# Creación del modelo de nuevo para pasarlo por cross validation (necesario porque hace un fit)
best_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', LogisticRegression(random_state=seed, C=best_params['svm__C'], penalty=best_params['svm__penalty'],
                               solver=best_params['svm__solver'], max_iter=best_params['svm__max_iter']))
  ])

n_splits = 6
scoring = ['accuracy', 'f1', 'precision', 'recall']
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
scores = cross_validate(best_pipeline, X_train,
                        y_train, cv=kfold, scoring=scoring)

for j in range(n_splits):
  print('Iteración %i: Accuracy %.2f, f1_score %.2f, precisión %.2f y recall %.2f' % (j+1, scores['test_accuracy'][j],
                                                                                      scores['test_f1'][j],
                                                                                      scores['test_precision'][j],
                                                                                      scores['test_recall'][j]))
print("""Medias del promedio (desvío):
  Accuracy %.2f%% (%.2f%%),
  f1_score %.2f%% (%.2f%%),
  precisión %.2f%% (%.2f%%) y
  recall %.2f%% (%.2f%%)\n""" % (
      scores['test_accuracy'].mean()*100, scores['test_accuracy'].std()*100,
      scores['test_f1'].mean()*100, scores['test_f1'].std()*100,
      scores['test_precision'].mean()*100, scores['test_precision'].std()*100 ,
      scores['test_recall'].mean()*100, scores['test_recall'].std()*100))


# ## Veamos qué threshold elegir

# In[80]:


y_predict_scores = model_regresion_logistica.predict_proba(X_val)[:,1]
precision, recall, thresholds = precision_recall_curve(y_val, y_predict_scores)

# Ploteo de la recall y precisión
plt.plot(thresholds, precision[:-1], label='precisión')
plt.plot(thresholds, recall[:-1], label='recall')
plt.xlabel('Threshold')
plt.legend()
plt.grid()
plt.yticks(np.linspace(0, 1, num=21))
plt.title('Precisión y Recall del modelo Regresión Logística')
plt.show()


# In[81]:


threshold = thresholds[np.argmin(recall>= 0.9)]
print(f'El threshold elegido es de: {threshold:.2f} \n')


# ## Curva ROC

# In[82]:


fper, tper, thresholds = roc_curve(y_val, y_predict_scores)
roc_auc = roc_auc_score(y_val, y_predict_scores)

# Ploteo de la curva ROC
plt.plot(fper, tper, marker='.', label='Curva ROC Regresión Logística', color='orange')
plt.plot(fper[np.argmax(thresholds < threshold)],
         tper[np.argmax(thresholds < threshold)], marker='o', markersize=7,
         color='r', label='Punto aproximado del threshold elegido')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.legend()
plt.grid()
print('El área de la curva ROC es', roc_auc)
plt.title('Curva ROC del modelo Regresión Logística')
plt.show()


# ## Evaluación del modelo en el conjunto de testeo

# In[83]:


y_predict_scores = model_regresion_logistica.predict_proba(X_test)[:,1]
y_predict_con_scores = (y_predict_scores > threshold).astype(int)
print('Métricas en el conjunto de testeo:\n', classification_report(y_test, y_predict_con_scores))


# Buena recall y precisión en ambas clases.

# # Con Ensamble (Votting Classifier)

# In[84]:


# pipeline para la red neuronal
pipeline_nn = Pipeline([
    ('scaler', StandardScaler()),
    ('nn', model_red_neuronal)
])


# In[85]:


# Según se recomienda, en un ensamble de modelos se recomienda aplicar tantos como clases haya (en este caso 2)
voting_clf = VotingClassifier(estimators=[
        #('nn_clf', pipeline_nn),
        #('svm_rbf', model_SVM_rbf),
        ('xgb', model_boosting),
        ('log', model_regresion_logistica)
    ], voting='soft')

voting_clf.fit(X_train, y_train)


# ## Evaluación si hay overfitting o underfitting

# In[86]:


# Creación del modelo de nuevo para pasarlo por la learning_curve (necesario porque hace un fit)
voting = VotingClassifier(estimators=[
        #('nn_clf', pipeline_nn),
        #('svm_rbf', model_SVM_rbf),
        ('xgb', model_boosting),
        ('log', model_regresion_logistica)
    ], voting='soft')

plot_learning_curve(voting, X_train, y_train, cv=6)
plt.title('Curva de la ROC AUC con cross validation del modelo Voting')
plt.show()


# Las curvas se acercan y se obtiene un buen valor de AUC. El modelo está bien.

# ## Evaluación de métricas y elección de threshold

# In[87]:


# Creación del modelo de nuevo para pasarlo por una cross validation (necesario porque hace un fit)
voting = VotingClassifier(estimators=[
        #('nn_clf', pipeline_nn),
        #('svm_rbf', model_SVM_rbf),
        ('xgb', model_boosting),
        ('log', model_regresion_logistica)
    ], voting='soft')

scoring = ['roc_auc', 'accuracy', 'recall', 'precision']
cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=seed)
scores = cross_validate(voting , X_train, y_train, scoring=scoring, cv=cv, verbose=0)
print(f'ROC AUC: {scores["test_roc_auc"].mean():.3f} +/- {scores["test_roc_auc"].std():.3f}')
print(f'Accuracy: {scores["test_accuracy"].mean():.3f} +/- {scores["test_accuracy"].std():.3f}')
print(f'Recall: {scores["test_recall"].mean():.3f} +/- {scores["test_recall"].std():.3f}')
print(f'Precision: {scores["test_precision"].mean():.3f} +/- {scores["test_precision"].std():.3f}')


# In[88]:


prob = voting_clf.predict_proba(X_val)[:,1]
precision, recall, thresholds = precision_recall_curve(y_val, prob)

# Ploteo de la recall y precisión
plt.plot(thresholds, precision[:-1], label='precisión')
plt.plot(thresholds, recall[:-1], label='recall')
plt.xlabel('Threshold')
plt.legend()
plt.grid()
plt.yticks(np.linspace(0, 1, num=21))
plt.title('Precisión y Recall del modelo Voting')
plt.show()


# In[89]:


# Threshold elegido
threshold = thresholds[np.argmin(recall >= 0.9)]
print(f'El threshold elegido es de: {threshold:.2f} \n')


# ## Curva ROC

# In[90]:


# Calcular la tasa de verdaderos positivos (TPR) y la tasa de falsos positivos (FPR)
fpr, tpr, thresholds = roc_curve(y_val, prob)

# Calcular el área bajo la curva ROC (AUC)
roc_auc = roc_auc_score(y_val, prob)

# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# ## Evaluación en el conjunto de testeo

# In[91]:


predictions = (voting_clf.predict_proba(X_test)[:,1] > threshold).astype(int)
print('Métricas en el conjunto de testeo:\n', classification_report(y_test, predictions))


# Balanceado modelo con respecto a la recall y precisión.

# # Conclusión

# In[92]:


# Redes Neuronales
y_predict_scores = model_red_neuronal.predict_proba(X_val_escalado)[:,1]
fpr, tper, thresholds = roc_curve(y_val, y_predict_scores)
roc_auc = roc_auc_score(y_val, y_predict_scores)
plt.plot(fpr, tper, marker='.', label='Red Neuronal (AUC = %0.2f)' % roc_auc, color='g', alpha=0.5)

# Boosting
y_predict_scores = model_boosting.predict_proba(X_val)[:,1]
fpr, tper, thresholds = roc_curve(y_val, y_predict_scores)
roc_auc = roc_auc_score(y_val, y_predict_scores)
plt.plot(fpr, tper, marker='.', label='Boosting (AUC = %0.2f)' % roc_auc, color='r', alpha=0.5)

# SVM con kernel lineal
y_predict_scores = model_SVM.decision_function(X_val)
fper, tper, thresholds = roc_curve(y_val, y_predict_scores)
roc_auc = roc_auc_score(y_val, y_predict_scores)
plt.plot(fper, tper, marker='.', label='SVM kernel lineal (AUC = %0.2f)' % roc_auc, color='g', alpha=0.5)

# SVM con kernel rbf
y_predict_scores = model_SVM_rbf.predict_proba(X_val)[:,1]
fper, tper, thresholds = roc_curve(y_val, y_predict_scores)
roc_auc = roc_auc_score(y_val, y_predict_scores)
plt.plot(fper, tper, marker='.', label='SVM kernel rbf (AUC = %0.2f)' % roc_auc, color='orange', alpha=0.5)

# Regresión logística
y_predict_scores = model_regresion_logistica.predict_proba(X_val)[:,1]
fper, tper, thresholds = roc_curve(y_val, y_predict_scores)
roc_auc = roc_auc_score(y_val, y_predict_scores)
plt.plot(fper, tper, marker='.', label='Regresión Logística (AUC = %0.2f)' % roc_auc ,color='blue', alpha=0.5)

# Voting
y_predict_scores = voting_clf.predict_proba(X_val)[:,1]
fpr, tper, thresholds = roc_curve(y_val, prob)
roc_auc = roc_auc_score(y_val, prob)
plt.plot(fpr, tper, color='purple', lw=2, label='Voting (AUC = %0.2f)' % roc_auc, alpha=0.5)

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Curvas ROC de todos los modelos')
plt.legend()
plt.grid()
plt.show()


# In[93]:


results = {
    'Model': ['Red Neuronal', 'Boosting', 'SVM lineal', 'SVM rbf', 'Regresión Logística', 'Ensamble (Votting)'],
    'Precisión': [0.68, 0.68, 0.69, 0.67, 0.71, 0.74],
    'Recall': [0.78, 0.81, 0.75, 0.75, 0.78, 0.72]
}

df = pd.DataFrame(results)

print(df.to_string())


# Los resultados utilizando diferentes modelos no dan tanta diferencia pero se podría destacar el ensamble (Votting Classifier) y el boosting al obtener los mejores resultados en cuanto a Roc Auc. Uno esperaría que el ensamble generalice mejor al utilizar 2 modelos diferentes. <br>
# Cabe aclarar que el proyecto se basaba más en explorar diferentes modelos y jugar con diferentes técnicas e hiperparámetros, no así tanto en centrarse en mejorar los modelos al máximo.

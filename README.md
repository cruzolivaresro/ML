### Apuntes de Machine learning ###

*Bibliografía*:
**Géron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow (2nd ed.). O'Reilly Media.**

*Antecedente*:

[Este articulo](https://www.cs.toronto.edu/~hinton/absps/ncfast.pdf) fue un relevo que tuvo deep learning el 2006, debido a que los investigadores en los 90's habian desistido de estas técnicas.

[Aquí puedes practicar los tópicos del libro](https://github.com/ageron/handson-ml2)

[Cursos de Deep Learning en Coursera](https://www.coursera.org/learn/machine-learning/?isNewUser=true#testimonials)

[Documentación scikit-learn](https://scikit-learn.org/stable/user_guide.html) 


**Capítulo 1: Panorama del aprendizaje automático**

¿Qué es y para qué usarlo?
Lo importante es que las máquinas sean programadas de alguna forma para que aprendan de los datos.

Tres cosas importantes existen en armonia: La experiencia a la que se enfrenta la máquina, las tareas que debe realizar la máquina y una medida de desempeño que oriente el aprendizaje.

Entonces, debemos entenderlo como: **si el desempeño en la tarea T, medido por P, mejora con la experiencia E.**

**Filtro spam:** dado una serie de ejemplos de correos spam etiquetados y una serie de ejemplos de correos normales (*ham*), se podría aprender a etiquetar correos en *spam* o *ham*.

**Training set:** conjunto de datos que el sistema utiliza para aprender, cada registro se le llama **sample**.

*Ejemplo:* La tarea es etiquetar correos **T** dada la experiencia de training set **E** y la medida de desempeño **P**.

Una medida de desempeño podría ser la proporción de correos correctamente etiquetados. También conocida como **Accuracy** o **Precisión** utilizada para tareas de clasificación:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Donde:
- **TP (True Positives)**: Correos spam correctamente clasificados como spam.
- **TN (True Negatives)**: Correos no spam correctamente clasificados como no spam.
- **FP (False Positives)**: Correos no spam incorrectamente clasificados como spam.
- **FN (False Negatives)**: Correos spam incorrectamente clasificados como no spam.

Un valor de **Accuracy** cercano a 1 indica un alto desempeño del clasificador.

**Data Mining:** Con técnicas de machine learning podemos cavar en grandes cantidades de datos para encontrar patrones en los datos que aparentemente no son faciles de reconocer en una primera instancia.

*Pasos:*
  - Estudiar el problema.
  - Recopilar datos.
  - Entrenar algoritmo ML.
  - Resultados.
  - Inspeccionar resultados.
  - Entender mejor el problema.
  - Iterar desde el paso 1 si es necesario.

*Recomendaciones de uso:*
  - Cuando los problemas tienen una gran cantidad de reglas.
  - Cuando los problemas son demasiado complejos, ML puede ser una solución util.
  - Cuando los entornos son fluctuantes, ML puede adaptarse a nuevos datos.
  - Para obtener conocimiento de problemas complejos y grandes cantidades de datos.

*Ejemplos de aplicación:*

  -**Clasificador de imágenes (CNN)**: Analizar imagenes de productos en una linea de producción para clasificarlos automáticamente.
  -**Segmentación semántica (CNN)**: Detector de tumores en scaneos cerebrales, cada pixel de la imagen es clasificado.  
  -**Procesamiento de lenguaje natural (RNN, CNN, Transformers)**: Clasificador automático de nuevos artículos o clasificador de comentarios ofensivos.  
  -**Summarization**: Rama del procesamiento de lenguaje natural para resumir textos.  
  -**Chatbots**: Involucra muchos topicos de PLN incluyendo **entendimiento de lenguaje natural (NLU)** y modulos de pregunta/respuesta.  
  -**Regresión**: Pronóstico de valores. Puedes utilizar métricas pasadas utilizando RNN, CNN o transformers.  
  -**Reconocimiento de voz**: Procesa archivos de audio, es considerado complejo, se utilizan RNN, CNN o transformers.  
  -**Detección de anomalías**: Detectar tarjetas de crédito fraudulentas.  
  -**Clustering**: Segmentar clientes basado en sus compras para crear distintas estrategias de marketing.
  -**Reducción de dimensionalidad**: Representar datos de alta dimension y complejos en un diagrama evidente.  
  -**Sistemas recomendadores**: Recomendar productos a clientes que podrían estar interesados basado en sus compras anteriores.  
  -**Aprendizaje reforzado**: Hacer un bot inteligente para juegos. Modelo en donde el agente escoge maximizar las recompensas a través del tiempo dado un ambiente específico. **AlphaGo**

*Tipos de Machine Learning*:
  - **(Supervised, Unsupervised, Semisupervised, RL)**: Si es que están o no siendo entrenados con supervisión humana.
  - **(online learning v/s batch learning)**: Si es que pueden o no aprender incrementalmente en el camino.
  - **(instance-based learning v/s model-based learning)**: Cuando se busca comparar nuevos datos con datos conocidos *o* busca detectar patrones en el set de entrenamiento y construir modelos predictivos.
  - Podemos mezclar estos tipos, por ejemplo, un filtro spam que pueda aprender en el camino usando una red neuronal profunda basado en modelo usando samples de spam o ham. *Online, model-based, Supervised*

**Supervisado**: El set de entrenamiento contiene las soluciones esperadas **(labels)**. Un ejemplo clásico es clasificación de spam/ham. Otro ejemplo es predecir el valor de un objetivo como el precio de un auto dado un set de cualidades llamados **predictores**.

Algoritmos relacionados al aprendizaje supervisado:

  - **K-nearest Neighbors**
  - **Linear Regression**
  - **Logistic Regression**
  - **Support vector machines**
  - **Decision Tree and Random Forests**
  - **Neural Networks**

**NO Supervisado**: El set de entrenamiento no contiene labels, por lo que aprende sin profesor.

Algoritmos relacionados al aprendizaje no supervisado:

  -**Clustering**
  
    K-Means
    DBSCAN
    Hierarchical Cluster Analysis (HCA)
    
  -**Anomaly detection and novelty detection**
  
    One-class SVM
    Isolation forest
    
  -**Visualization and Dimensionality reduction**
  
    Principal Component Analysis (PCA)
    Kernel PCA
    Locally Linear Embedding (LLE)
    t-Distributed Stochastic Neighbor Embedding (t-SNE)
    
  -**Association rule learning**
  
    Apriori
    Eclat

El libro recomienda aplicar reducción de dimensionalidad al dataset, ya sea de manera supervisada o no, para acelerar la ejecución del programa, reducir el uso de memoria y, en algunos casos, mejorar el rendimiento. **Sin embargo**, esto depende del contexto, ya que en ciertas situaciones podría ser más beneficioso aplicar técnicas de feature engineering. Es importante recordar que reducir la dimensionalidad puede ayudar a disminuir la correlación entre variables, pero también podría llevar a la pérdida de información relevante.

**Semisupervisado**: Cuando tienes algunos samples con label y otros no, por temas de economía. Algunos algoritmos pueden tratar con la falta de datos etiquetados. *(DBN y RBM)*

**Reinforcement Learning**: El sistema de aprendizaje es un **agente**, el cual puede observar un **ambiente** específico, puede seleccionar o realizar **acciones** y recibe **rewards** como retorno o **penalties**. Por lo tanto, debe aprender por sí mismo cuál es la mejor estrategia para obtener la mejor recompensa en el tiempo. Una **política** establecerá cual acción el agente debería elegir cuando se le da una situación.

*Pasos*:

  - Observar.
  - Seleccionar acción utilizando una política.
  - Actuar.
  - Obtener reward o penalty.
  - Actualizar la política **(paso de aprendizaje)**.
  - Iterar hasta que una política óptima a sido encontrada.


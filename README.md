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

***

**1. Supervisado**: El set de entrenamiento contiene las soluciones esperadas **(labels)**. Un ejemplo clásico es clasificación de spam/ham. Otro ejemplo es predecir el valor de un objetivo como el precio de un auto dado un set de cualidades llamados **predictores**.

Algoritmos relacionados al aprendizaje supervisado:

  - **K-nearest Neighbors**
  - **Linear Regression**
  - **Logistic Regression**
  - **Support vector machines**
  - **Decision Tree and Random Forests**
  - **Neural Networks**

***

**2. NO Supervisado**: El set de entrenamiento no contiene labels, por lo que aprende sin profesor.

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

***

**3. Semisupervisado**: Cuando tienes algunos samples con label y otros no, por temas de economía. Algunos algoritmos pueden tratar con la falta de datos etiquetados. *(DBN y RBM)*

***

**4. Reinforcement Learning**: El sistema de aprendizaje es un **agente**, el cual puede observar un **ambiente** específico, puede seleccionar o realizar **acciones** y recibe **rewards** como retorno o **penalties**. Por lo tanto, debe aprender por sí mismo cuál es la mejor estrategia para obtener la mejor recompensa en el tiempo. Una **política** establecerá cual acción el agente debería elegir cuando se le da una situación.

*Pasos*:

  - Observar.
  - Seleccionar acción utilizando una política.
  - Actuar.
  - Obtener reward o penalty.
  - Actualizar la política **(paso de aprendizaje)**.
  - Iterar hasta que una política óptima a sido encontrada.

***

**Batch & Online Learning**: Cuando el sistema puede o no aprender incrementalmente a partir de un flujo de datos entrantes.

**5. Batch**: El sistema es incapaz de aprender incrementalmente, es decir, se entrena utilizando toda la data. Si quieres que aprenda de nuevos datos, es necesario entrenar nuevamente utilizando la data nueva y la antigua. Debemos recordar que entrenar con todo el dataset puede ser costoso en recursos *(CPU, memoria, disco, I/O, network I*O, etc).* 

**6. Online**: El sistema es capaz de aprender incrementalmente, ya sea por instancias secuenciales de agregación de datos o en grupos pequeños **mini-batches**. En este sistema, un modelo es entrenado y lanzado en producción, pero seguirá aprendiendo de los nuevos datos que entren al modelo. Debemos recordar que este aprendizaje no es *en vivo*, sino que es incremental, es decir, el aprendizaje de todas formas es offline. También es importante considerar el parámetro con el que el sistema se adapta a los cambios en los datos llamado **learning rate**. Alto learning rate, tenderá a olvidar la data antigua. Bajo learning rate, aprenderá lento, pero será menos sensitivo al ruido en la nueva data u outliers. Si ves un problema en los resultados, siempre es util chequear la calidad del imput.

***

**Instance v/s model learning**: ¿Cómo se generaliza? En el sentido de que los sistemas ML tratan hacer predicciones y estan deben ser buenas pasa samples que no ha visto antes. Lo fundamental no es tanto las métricas, sino como funciona frente a nuevos samples.

**7. Instance-based learning**: Se utiliza una medida de similaridad entre objetos, por ejemplo, el numero de palabras que tienen en común. Por lo tanto, para este tipo de aprendizaje, el sistema aprende de memoria y generaliza para objetos comunes mediante una medida de similaridad.

**8. Model-based learning**: Se construye un modelo con el set de entrenamiento y se usa para realizar predicciones. Por ejemplo, si quiero saber si el dinero hace feliz a las personas, descargo datos especificos [Better Life Index](https://data-explorer.oecd.org/vis?tenant=archive&df[ds]=DisseminateArchiveDMZ&df[id]=DF_BLI&df[ag]=OECD) y descargo métricas del [PIB](https://www.imf.org/en/Publications/SPROLLS/world-economic-outlook-databases#sort=%40imfdate%20descending). Uno ambos datos y ordeno por PIB. Se observa que hay un patrón lineal entre ambas variables, entonces decido realizar un modelo de satisfacción de vida en función lineal de PIB. Este paso es selección de modelo, es decir, seleccionas un modelo lineal con un solo atributo:

$$
\mathrm{Life\_Satisfaction} = \Theta_0 + \Theta_1 \cdot PIB
$$


El cual es un modelo con dos parametros, donde al manejarlos, podemos representar una función lineal que mejor se ajuste al comportamiento de los datos. Para encontrar los mejores parametros, es util utilizar una medida de funcionamiento, por ejemplo, fitness (qué tan bueno) o cost (qué tan malo) function. Por ejemplo, usualmente se utiliza la función de costo que mide la distancia entre las predicciones del modelo y los samples de entrenamiento, con el fin de minimizar esta distancia.

*Pasos*:
  - Encuentras un patrón.
  - Estableces un modelo para una variable específica.
  - Alimentas el modelo con entrenamiento.
  - Encuentras los mejores parametros de ajuste.
  - Realizas predicciones.

Por ejemplo, podría predecir la satisfacción de vida, dado los datos de PIB de un país. Siempre debemos estar atento a los sesgos que podríamos imponer al utilizar modelos de esta manera.
Si el modelo no funcionace bien, entonces sería **útil agregar más predictores** *(employment rate, health, air pollution, etc.)* o **encontrar una mejor fuente de datos** o **seleccionar un mejor modelo** *(e.g., a Polynomial
Regression model)*.

***

*En general:*
  - Estudiamos los datos.
  - Seleccionamos un modelo.
  - Lo entrenas con datos de entrenamiento con la medida especifica.
  - Aplicas el modelo para realizar predicciones con la esperanza de que generalice bien.

***
**Sampling Bias**: Si tomas una muestra demasiado pequeña de la población, los resultados pueden estar influenciados por el azar y no reflejar realmente las tendencias generales. Aumentar el tamaño de la muestra no garantiza que sea representativa si el método de selección está sesgado.

**Ejemplo**: Si preguntas a solo 5 clientes de Studio Face qué servicio prefieren, podrías obtener respuestas que no reflejan las preferencias del total de clientes.

**Ejemplo**: Si solo encuestas a clientes que visitan el salón los martes, podrías obtener resultados sesgados y no reflejar las preferencias de quienes van otros días.

***

**Overfitting**: cuando un modelo es demasiado complejo en relacion con la cantidad y calidad de los datos de entrenamiento, es decir, el modelo aprende muy bien los detalles y el ruido de los datos de entrenamiento. Esto generaliza mal.

  -*Demasiado complejo*: Demasiados parametros o usa técnicas avanzadas que lo vuelven complejo.
  
  -*Poco training set*: Si el conjunto de datos pequeño, entonces el modelo puede memorizar datos en lugar de aprender patrones.
  
  -*Datos con mucho ruido*: Errores o valores atípicos, aprenderá de patrones incorrectos.

**Regularización**: Hacer modelos simples y reducir el riesgo de sobreajuste.

**Grados de libertad en estadistica**: indica cuantos valores pueden variar libremente en un conjunto de datos despues de aplicar restricciones. Ejemplo: tengo 5 variables que en promedio resulta 10. Puedo mover todas las variables, pero me restringo a que la ultima variable será constante. Por lo tanto, los grados de libertad son 4, debido a que la ultima variable está restringida a tener valores.

**Grados de libertad en ML**: indica cuantos parametros del modelo pueden ajustarse libremente para mejorar el ajuste de los datos. EJ: RL simple, hay 2 parametros ajustables.

*En ambos casos, los grados de libertad representan la cantidad de valores o parámetros que pueden cambiar libremente:*

  -En estadística inferencial, se usan para calcular pruebas como t-student, Chi-cuadrado o ANOVA.
  
  -En machine learning, se usan para describir la flexibilidad del modelo y su capacidad de ajustarse a los datos.

***

**Debemos evaluar y afinar modelo si es necesario para asegurar mejores predicciones**

Si el error de entrenamiento es bajo, pero la generalización de errores es alta, significa que el modelo esta sobreajustando los datos de entrenamiento.

**Sobreajuste al set test** o **data lakeable (fuga de datos) del set test**: Se sobreajusta el modelo para el conjunto test especifico. En un flujo de trabajo adecuado, el conjunto de prueba solo debe usarse una vez, al final del proceso, para evaluar el modelo final. Sin embargo, si mides el error en el test set varias veces y ajustas el modelo basándote en estos resultados, el modelo comienza a aprender las características específicas del test set, en lugar de aprender una representación general de los datos.

**Holdout validation**: se reserva una parte del conjunto de entrenamiento para evaluar varios modelos candidatos y seleccionar el mejor. El conjunto será el **validation set**.

En lugar de usar el test set para ajustar el modelo, divides el conjunto de datos en tres partes:

1. Training Set (conjunto de entrenamiento): Se usa para entrenar el modelo.
  
2. Validation Set (conjunto de validación o dev set): Se reserva una parte del training set (por ejemplo, el 20-30%) para evaluar diferentes modelos y ajustar los hiperparámetros. Permite comparar el rendimiento de distintos modelos sin afectar el test set. Si es muy pequeño, entonces sus evaluaciones seran imprecisas. Si es muy grande, training set será muy pequeño.

3. Test Set (conjunto de prueba): Se usa solo al final, después de elegir el mejor modelo, para obtener una estimación imparcial del error de generalización.

*Pasos*:

  -Dividir datos en entrenamiento y validacion.
  
  -Entrenar varios modelos con diferentes hiperparametros en el training set (sin validacion).
  
  -Evaluo modelos con validation set.
  
  -Reentreno el modelo selecionado usando el conjunto de entrenamiento completo (incluyendo validacion).
  
  -Prueba el modelo en set test para obtener una vision realista sobre el error de generalización.

**EJ**: Evaluación final con el Test Set, probamos este modelo en los 100 estudiantes que nunca ha visto (test set) para medir su verdadera precisión.

🔹 Si la precisión en el test set es similar a la del validation set, el modelo generaliza bien.

🔹 Si la precisión baja mucho, es posible que el modelo haya memorizado los datos de entrenamiento y necesite ajustes.

***

**Cross-validation**: Si usamos solo un conjunto de validación (como en holdout validation), podríamos tener suerte o mala suerte con la división de los datos.

*Pasos con K-Fold-Cross-Validation K=5*: **la idea es que cada parte se use como conjunto de validación una vez, mientras las otras se usan para entrenar.**

  -Dividimos los datos en 5 partes iguales.
  -Entrenamos el modelo con 4 partes y lo probamos en la restante.
  -Repetimos esto 5 veces, utilizando una parte distinta para probar cada ronda. (Entrenar y validar en cada iteración. Registrar métricas.)
  -Promediamos los resultados de las 5 pruebas para obtener una mejor estimación del rendimiento real del modelo.

**La validación cruzada repetida nos da una mejor medida del rendimiento del modelo al probarlo en muchos conjuntos de validación diferentes, pero a cambio aumenta mucho el tiempo de entrenamiento.**

**Por lo tanto, iteras por cada parte, y las restantes las utilizas de entrenamiento.**

Supongamos que tenemos 5000 datos en total.

1️⃣ Dividimos los datos inicialmente en:

  -Training Set (4000 datos) → Usado para K-Fold.
  
  -Test Set (1000 datos) → Guardado aparte para la evaluación final.
  
2️⃣ Aplicamos K-Fold (K=5) en el Training Set (4000 datos)

  -Entrenamos y validamos 5 veces con distintos folds.

  -Calculamos el promedio de precisión.

  -Elegimos el mejor modelo según los resultados de validación.
  
3️⃣ Entrenamos el modelo final con los 4000 datos completos

  -Ya no usamos folds.
  
  -El objetivo es que el modelo aproveche toda la información posible.
  
4️⃣ Evaluamos en el Test Set (1000 datos)

  -Ahora usamos los datos de prueba reales que el modelo nunca ha visto.
  
  -Esto nos da la verdadera precisión del modelo en datos nuevos.

***

**Debemos darnos la posibilidad de siempre preguntarnos si los datos son representativos**

**En este caso, la regla más importante a recordar es que el conjunto de validación y el conjunto de prueba deben ser lo más representativos posible de los datos que espera utilizar en producción**

**Desajuste de los datos**:

EJ: Debemos asegurarnos de que estos conjuntos contengan solo imágenes tomadas con la app, no imágenes de internet. Esto nos ayudará a medir el rendimiento real del modelo en condiciones reales.

Después de entrenar con imágenes de internet, podríamos notar que el modelo tiene mal desempeño en el validation set **¿Por qué?**:

  -Modelo ha sobreajustado a las imagenes de internet.
  
  -Las imagenes de internet no se parecen a las de la app.

**Para esto utilizamos Train-Dev set**

*¿Qué es el Train-Dev Set y cómo ayuda?*

Imagina que quieres entrenar un modelo de Machine Learning que clasifique imágenes de perros y gatos.

1️⃣ Los datos disponibles:

  -Descargas 1,000,000 imágenes de perros y gatos de internet.
  
  -Tomas 10,000 imágenes de perros y gatos tomadas con la cámara del celular, que representan las imágenes reales que los usuarios tomarán en la app.

Conjunto        /      Cantidad      /      Origen

Training Set	  /      990,000       /	    Imágenes de internet (se usa para entrenar)

Train-Dev Set	  /      10,000        /      Imágenes de internet (se usa para evaluar si hay overfitting)

Validation Set  /	     5,000         /    	Imágenes tomadas con el celular (se usa para medir el desempeño en datos reales)

Test Set        /      5,000         /      Imágenes tomadas con el celular (se usa solo para la evaluación final)


**Train-Dev Set: Es un subconjunto de imágenes de internet que NO se usa para entrenar, solo para evaluar si el modelo ha sobreajustado.**

**Validation Set y Test Set: Son imágenes reales tomadas con celular, ya que representan los datos que el modelo verá en producción.**


📌 Entrenas el modelo con el Training Set (990,000 imágenes de internet).

📌 Luego, lo evalúas en el Train-Dev Set (10,000 imágenes de internet que no se usaron en el entrenamiento).

📌 Luego, lo evalúas en el Validation Set (5,000 imágenes reales del celular).


Caso 1: El modelo tiene buen desempeño en Train-Dev pero mal en Validación

Conjunto	                            Precisión

Train-Dev Set (imágenes de internet)	90% ✅

Validation Set (imágenes del celular)	60% ❌

**El modelo funciona bien en imágenes de internet, pero mal en imágenes reales. Hay un desajuste de datos (data mismatch).**


Caso 2: El modelo tiene mal desempeño en Train-Dev y en Validación

Conjunto	                            Precisión

Train-Dev Set (imágenes de internet)	70% ❌

Validation Set (imágenes del celular)	60% ❌

**El modelo no generaliza bien ni siquiera en imágenes de internet. Está sobreajustando al Training Set.**


***
📌 Una vez que encuentras la mejor versión del modelo, lo reentrenas con todos los datos de internet + datos reales.

📌 Finalmente, lo pruebas en el Test Set (5,000 imágenes reales de celular) para obtener su precisión final.

📌 Train-Dev Set se usa para ver si hay overfitting al entrenamiento.

📌 Si el modelo falla en Train-Dev, hay overfitting.

📌 Si el modelo solo falla en Validación, hay desajuste de datos (data mismatch).

# Descripción del Código

El código implementado en C++ trata sobre la multiplicación de matrices utilizando la biblioteca MPI (Message Passing Interface) para la computación paralela. Esta técnica permite distribuir la carga de trabajo entre múltiples procesos o nodos de un sistema distribuido, mejorando así el rendimiento y la eficiencia computacional al aprovechar los recursos disponibles de manera concurrente.

## Descripción Detallada del Código

El programa comienza con la inicialización de MPI, configurando el entorno de ejecución para permitir la comunicación entre procesos. A continuación, el código verifica y lee los parámetros de entrada desde la línea de comandos, que especifican las dimensiones de las matrices A, B y C para la multiplicación matricial C = A * B.

Una vez que se han leído y validado las dimensiones de las matrices, el proceso principal (rank 0) inicializa las matrices A y B. La matriz A se inicializa con valores ascendentes en cada fila, mientras que la matriz B se inicializa con valores ascendentes en cada columna. Estas matrices se muestran por pantalla para verificar su correcta inicialización antes de proceder con la multiplicación.

Luego, se procede con la distribución de los datos necesarios para la multiplicación entre los procesos utilizando MPI. Cada proceso recibe una porción de la matriz A y la matriz B completa se difunde (broadcast) a todos los procesos. A continuación, cada proceso realiza la multiplicación de la porción de A que recibió con la matriz B completa, calculando así su parte de la matriz C.

Una vez completada la multiplicación local, los resultados se recolectan de vuelta en el proceso 0 utilizando operaciones de recolección (Gather) y recolección variable (Gatherv) de MPI. Estas operaciones aseguran que la matriz C resultante se reconstruya correctamente en el proceso 0 a partir de las contribuciones individuales de cada proceso.

Finalmente, se calcula y muestra el tiempo total de ejecución del programa para evaluar el rendimiento del algoritmo paralelo implementado.

## Importancia y Aplicaciones

La multiplicación de matrices es un problema fundamental en la computación científica y la inteligencia artificial, utilizado en una amplia variedad de aplicaciones que van desde simulaciones físicas y financieras hasta aprendizaje automático y procesamiento de imágenes. La implementación paralela de este problema, como se muestra en este código, es esencial para manejar grandes volúmenes de datos y mejorar significativamente los tiempos de ejecución en sistemas distribuidos.

En resumen, el código proporcionado demuestra cómo aprovechar MPI para realizar eficientemente la multiplicación de matrices en un entorno distribuido, mostrando paso a paso cómo se distribuyen los datos, se realiza el cálculo paralelo y se recolectan los resultados finales para su uso posterior.

#include <iostream>
#include <vector>
#include <stdexcept>
#include <mpi.h>

/**
 * @brief Imprime una matriz representada en un vector por filas.
 *
 * @param mat Vector que contiene los elementos de la matriz en formato de fila.
 * @param rows Número de filas de la matriz.
 * @param cols Número de columnas de la matriz.
 *
 * Esta función toma una matriz almacenada en un vector y la imprime en formato
 * de filas y columnas. Es útil para visualizar matrices durante la depuración.
 */
void print_mat(const std::vector<double>& mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Punto de entrada del programa.
 *
 * Realiza la multiplicación de matrices C = A * B utilizando MPI.
 *
 * @param argc Número de argumentos en la línea de comandos.
 * @param argv Argumentos de la línea de comandos. Se espera:
 *             argv[1] --l [filas de A]
 *             argv[2] --m [columnas de B]
 *             argv[3] --n [columnas de A/filas de B]
 * @return 0 si se ejecuta correctamente, otro valor si hay un error.
 *
 * Este programa realiza la multiplicación de matrices de forma paralela utilizando MPI.
 * Los pasos principales incluyen:
 *
 * 1. **Verificar el número de argumentos**:
 *   @code
 *   if (argc != 7) {
 *       std::cerr << "Usage: " << argv[0] <<
 *           " --l [filas de A] --m [columnas de B] --n [columnas de A/filas de B]" << std::endl;
 *       return 1;
 *   }
 *   @endcode
 *
 * 2. **Leer las dimensiones de las matrices**:
 *   @code
 *   int l, m, n;
 *   try {
 *       l = std::stoi(argv[2]);
 *       m = std::stoi(argv[4]);
 *       n = std::stoi(argv[6]);
 *   } catch (const std::invalid_argument& e) {
 *       std::cerr << "Error: Argumentos no válidos. Se esperaban números enteros." << std::endl;
 *       return 1;
 *   } catch (const std::out_of_range& e) {
 *       std::cerr << "Error: Desbordamiento al convertir los números." << std::endl;
 *       return 1;
 *   }
 *   @endcode
 *
 * 3. **Inicializar MPI y obtener el número de procesos y el rango de cada proceso**:
 *   @code
 *   MPI_Init(&argc, &argv);
 *   int num_procs, rank;
 *   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);  // Obtener número de procesos
 *   MPI_Comm_rank(MPI_COMM_WORLD, &rank);      // Obtener rango de este proceso
 *   @endcode
 *
 * 4. **Definir e inicializar las matrices A y B en el proceso 0**:
 *   @code
 *   std::vector<double> A(l * n, 0.0);  // Matriz A de tamaño l x n inicializada a 0
 *   std::vector<double> B(n * m, 0.0);  // Matriz B de tamaño n x m inicializada a 0
 *   if (rank == 0) {
 *       // Inicialización de la matriz A
 *       for (int i = 0; i < l; ++i) {
 *           for (int j = 0; j < n; ++j) {
 *               A[i * n + j] = i;  // Asignar valor a cada elemento de A
 *           }
 *       }
 *       // Inicialización de la matriz B
 *       for (int i = 0; i < n; ++i) {
 *           for (int j = 0; j < m; ++j) {
 *               B[i * m + j] = j;  // Asignar valor a cada elemento de B
 *           }
 *       }
 *       // Mostrar las matrices A y B
 *       std::cout << "Matrix A = " << std::endl;
 *       print_mat(A, l, n);
 *       std::cout << "Matrix B = " << std::endl;
 *       print_mat(B, n, m);
 *       std::cout << "Calculando multiplicación..." << std::endl;
 *   }
 *   @endcode
 *
 * 5. **Distribuir las porciones de las matrices a los diferentes procesos**:
 *   @code
 *   MPI_Bcast(B.data(), n * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 *   int llocal = l / num_procs;  // Filas por proceso
 *   int rest = l % num_procs;    // Resto de filas
 *   if (rank < rest) {
 *       llocal += 1;  // Algunos procesos tendrán una fila adicional
 *   }
 *   int start = rank * llocal;  // Índice de inicio local para este proceso
 *   if (rank >= rest) {
 *       start += rest;  // Ajustar inicio para compensar los procesos adicionales
 *   }
 *   int end = start + llocal;  // Índice de fin local para este proceso
 *   std::vector<double> llocal_A(llocal * n);  // Porción local de A para este proceso
 *   MPI_Scatter(A.data(), llocal * n, MPI_DOUBLE,
 *               llocal_A.data(), llocal * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 *   @endcode
 *
 * 6. **Realizar la multiplicación de matrices en paralelo**:
 *   @code
 *   std::vector<double> C(llocal * m, 0.0);  // Matriz C local para este proceso
 *   for (int i = 0; i < llocal; ++i) {
 *       for (int j = 0; j < m; ++j) {
 *           for (int k = 0; k < n; ++k) {
 *               C[i * m + j] += llocal_A[i * n + k] * B[k * m + j];
 *           }
 *       }
 *   }
 *   @endcode
 *
 * 7. **Recolectar los resultados y calcular el tiempo total de ejecución**:
 *   @code
 *   std::vector<double> C_total;  // Matriz C completa en el proceso 0
 *   if (rank == 0) {
 *       C_total.resize(l * m);  // Redimensionar C_total para contener la matriz completa
 *   }
 *   int counts_recv[num_procs];    // Cantidad de elementos que cada proceso enviará
 *   int displacements[num_procs];  // Desplazamientos de los datos enviados por cada proceso
 *   int counts = llocal * m;       // Cantidad de elementos enviados por este proceso
 *   if (rank == 0) {
 *       MPI_Gather(&counts, 1, MPI_INT, counts_recv, 1, MPI_INT, 0, MPI_COMM_WORLD);
 *   } else {
 *       MPI_Gather(&counts, 1, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
 *   }
 *   int displace = start * m;  // Desplazamiento para este proceso
 *   if (rank == 0) {
 *       MPI_Gather(&displace, 1, MPI_INT, displacements, 1, MPI_INT, 0, MPI_COMM_WORLD);
 *   } else {
 *       MPI_Gather(&displace, 1, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
 *   }
 *   if (rank == 0) {
 *       MPI_Gatherv(&C[0], llocal * m, MPI_DOUBLE, &C_total[0],
 *                   counts_recv, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 *   } else {
 *       MPI_Gatherv(&C[0], llocal * m, MPI_DOUBLE, nullptr,
 *                   nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 *   }
 *   if (rank == 0) {
 *       double time2 = MPI_Wtime();
 *       std::cout << "Tiempo total de ejecución: " << time2 - time1 << " segundos." << std::endl;
 *   }
 *   MPI_Finalize();
 *   return 0;
 *   @endcode
 */
int main(int argc, char** argv) {
    std::cout << "Multiplicación de matrices!" << std::endl;
    std::cout << "La idea es calcular C = A * B" << std::endl;

    // Verificar número de argumentos
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] <<
            " --l [filas de A] --m [columnas de B] --n [columnas de A/filas de B]" << std::endl;
        return 1;
    }

    // Leer dimensiones de las matrices
    int l, m, n;
    try {
        l = std::stoi(argv[2]);
        m = std::stoi(argv[4]);
        n = std::stoi(argv[6]);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: Argumentos no válidos. Se esperaban números enteros." << std::endl;
        return 1;
    } catch (const std::out_of_range& e) {
        std::cerr << "Error: Desbordamiento al convertir los números." << std::endl;
        return 1;
    }

    // Inicializar MPI
    MPI_Init(&argc, &argv);
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);  // Obtener número de procesos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);      // Obtener rango de este proceso

    // Definir matrices A, B
    std::vector<double> A(l * n, 0.0);  // Matriz A de tamaño l x n inicializada a 0
    std::vector<double> B(n * m, 0.0);  // Matriz B de tamaño n x m inicializada a 0

    // Si el proceso es el 0, inicializar las matrices A y B y mostrarlas
    if (rank == 0) {
        // Inicialización de la matriz A
        for (int i = 0; i < l; ++i) {
            for (int j = 0; j < n; ++j) {
                A[i * n + j] = i;  // Asignar valor a cada elemento de A
            }
        }

        // Inicialización de la matriz B
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                B[i * m + j] = j;  // Asignar valor a cada elemento de B
            }
        }

        // Mostrar las matrices A y B
        std::cout << "Matrix A = " << std::endl;
        print_mat(A, l, n);
        std::cout << "Matrix B = " << std::endl;
        print_mat(B, n, m);

        std::cout << "Calculando multiplicación..." << std::endl;
    }

    double time1 = MPI_Wtime();  // Iniciar cronómetro MPI

    // Compartir los datos de B a todos los procesos
    MPI_Bcast(B.data(), n * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // IMPORTANTE: Descomposición del dominio
    // Calcular cuántas filas le corresponden a cada proceso
    // Y también calcular los índices globales y locales
    int llocal = l / num_procs;  // Filas por proceso
    int rest = l % num_procs;    // Resto de filas
    if (rank < rest) {
        llocal += 1;  // Algunos procesos tendrán una fila adicional
    }
    int start = rank * llocal;  // Índice de inicio local para este proceso
    if (rank >= rest) {
        start += rest;  // Ajustar inicio para compensar los procesos adicionales
    }
    int end = start + llocal;  // Índice de fin local para este proceso

    // Compartir las porciones de A a cada proceso
    std::vector<double> llocal_A(llocal * n);  // Porción local de A para este proceso
    MPI_Scatter(A.data(), llocal * n, MPI_DOUBLE,
                llocal_A.data(), llocal * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Solo el pedazo de C en el cual calculo
    std::vector<double> C(llocal * m, 0.0);  // Matriz C local para este proceso

    // Calcular multiplicación de matrices C = A * B para las filas asignadas
    for (int i = 0; i < llocal; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i * m + j] += llocal_A[i * n + k] * B[k * m + j];
            }
        }
    }

    // Preparación para recolección de resultados
    std::vector<double> C_total;  // Matriz C completa en el proceso 0

    // Si este proceso es el 0, preparar para recolectar resultados en C_total
    if (rank == 0) {
        C_total.resize(l * m);  // Redimensionar C_total para contener la matriz completa
    }

    // Comunicación para recolectar resultados de todos los procesos
    // Determinar cuántos elementos enviará cada proceso y a dónde se enviarán
    int counts_recv[num_procs];    // Cantidad de elementos que cada proceso enviará
    int displacements[num_procs];  // Desplazamientos de los datos enviados por cada proceso
    int counts = llocal * m;       // Cantidad de elementos enviados por este proceso
    if (rank == 0) {
        MPI_Gather(&counts, 1, MPI_INT, counts_recv, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gather(&counts, 1, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Calcular desplazamientos para los datos enviados
    int displace = start * m;  // Desplazamiento para este proceso
    if (rank == 0) {
        MPI_Gather(&displace, 1, MPI_INT, displacements, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gather(&displace, 1, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Recolectar los datos de C desde todos los procesos al proceso 0
    if (rank == 0) {
        MPI_Gatherv(&C[0], llocal * m, MPI_DOUBLE, &C_total[0],
                    counts_recv, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(&C[0], llocal * m, MPI_DOUBLE, nullptr,
                    nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Tiempo de finalización y finalización de MPI
    if (rank == 0) {
        double time2 = MPI_Wtime();
        std::cout << "Tiempo total de ejecución: " << time2 - time1 << " segundos." << std::endl;
    }

    MPI_Finalize();
    return 0;
}

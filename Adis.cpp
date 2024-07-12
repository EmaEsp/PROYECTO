#include <iostream>
#include <vector>
#include <stdexcept>
#include <mpi.h>

/**
 * Función para imprimir una matriz representada como un vector.
 *
 * @param mat Vector que representa la matriz.
 * @param rows Número de filas de la matriz.
 * @param cols Número de columnas de la matriz.
 */
void print_mat(const std::vector<double>& mat, int rows, int cols){
  for (int i = 0; i < rows; ++i){
    for(int j = 0; j < cols; ++j){
      std::cout << mat[i * cols + j] << " ";  // Imprime el elemento i, j
    }
    std::cout << std::endl;  // Nueva línea después de cada fila
  }
}

/**
 * Función principal para realizar la multiplicación de matrices usando MPI.
 *
 * @param argc Número de argumentos de la línea de comandos.
 * @param argv Argumentos de la línea de comandos.
 * @return 0 si el programa se ejecuta correctamente, 1 si hay errores.
 *
 * Este programa realiza la multiplicación de matrices C = A * B utilizando MPI para computación paralela.
 * Los pasos principales son los siguientes:
 *
 * 1. **Verificar número de argumentos:**
 *    @code
 *    if (argc != 7) {
 *        std::cerr << "Usage: " << argv[0] <<
 *            " --l [filas de A] --m [columnas de B] --n [columnas de A/filas de B]" << std::endl;
 *        return 1;
 *    }
 *    @endcode
 *
 * 2. **Leer dimensiones de las matrices:**
 *    @code
 *    int l, m, n;
 *    try {
 *        l = std::stoi(argv[2]);
 *        m = std::stoi(argv[4]);
 *        n = std::stoi(argv[6]);
 *    } catch (const std::invalid_argument& e) {
 *        std::cerr << "Error: Argumentos no válidos. Se esperaban números enteros." << std::endl;
 *        return 1;
 *    } catch (const std::out_of_range& e) {
 *        std::cerr << "Error: Desbordamiento al convertir los números." << std::endl;
 *        return 1;
 *    }
 *    @endcode
 *
 * 3. **Inicializar MPI:**
 *    @code
 *    MPI_Init(&argc, &argv);
 *    int num_procs, rank;
 *    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
 *    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 *    @endcode
 *
 * 4. **Inicializar matrices A y B (en el proceso 0):**
 *    @code
 *    if(rank == 0) {
 *        for(int i = 0; i < l; ++i) {
 *            for(int j = 0; j < n; ++j) {
 *                A[i * n + j] = i;
 *            }
 *        }
 *        for(int i = 0; i < n; ++i) {
 *            for(int j = 0; j < m; ++j) {
 *                B[i * m + j] = j;
 *            }
 *        }
 *    }
 *    @endcode
 *
 * 5. **Compartir los datos de B con todos los procesos:**
 *    @code
 *    MPI_Bcast(B.data(), n * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 *    @endcode
 *
 * 6. **Distribuir las filas de A entre los procesos:**
 *    @code
 *    std::vector<int> counts_recv(num_procs);
 *    std::vector<int> displacements(num_procs);
 *    for (int i = 0; i < num_procs; ++i) {
 *        counts_recv[i] = (llocal + (i < rest ? 1 : 0)) * n;
 *        displacements[i] = i * llocal * n + std::min(i, rest) * n;
 *    }
 *    MPI_Scatterv(A.data(), counts_recv.data(), displacements.data(), MPI_DOUBLE,
 *                 llocal_A.data(), llocal * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 *    @endcode
 *
 * 7. **Calcular la multiplicación local de matrices:**
 *    @code
 *    for (int i = 0; i < llocal; ++i) {
 *        for (int j = 0; j < m; ++j) {
 *            for (int k = 0; k < n; ++k) {
 *                C[i * m + j] += llocal_A[i * n + k] * B[k * m + j];
 *            }
 *        }
 *    }
 *    @endcode
 *
 * 8. **Reunir los resultados parciales en el proceso 0:**
 *    @code
 *    MPI_Gatherv(C.data(), llocal * m, MPI_DOUBLE, C_total.data(),
 *                counts_recv.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
 *    @endcode
 *
 * 9. **Imprimir la matriz resultante y el tiempo de ejecución (en el proceso 0):**
 *    @code
 *    if(rank == 0) {
 *        print_mat(C_total, l, m);
 *        std::cout << "Tiempo de multiplicación de matrices: " << time2 - time1 << std::endl;
 *    }
 *    @endcode
 *
 * 10. **Finalizar MPI:**
 *    @code
 *    MPI_Finalize();
 *    @endcode
 */
 
int main(int argc, char** argv) {
  std::cout << "Multiplicación de matrices!" << std::endl;
  std::cout << "La idea es calcular C = A * B" << std::endl;

  // Verificar número de argumentos
  if (argc != 7) {
    std::cerr << "Usage: " << argv[0] <<
        " --l [filas de A] --m [columnas de B] --n [columnas de A/filas de B]" << std::endl;
    return 1;  // Salir con código de error si el número de argumentos es incorrecto
  }

  // Leer dimensiones de las matrices desde los argumentos de línea de comandos
  int l, m, n;
  try{
    l = std::stoi(argv[2]);  // Filas de A
    m = std::stoi(argv[4]);  // Columnas de B
    n = std::stoi(argv[6]);  // Columnas de A / Filas de B
  }catch (const std::invalid_argument& e) {
    std::cerr << "Error: Argumentos no válidos. Se esperaban números enteros." << std::endl;
    return 1;
  }catch (const std::out_of_range& e) {
    std::cerr << "Error: Desbordamiento al convertir los números." << std::endl;
    return 1;
  }

  // Inicializar MPI
  MPI_Init(&argc, &argv);
  int num_procs, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);  // Obtener el número de procesos
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);      // Obtener el rango (ID) del proceso actual

  // Definir matrices A, B
  std::vector<double> A(l * n, 0.0);  // Matriz A de tamaño l x n inicializada a 0
  std::vector<double> B(n * m, 0.0);  // Matriz B de tamaño n x m inicializada a 0

  if(rank == 0){
    // Inicialización de A
    for(int i = 0; i < l; ++i){
      for(int j = 0; j < n; ++j){
        A[i * n + j] = i;  // Inicializar A con valores crecientes en las filas
      }
    }

    // Inicialización de B
    for(int i = 0; i < n; ++i){
      for(int j = 0; j < m; ++j){
        B[i * m + j] = j;  // Inicializar B con valores crecientes en las columnas
      }
    }
    std::cout << "Matrix A = " << std::endl;
    print_mat(A, l, n);
    std::cout << "Matrix B = " << std::endl;
    print_mat(B, n, m);

    std::cout << "Calculando multiplicación..." << std::endl;
  }

  double time1 = MPI_Wtime();
  // Compartir los datos de B en todos los procesos 
  MPI_Bcast(B.data(), n * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Descomposición del dominio
  int llocal = l / num_procs;  // Número de filas que cada proceso manejará
  int rest = l % num_procs;    // Filas adicionales que se distribuirán a algunos procesos

  // Preparar counts y displacements para MPI_Scatterv
  std::vector<int> counts_recv(num_procs);
  std::vector<int> displacements(num_procs);
  for (int i = 0; i < num_procs; ++i) {
    counts_recv[i] = (llocal + (i < rest ? 1 : 0)) * n;
    displacements[i] = i * llocal * n + std::min(i, rest) * n;
  }
  llocal = counts_recv[rank] / n;  // Actualizar llocal basado en los datos recibidos

  // Distribuir los bloques de A usando MPI_Scatterv
  std::vector<double> llocal_A(llocal * n);
  MPI_Scatterv(A.data(), counts_recv.data(), displacements.data(), MPI_DOUBLE,
               llocal_A.data(), llocal * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Calcular multiplicación de matrices C = A * B para las filas asignadas
  std::vector<double> C(llocal * m, 0.0);  // Resultado parcial de la multiplicación

  for (int i = 0; i < llocal; ++i) {
    for (int j = 0; j < m; ++j) {
      for (int k = 0; k < n; ++k) {
        C[i * m + j] += llocal_A[i * n + k] * B[k * m + j];
      }
    }
  }

  // Preparar para reunir los resultados en el proceso 0
  std::vector<double> C_total;  // Matriz final C
  if(rank == 0){
    C_total.resize(l * m);  // Reservar espacio para la matriz completa C
  }

  // Obtener tamaños y desplazamientos para MPI_Gatherv
  if(rank == 0)
    MPI_Gather(&llocal, 1, MPI_INT, counts_recv.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  else
    MPI_Gather(&llocal, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);

  int displace = (displacements[rank] / n) * m;  // Desplazamiento para cada proceso
  if(rank == 0)
    MPI_Gather(&displace, 1, MPI_INT, displacements.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  else
    MPI_Gather(&displace, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);

  // Reunir los resultados parciales en C_total usando MPI_Gatherv
  if(rank == 0)
    MPI_Gatherv(C.data(), llocal * m, MPI_DOUBLE, C_total.data(),
                counts_recv.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  else
    MPI_Gatherv(C.data(), llocal * m, MPI_DOUBLE, NULL,
                NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  double time2 = MPI_Wtime();

  // Imprimir la matriz resultante C_total en el proceso 0
  if(rank == 0)
    print_mat(C_total, l, m);

  // Imprimir el tiempo de ejecución en el proceso 0
  if(rank == 0){
    std::cout << "Tiempo de multiplicación de matrices: " << time2 - time1 << std::endl;
  }

  // Finalizar MPI
  MPI_Finalize();

  return 0;
}


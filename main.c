#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

double** perform_relaxation(double** array, int size, int threads, double precision) {
    if (size < 3) return array; // 3x3 is the smallest array to perform this operation.
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    int changed = 1;
    double currValue;
    while (changed) { // Iterate until the values have converged.
        changed = 0;
        // Main relaxation computation
        // Compute the average for the same set range of elements in all rows.
        #pragma omp parallel for private(currValue)
        for (unsigned int i = 1; i < size-1; i++) {
            for (unsigned int j = 1; j < size-1; j++) {
                // Take average and round to specified precision.
                currValue = array[i][j];
                array[i][j] = (array[i-1][j] + array[i+1][j] +
                        array[i][j-1] + array[i][j+1])/4;

                // Set bool flag to whether value has changed.
                // No need for critical region, as "changed" is only ever set to 1.
                if (!changed) {
                    if (fabs(array[i][j]-currValue) >= precision) changed = 1;
                }
            }
        }
    }
    /*
     * Correctness testing:
     * Test different values against the average of their neighbours
     * If the values match, then we can assume convergence has been reached
     * Hence, the program functions correctly.
     */
    // Value to check must not be part of the outer values and must be in range.
    int i = 4; int j = 6;
    printf("Value to test: %.3lf\tAbove: %.3lf\tBelow: %.3lf\tLeft: %.3lf\t"
           "Right: %.3lf\n", array[i][j],
           array[i-1][j], array[i+1][j],
           array[i][j-1], array[i][j+1]);
    return array;
}

int main () {
    int size = 3000; // Size
    // Create original array to pass into the function.
    double** newArray = malloc(size * sizeof(double*));
    if (newArray == NULL) return -1;
    for (int unsigned i = 0; i < size; i++) {
        newArray[i] = malloc(size * sizeof(double));
        if (newArray[i] == NULL) return -1;
    }
    for (unsigned int j = 0; j < size; j++) {
        newArray[0][j] = 1;
    }
    for (unsigned int i = 1; i < size; i++) {
        newArray[i][0] = 1;
    }
    for (unsigned int i = 1; i < size; i++) {
        for (unsigned int j = 1; j < size; j++) {
            newArray[i][j] = 0;
        }
    }

    time_t start, end;
    time(&start);
    newArray = perform_relaxation(newArray, size, 1, 0.001); // Array, size, threads, precision
    time(&end);

    printf("Time taken: %g seconds.\n", difftime(end, start));

    // Script used for testing.
    // FILE* file = fopen("results.out", "a");
//     for (unsigned int i = 0; i < size; i++) {
//         for (unsigned int j = 0; j < size; j++) {
//             printf("%.3f\t", newArray[i][j]);
    // fprintf(file, "%.3f,", newArray[i][j]);
//         }
//         printf("\n");
//     }
    // fprintf(file, "\n");

    for (unsigned int i = 0; i < size; i++) free(newArray[i]);
    free(newArray);
    return 0;
}

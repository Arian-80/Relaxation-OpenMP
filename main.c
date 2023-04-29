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
        #pragma omp parallel for default(none) private(currValue) \
        shared(size, array, changed, precision)
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
    int i = 23153/2000; int j = 23153%2000;
    printf("Value to test: %.3lf\tAbove: %.3lf\tBelow: %.3lf\tLeft: %.3lf\t"
           "Right: %.3lf\n", array[i][j],
           array[i-1][j], array[i+1][j],
           array[i][j-1], array[i][j+1]);
    printf("Expected value: %.3lf\n", (array[i][j+1] + array[i][j-1] + array[i+1][j] + array[i-1][j]) / 4);
    return array;
}

int main () {
    FILE *f = fopen("times.txt", "a");
        for (int k = 1; k < 9; k++) {
            if (k == 3) k = 4;
            if (k == 5) k = 6;
            if (k == 7) k = 8;
            int size = 5000; // Size
            // Create original array to pass into the function.
            double **newArray = malloc(size * sizeof(double *));
            if (newArray == NULL) return -1;
            for (int unsigned i = 0; i < size; i++) {
                newArray[i] = calloc(size, sizeof(double));
                if (newArray[i] == NULL) return -1;
            }
            for (unsigned int i = 0; i < size; i++) {
                newArray[0][i] = 1;
                newArray[i][0] = 1;
            }

            double start, end;
            start = omp_get_wtime();
            newArray = perform_relaxation(newArray, size, k, 0.001); // Array, size, threads, precision
            end = omp_get_wtime();

            printf("Time taken: %g seconds.\n", end - start);

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
            fprintf(f, "%g,", end - start);
            for (unsigned int i = 0; i < size; i++) free(newArray[i]);
            free(newArray);
        }
        fprintf(f, "\n");
    fclose(f);
    return 0;
}

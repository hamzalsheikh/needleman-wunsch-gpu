
#include <assert.h>
#include <stdio.h>
#include <unistd.h>

#include "common.h"
#include "timer.h"

void nw_cpu(unsigned char* reference, unsigned char* query, int* matrix, unsigned int N) {
    for(int q = 0; q < N; ++q) {
        for (int r = 0; r < N; ++r) {
            // Get neighbors
            int top     = (q == 0)?((r + 1)*DELETION):(matrix[(q - 1)*N + r]);
            int left    = (r == 0)?((q + 1)*INSERTION):(matrix[q*N + (r - 1)]);
            int topleft = (q == 0)?(r*DELETION):((r == 0)?(q*INSERTION):(matrix[(q - 1)*N + (r - 1)]));
            // Find scores based on neighbors
            int insertion = top + INSERTION;
            int deletion  = left + DELETION;
            int match     = topleft + ((query[q] == reference[r])?MATCH:MISMATCH);
            // Select best score
            int max = (insertion > deletion)?insertion:deletion;
            max = (match > max)?match:max;
            matrix[q*N + r] = max;
        }
    }
}

void generateQuery(unsigned char* reference, unsigned char* query, unsigned int N) {
    const float PROB_MATCH = 0.80f;
    const float PROB_INS   = 0.10f;
    const float PROB_DEL   = 1.00f - PROB_MATCH - PROB_INS;
    assert(PROB_MATCH >= 0.00f && PROB_MATCH <= 1.00f);
    assert(PROB_INS   >= 0.00f && PROB_INS   <= 1.00f);
    assert(PROB_DEL   >= 0.00f && PROB_DEL   <= 1.00f);
    unsigned int r = 0, q = 0;
    while(r < N && q < N) {
        float prob = rand()*1.0f/RAND_MAX;
        if(prob < PROB_MATCH) {
            query[q++] = reference[r++]; // Match
        } else if(prob < PROB_MATCH + PROB_INS) {
            query[q++] = rand()%256; // Insertion
        } else {
            ++r; // Deletion
        }
    }
    while(q < N) {
        query[q++] = rand()%256; // Tail insertions
    }
}

int main(int argc, char**argv) {

    // Parse arguments
    unsigned int N = (argc > 1)?(atoi(argv[1])):(32000);

    // Allocate memory and initialize data
    Timer timer;
    unsigned char* reference = (unsigned char*) malloc(N*sizeof(unsigned char));
    unsigned char* query = (unsigned char*) malloc(N*sizeof(unsigned char));
    int* matrix_cpu = (int*) malloc(N*N*sizeof(int));
    for(unsigned int r = 0; r < N; ++r) {
        reference[r] = rand()%256;
    }
    generateQuery(reference, query, N);

    // Compute on CPU
    startTime(&timer);
    nw_cpu(reference, query, matrix_cpu, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);
    
    free(matrix_cpu);

    return 0;

}


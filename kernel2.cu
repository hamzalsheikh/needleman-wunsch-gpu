#include "common.h"
#include "timer.h"
#include <cuda_profiler_api.h>

#define COARSE_FACTOR 2 
#define BLOCK_DIM 512

__global__ void nw_upper_left_kernel2(unsigned char* reference, unsigned char* query, int* matrix, unsigned int N, int verticalBlockOffset, int horizontalBlockOffset) {

    int threadArray[COARSE_FACTOR];
    const int dim = BLOCK_DIM*COARSE_FACTOR;
    __shared__ int leftCol_s[dim];
    __shared__ int upperRow_s[dim];
    __shared__ int diagonal1_s[dim];
    __shared__ int diagonal2_s[dim];
    int* ptrDiagonal1_s = diagonal1_s;
    int* ptrDiagonal2_s = diagonal2_s;

    int verticalOffset = verticalBlockOffset*dim;
    int horizontalOffset = horizontalBlockOffset*dim;

    int startRow = verticalOffset - blockIdx.x*dim;
    int startCol = horizontalOffset + blockIdx.x*dim;

    for (int i = 0; i < COARSE_FACTOR ; i++){
    // Load the row at the top of the triangle into shared memory
        if (startCol + threadIdx.x + i*BLOCK_DIM < N && startRow != 0) {
                upperRow_s[threadIdx.x + i*BLOCK_DIM] = matrix[(startRow - 1)*N + startCol + threadIdx.x + i*BLOCK_DIM];
        }
    }
    for (int i = 0; i < COARSE_FACTOR ; ++i){
    // Load the column to the left of the triangle into shared memory
        if (startRow + threadIdx.x + i*BLOCK_DIM < N && startCol != 0) {
                leftCol_s[threadIdx.x + i*BLOCK_DIM] = matrix[(startRow + threadIdx.x + i*BLOCK_DIM)*N + startCol - 1];
        }
    }
    __syncthreads();
    for (int i = 0; i < blockDim.x*COARSE_FACTOR; i++){
        int diagStartRow = startRow + i;
        int diagStartCol = startCol;
        for (int j = 0; j < COARSE_FACTOR; j++){
                int outRow = diagStartRow - (threadIdx.x*COARSE_FACTOR + j);
                int outCol = diagStartCol + (threadIdx.x*COARSE_FACTOR + j);

                if (threadIdx.x*COARSE_FACTOR + j <= i && outRow < N && outCol < N) {
                        // Get neighbors
                        int m = threadIdx.x * COARSE_FACTOR + j;
                        int top = (outRow == 0)?((outCol + 1)*DELETION):(outRow == startRow ? upperRow_s[m] : ptrDiagonal2_s[m]);
                        int left = (outCol == 0) ? ((outRow + 1)*INSERTION) : (threadIdx.x == 0 && j == 0 ? leftCol_s[i] : (ptrDiagonal2_s[m - 1]));
                        int topleft = (outRow == 0) ? (outCol*DELETION) :
                        ((outCol == 0) ? (outRow*INSERTION) :
                        ((threadIdx.x == 0 && i == 0) ? (matrix[(outRow - 1)* N + outCol - 1]) :
                        ((threadIdx.x == 0 && j == 0) ? (leftCol_s[i-1]) :
                        ((outRow == startRow) ? (upperRow_s[m - 1]) : ptrDiagonal1_s[m -1]))));

                        // Find scores based on neighbors
                        int insertion = top + INSERTION;
                        int deletion = left + DELETION;
                        int match = topleft + ((query[outRow] == reference[outCol])?MATCH:MISMATCH);

                        // Select best score
                        int max = (insertion > deletion)?insertion:deletion;
                        max = (match > max)?match:max;
                        matrix[outRow *N + outCol] = max;
                        threadArray[j] = max;
                }
        }
        __syncthreads();

        // Update the diagonals and swap them
        for (int k = 0; k < COARSE_FACTOR; ++k){
                int outRow = diagStartRow - (threadIdx.x*COARSE_FACTOR + k);
                int outCol = diagStartCol + (threadIdx.x*COARSE_FACTOR + k);
                if (threadIdx.x*COARSE_FACTOR + k <= i && outRow < N && outCol < N) {
                        ptrDiagonal1_s[threadIdx.x*COARSE_FACTOR + k] = threadArray[k];
                }
        }
        __syncthreads();

        int* temp = ptrDiagonal1_s;
        ptrDiagonal1_s = ptrDiagonal2_s;
        ptrDiagonal2_s = temp;

        __syncthreads();
    }
}

__global__ void nw_lower_right_kernel2(unsigned char* reference, unsigned char* query, int* matrix, unsigned int N, int verticalBlockOffset, int horizontalBlockOffset) {

    int threadArray[COARSE_FACTOR];

    __shared__ int diagonal1_s[BLOCK_DIM*COARSE_FACTOR];
    __shared__ int diagonal2_s[BLOCK_DIM*COARSE_FACTOR];
    int* ptrDiagonal1_s = diagonal1_s;
    int* ptrDiagonal2_s = diagonal2_s;

    int verticalOffset = verticalBlockOffset*blockDim.x*COARSE_FACTOR;
    int horizontalOffset = horizontalBlockOffset*blockDim.x*COARSE_FACTOR;

    int startRow = verticalOffset + blockDim.x*COARSE_FACTOR - 1 - blockIdx.x*blockDim.x*COARSE_FACTOR;
    int startCol = horizontalOffset + 1 + blockIdx.x*blockDim.x*COARSE_FACTOR;

    // Load the values that come before the first diagonal into shared memory
    for (int s = 0 ; s < COARSE_FACTOR ; ++s){
        if ((startRow - (threadIdx.x*COARSE_FACTOR + s) >= 0 && startRow - (threadIdx.x*COARSE_FACTOR + s) < N) && startCol - 1 + threadIdx.x*COARSE_FACTOR + s < N) {
                ptrDiagonal1_s[threadIdx.x*COARSE_FACTOR + s] = matrix[(startRow - (threadIdx.x*COARSE_FACTOR +s))*N + (startCol - 1 + (threadIdx.x*COARSE_FACTOR +s))];
        }
    }
    __syncthreads();

    for (int i = blockDim.x*COARSE_FACTOR - 2; i >= 0; i--) {
        int diagStartRow = startRow;
        int diagStartCol = startCol + (blockDim.x*COARSE_FACTOR - 2 - i);

        for (int k = 0 ; k < COARSE_FACTOR ; k++){
                int outRow = diagStartRow - (threadIdx.x*COARSE_FACTOR + k);
                int outCol = diagStartCol + (threadIdx.x*COARSE_FACTOR + k);

                if (threadIdx.x*COARSE_FACTOR + k <= i && outRow < N && outCol < N) {

                // Get neighbors
                int m = threadIdx.x * COARSE_FACTOR + k;
                int top = (outRow == 0)?((outCol + 1)*DELETION):(i == blockDim.x*COARSE_FACTOR - 2 ? ptrDiagonal1_s[m + 1] : ptrDiagonal2_s[m + 1]);
                int left = (outCol == 0)?((outRow + 1)*INSERTION):((i == blockDim.x*COARSE_FACTOR - 2) ? ptrDiagonal1_s[m] : ptrDiagonal2_s[m]);

                int topleft = (outRow == 0)?(outCol*DELETION):
                ((outCol == 0)?(outRow*INSERTION):
                ((i == blockDim.x*COARSE_FACTOR - 2) ? (matrix[(outRow - 1)*N + (outCol - 1)]):
                (ptrDiagonal1_s[m + 1])));

                // Find scores based on neighbors
                int insertion = top + INSERTION;
                int deletion = left + DELETION;
                int match = topleft + ((query[outRow] == reference[outCol])?MATCH:MISMATCH);

                // Select best score
                int max = (insertion > deletion)?insertion:deletion;
                max = (match > max)?match:max;
                matrix[outRow*N + outCol] = max;
                threadArray[k] = max;
                }
        }
        __syncthreads();


        if (i == blockDim.x*COARSE_FACTOR - 2) {
            for (int t = 0; t < COARSE_FACTOR; ++t){
                int outRow = diagStartRow - (threadIdx.x*COARSE_FACTOR + t);
                int outCol = diagStartCol + (threadIdx.x*COARSE_FACTOR + t);
                // First diagonal writes directly in diagonal 2
                if (threadIdx.x*COARSE_FACTOR + t <= i && outRow < N && outCol < N) {
                        ptrDiagonal2_s[threadIdx.x*COARSE_FACTOR + t] = threadArray[t];
                }
            }
        }
        else {
            for (int t = 0; t < COARSE_FACTOR; ++t){
                int outRow = diagStartRow - (threadIdx.x*COARSE_FACTOR + t);
                int outCol = diagStartCol + (threadIdx.x*COARSE_FACTOR + t);
                // The rest of the diagonals write in diagonal 1 and then
                // the pointers are swapped
                if (threadIdx.x*COARSE_FACTOR + t <= i && outRow < N && outCol < N) {
                        ptrDiagonal1_s[threadIdx.x*COARSE_FACTOR + t] = threadArray[t];
                }
         }
          __syncthreads();

            int* temp = ptrDiagonal1_s;
            ptrDiagonal1_s = ptrDiagonal2_s;
            ptrDiagonal2_s = temp;
        }
        __syncthreads();
    }
}


void nw_gpu2(unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N) {

    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int sizeOfTriangle = numThreadsPerBlock*COARSE_FACTOR;
    const unsigned int numKernelCalls = (N - 1 + sizeOfTriangle)/sizeOfTriangle;

    for (int i = 0; i < numKernelCalls; i++) {
        unsigned int numBlocks = i + 1;

        // First call concerns the upper left triangles
        nw_upper_left_kernel2<<< numBlocks, numThreadsPerBlock >>>(reference_d, query_d, matrix_d, N, i, 0);

        cudaDeviceSynchronize();

        // Second call concerns the lower right triangles
        nw_lower_right_kernel2<<< numBlocks, numThreadsPerBlock >>>(reference_d, query_d, matrix_d, N, i, 0);

        cudaDeviceSynchronize();
    }

    for (int i = numKernelCalls - 2; i >= 0; i--) {
        unsigned int numBlocks = i + 1;
        int horizontalOffset = numKernelCalls - 1 - i;

        // Same as above; first call concerns the upper left triangles
        nw_upper_left_kernel2<<< numBlocks, numThreadsPerBlock >>>(reference_d, query_d, matrix_d, N, numKernelCalls - 1, horizontalOffset);

        cudaDeviceSynchronize();

        // Second call concerns the lower right triangles
        nw_lower_right_kernel2<<< numBlocks, numThreadsPerBlock >>>(reference_d, query_d, matrix_d, N, numKernelCalls - 1, horizontalOffset);

        cudaDeviceSynchronize();
    }
}


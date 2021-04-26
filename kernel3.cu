#include "common.h"
#include "timer.h"

#define BLOCK_DIM 1024

__global__ void nw_upper_left_kernel3(unsigned char* reference, unsigned char* query, int* matrix, unsigned int N, int verticalBlockOffset, int horizontalBlockOffset) {

    __shared__ int leftCol_s[BLOCK_DIM];
    __shared__ int upperRow_s[BLOCK_DIM];
    __shared__ int diagonal1_s[BLOCK_DIM];
    __shared__ int diagonal2_s[BLOCK_DIM];
    int* ptrDiagonal1_s = diagonal1_s;
    int* ptrDiagonal2_s = diagonal2_s;

    int verticalOffset = verticalBlockOffset*blockDim.x;
    int horizontalOffset = horizontalBlockOffset*blockDim.x;

    int startRow = verticalOffset - blockIdx.x*blockDim.x;
    int startCol = horizontalOffset + blockIdx.x*blockDim.x;

    // Load the row at the top of the triangle into shared memory
    if (startCol + threadIdx.x < N && startRow != 0) {
        upperRow_s[threadIdx.x] = matrix[(startRow - 1)*N + startCol + threadIdx.x];
    }

    // Load the column to the left of the triangle into shared memory
    if (startRow + threadIdx.x < N && startCol != 0) {
        leftCol_s[threadIdx.x] = matrix[(startRow + threadIdx.x)*N + startCol - 1];
     }

    __syncthreads();
    for (int i = 0; i < blockDim.x; i++) {

        int diagStartRow = startRow + i;
        int diagStartCol = startCol;
        int outRow = diagStartRow - threadIdx.x;
        int outCol = diagStartCol + threadIdx.x;
        // Arbitrary value just for the code to compile
        int max = -1;

        if (threadIdx.x <= i && outRow < N && outCol < N) {
            // Get neighbors
            int top = (outRow == 0)?((outCol + 1)*DELETION):(outRow == startRow ? upperRow_s[threadIdx.x] : ptrDiagonal2_s[threadIdx.x]);
            int left = (outCol == 0) ? ((outRow + 1)*INSERTION) : (threadIdx.x== 0 ? leftCol_s[i] : (ptrDiagonal2_s[threadIdx.x - 1]));
            int topleft = (outRow == 0) ? (outCol*DELETION) :
((outCol == 0) ? (outRow*INSERTION) :
((threadIdx.x == 0 && i == 0) ? (matrix[(outRow - 1)* N + outCol - 1]) :
((threadIdx.x == 0) ? (leftCol_s[i-1]) :
((outRow == startRow) ? (upperRow_s[threadIdx.x - 1]) : ptrDiagonal1_s[threadIdx.x -1]))));

            // Find scores based on neighbors
            int insertion = top + INSERTION;
            int deletion = left + DELETION;
            int match = topleft + ((query[outRow] == reference[outCol])?MATCH:MISMATCH);

            // Select best score
            max = (insertion > deletion)?insertion:deletion;
            max = (match > max)?match:max;
            matrix[outRow *N + outCol] = max;
        }
        __syncthreads();

        // Update the diagonals and swap them
        if (threadIdx.x <= i && outRow < N && outCol < N) {
            ptrDiagonal1_s[threadIdx.x] = max;
        }

        __syncthreads();

        int* temp = ptrDiagonal1_s;
        ptrDiagonal1_s = ptrDiagonal2_s;
        ptrDiagonal2_s = temp;

        __syncthreads();
    }
}

__global__ void nw_lower_right_kernel3(unsigned char* reference, unsigned char* query, int* matrix, unsigned int N, int verticalBlockOffset, int horizontalBlockOffset) {

    __shared__ int diagonal1_s[BLOCK_DIM];
    __shared__ int diagonal2_s[BLOCK_DIM];
    int* ptrDiagonal1_s = diagonal1_s;
    int* ptrDiagonal2_s = diagonal2_s;

    int verticalOffset = verticalBlockOffset*blockDim.x;
    int horizontalOffset = horizontalBlockOffset*blockDim.x;

    int startRow = verticalOffset + blockDim.x - 1 - blockIdx.x*blockDim.x;
    int startCol = horizontalOffset + 1 + blockIdx.x*blockDim.x;

    // Load the values that come before the first diagonal into shared memory
    if ((startRow - threadIdx.x >= 0 && startRow - threadIdx.x < N) && startCol - 1 + threadIdx.x < N) {
        ptrDiagonal1_s[threadIdx.x] = matrix[(startRow - threadIdx.x)*N + (startCol - 1 + threadIdx.x)];
    }

    __syncthreads();

    for (int i = blockDim.x - 2; i >= 0; i--) {
        int diagStartRow = startRow;
        int diagStartCol = startCol + (blockDim.x - 2 - i);
        int outRow = diagStartRow - threadIdx.x;
        int outCol = diagStartCol + threadIdx.x;
        // Arbitrary value just for the code to compile
        int max = -1;

        if (threadIdx.x <= i && outRow < N && outCol < N) {

            // Get neighbors
            int top = (outRow == 0)?((outCol + 1)*DELETION):(i == blockDim.x - 2 ? ptrDiagonal1_s[threadIdx.x + 1] : ptrDiagonal2_s[threadIdx.x + 1]);
            int left = (outCol == 0)?((outRow + 1)*INSERTION):((i == blockDim.x - 2) ? ptrDiagonal1_s[threadIdx.x] : ptrDiagonal2_s[threadIdx.x]);

            int topleft = (outRow == 0)?(outCol*DELETION):
((outCol == 0)?(outRow*INSERTION):
((i == blockDim.x - 2) ? (matrix[(outRow - 1)*N + (outCol - 1)]):
(ptrDiagonal1_s[threadIdx.x + 1])));

            // Find scores based on neighbors
            int insertion = top + INSERTION;
            int deletion = left + DELETION;
            int match = topleft + ((query[outRow] == reference[outCol])?MATCH:MISMATCH);

            // Select best score
            max = (insertion > deletion)?insertion:deletion;
            max = (match > max)?match:max;
            matrix[outRow*N + outCol] = max;

        }
        __syncthreads();

        if (i == blockDim.x - 2) {
            // First diagonal writes directly in diagonal 2
            if (threadIdx.x <= i && outRow < N && outCol < N) {
                ptrDiagonal2_s[threadIdx.x] = max;
            }
        }
        else {
            // The rest of the diagonals write in diagonal 1 and then 
            // the pointers are swapped
            if (threadIdx.x <= i && outRow < N && outCol < N) {
                ptrDiagonal1_s[threadIdx.x] = max;
            }

            __syncthreads();

            int* temp = ptrDiagonal1_s;
            ptrDiagonal1_s = ptrDiagonal2_s;
            ptrDiagonal2_s = temp;
        }
        __syncthreads();
    }
}


void nw_gpu3(unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N) {

    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numKernelCalls = (N - 1 + numThreadsPerBlock)/numThreadsPerBlock;

    for (int i = 0; i < numKernelCalls; i++) {
        unsigned int numBlocks = i + 1;

        // First call concerns the upper left triangles
        nw_upper_left_kernel3<<< numBlocks, numThreadsPerBlock >>>(reference_d, query_d, matrix_d, N, i, 0);

        cudaDeviceSynchronize();

        // Second call concerns the lower right triangles
        nw_lower_right_kernel3<<< numBlocks, numThreadsPerBlock >>>(reference_d, query_d, matrix_d, N, i, 0);

        cudaDeviceSynchronize();
    }

    for (int i = numKernelCalls - 2; i >= 0; i--) {
        unsigned int numBlocks = i + 1;
        int horizontalOffset = numKernelCalls - 1 - i;

        // Same as above; first call concerns the upper left triangles
        nw_upper_left_kernel3<<< numBlocks, numThreadsPerBlock >>>(reference_d, query_d, matrix_d, N, numKernelCalls - 1, horizontalOffset);

        cudaDeviceSynchronize();
        
        // Second call concerns the lower right triangles
        nw_lower_right_kernel3<<< numBlocks, numThreadsPerBlock >>>(reference_d, query_d, matrix_d, N, numKernelCalls - 1, horizontalOffset);

        cudaDeviceSynchronize();
    }
}


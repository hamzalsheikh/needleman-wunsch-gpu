
#include "common.h"
#include "timer.h"

__global__ void nw_upper_left_kernel(unsigned char* reference, unsigned char* query, int* matrix, unsigned int N, int verticalBlockOffset, int horizontalBlockOffset) {

    int verticalOffset = verticalBlockOffset*blockDim.x;
    int horizontalOffset = horizontalBlockOffset*blockDim.x;

    int startRow = verticalOffset - blockIdx.x*blockDim.x;
    int startCol = horizontalOffset + blockIdx.x*blockDim.x;

    for (int i = 0; i < blockDim.x; i++) {
        int diagStartRow = startRow + i;
        int diagStartCol = startCol;
        if (threadIdx.x <= i) {
            int outRow = diagStartRow - threadIdx.x;
            int outCol = diagStartCol + threadIdx.x;

            if (outRow < N && outCol < N) {

                // Get neighbors
                int top = (outRow == 0)?((outCol + 1)*DELETION):(matrix[(outRow - 1)*N + outCol]);
                int left = (outCol == 0)?((outRow + 1)*INSERTION):(matrix[outRow*N + (outCol - 1)]);
                int topleft = (outRow == 0)?(outCol*DELETION):((outCol == 0)?(outRow*INSERTION):(matrix[(outRow - 1)*N + (outCol - 1)]));

                // Find scores based on neighbors
                int insertion = top + INSERTION;
                int deletion = left + DELETION;
                int match = topleft + ((query[outRow] == reference[outCol])?MATCH:MISMATCH);

                // Select best score
                int max = (insertion > deletion)?insertion:deletion;
                max = (match > max)?match:max;
                matrix[outRow*N + outCol] = max;
            }
        }
        __syncthreads();
    }
}

__global__ void nw_lower_right_kernel(unsigned char* reference, unsigned char* query, int* matrix, unsigned int N, int verticalBlockOffset, int horizontalBlockOffset) {

    int verticalOffset = verticalBlockOffset*blockDim.x;
    int horizontalOffset = horizontalBlockOffset*blockDim.x;

    int startRow = verticalOffset + blockDim.x - 1 - blockIdx.x*blockDim.x;
    int startCol = horizontalOffset + 1 + blockIdx.x*blockDim.x;

    for (int i = blockDim.x - 2; i >= 0; i--) {
        int diagStartRow = startRow;
        int diagStartCol = startCol + (blockDim.x - 2 - i);
        if (threadIdx.x <= i) {
            int outRow = diagStartRow - threadIdx.x;
            int outCol = diagStartCol + threadIdx.x;

            if (outRow < N && outCol < N) {

                // Get neighbors
                int top = (outRow == 0)?((outCol + 1)*DELETION):(matrix[(outRow - 1)*N + outCol]);
                int left = (outCol == 0)?((outRow + 1)*INSERTION):(matrix[outRow*N + (outCol - 1)]);
                int topleft = (outRow == 0)?(outCol*DELETION):((outCol == 0)?(outRow*INSERTION):(matrix[(outRow - 1)*N + (outCol - 1)]));

                // Find scores based on neighbors
                int insertion = top + INSERTION;
                int deletion = left + DELETION;
                int match = topleft + ((query[outRow] == reference[outCol])?MATCH:MISMATCH);

                // Select best score
                int max = (insertion > deletion)?insertion:deletion;
                max = (match > max)?match:max;
                matrix[outRow*N + outCol] = max;
            }
        }
        __syncthreads();
    }
}

void nw_gpu0(unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N) {

    const unsigned int numThreadsPerBlock = 1024;
    const unsigned int numKernelCalls = (N - 1 + numThreadsPerBlock)/numThreadsPerBlock;

    for (int i = 0; i < numKernelCalls; i++) {
        unsigned int numBlocks = i + 1;

        // First call concerns the upper left triangles
        nw_upper_left_kernel<<< numBlocks, numThreadsPerBlock >>>(reference_d, query_d, matrix_d, N, i, 0);
		
		cudaDeviceSynchronize();

        // Second call concerns the lower right triangles
        nw_lower_right_kernel<<< numBlocks, numThreadsPerBlock >>>(reference_d, query_d, matrix_d, N, i, 0);
		
		cudaDeviceSynchronize();
    }
 
    for (int i = numKernelCalls - 2; i >= 0; i--) {
        unsigned int numBlocks = i + 1;
        int horizontalOffset = numKernelCalls - 1 - i;

        // Same as above; first call concerns the upper left triangles
        nw_upper_left_kernel<<< numBlocks, numThreadsPerBlock >>>(reference_d, query_d, matrix_d, N, numKernelCalls - 1, horizontalOffset);
		
		cudaDeviceSynchronize();

        // Second call concerns the lower right triangles 
        nw_lower_right_kernel<<< numBlocks, numThreadsPerBlock >>>(reference_d, query_d, matrix_d, N, numKernelCalls - 1, horizontalOffset);
		
		cudaDeviceSynchronize();
    }
}


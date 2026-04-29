#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define BLURSIZE 1

__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col<w && row<h){
        int pixVal = 0;
        int pixels = 0

        for(int blurRow = -BLURSIZE; blurRow<BLURSIZE+1; blurRow++){
            for(int blurCol = -BLURSIZE; blurCol<BLURSIZE+1; blurCol++){
                int currRow = row + blurRow;
                int currCol = col + blurCol;

                if(currRow>=0 && currRow<h && currCol>=0 && currCol<w){
                    pixVal += in[currRow * w + currCol];
                    ++pixels;
                }

            }
        }
        out[row * w + col] = pixVal / pixels;
    }
}

int main(int argc, char **argv){
    const char *inpath = (argc >= 2) ? argv[1] : "image.png";
    const char *outpath = (argc >= 3) ? argv[2] : "image_blur.png";

    unsigned char *h_in = NULL;
    int w, h;
    image_heap_kind heap_kind = IMAGE_HEAP_MALLOC;
    if (image_load_rgb(inpath, &h_in, &w, &h, &heap_kind) != 0) {
        fprintf(stderr, "Could not load image: %s\n", inpath);
        return 1;
    }

    size_t npix = (size_t)w * (size_t)h;
    size_t nbytes_rgb = npix * 3;
    unsigned char *h_blur = (unsigned char *)malloc(npix);
    if (!h_blur) {
        image_free_rgb(h_in, heap_kind);
        fprintf(stderr, "Out of memory.\n");
        return 1;
    }

    unsigned char *d_in = NULL;
    unsigned char *d_out = NULL;
    cudaMalloc(&d_in, nbytes_rgb);
    cudaMalloc(&d_out, npix);
    cudaMemcpy(d_in, h_in, nbytes_rgb, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    blurKernel<<<grid, block>>>(d_in, d_out, w, h);
    cudaError_t err = cudaDeviceSynchronize();  
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel failed: %s\n", cudaGetErrorString(err));
        image_free_rgb(h_in, heap_kind);
        free(h_blur);
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }

    cudaMemcpy(h_blur, d_out, npix, cudaMemcpyDeviceToHost);

    image_save_rgb(outpath, h_blur, w, h, heap_kind);
    image_free_rgb(h_in, heap_kind);
    free(h_blur);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

__global__
void colortoGrayscale(unsigned char *input, unsigned char *output, int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < w && row < h) {
        int pixelIndex = row * w + col;
        int rgbPixelIndex = pixelIndex * 3;
        unsigned char r = input[rgbPixelIndex];
        unsigned char g = input[rgbPixelIndex + 1];
        unsigned char b = input[rgbPixelIndex + 2];
        float y = r * 0.299f + g * 0.587f + b * 0.114f;
        output[pixelIndex] = (unsigned char)(y + 0.5f);
    }
}

static int read_ppm_p3(const char *path, unsigned char **rgb_out, int *width, int *height) {
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    char magic[4] = {0};
    if (fscanf(f, "%3s", magic) != 1 || strcmp(magic, "P3") != 0) {
        fclose(f);
        return -1;
    }
    int w, h, maxv;
    if (fscanf(f, "%d %d %d", &w, &h, &maxv) != 3) {
        fclose(f);
        return -1;
    }
    unsigned char *buf = (unsigned char *)malloc((size_t)w * h * 3);
    if (!buf) {
        fclose(f);
        return -1;
    }
    for (int i = 0; i < w * h; i++) {
        int r, g, b;
        if (fscanf(f, "%d %d %d", &r, &g, &b) != 3) {
            free(buf);
            fclose(f);
            return -1;
        }
        buf[i * 3 + 0] = (unsigned char)r;
        buf[i * 3 + 1] = (unsigned char)g;
        buf[i * 3 + 2] = (unsigned char)b;
    }
    fclose(f);
    *rgb_out = buf;
    *width = w;
    *height = h;
    return 0;
}

static int write_pgm_p5(const char *path, const unsigned char *gray, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;
    fprintf(f, "P5\n%d %d\n255\n", w, h);
    if (fwrite(gray, 1, (size_t)w * h, f) != (size_t)w * h) {
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}

#define BLOCK_X 16
#define BLOCK_Y 16

int main(int argc, char **argv) {
    const char *inpath = (argc >= 2) ? argv[1] : "dummy_image.ppm";
    const char *outpath = (argc >= 3) ? argv[2] : "dummy_image_gray.pgm";

    unsigned char *h_rgb = NULL;
    int w, h;
    if (read_ppm_p3(inpath, &h_rgb, &w, &h) != 0) {
        fprintf(stderr, "Could not read PPM image: %s\n", inpath);
        return 1;
    }

    size_t npix = (size_t)w * (size_t)h;
    size_t nbytes_rgb = npix * 3;
    unsigned char *h_gray = (unsigned char *)malloc(npix);
    if (!h_gray) {
        free(h_rgb);
        fprintf(stderr, "Out of memory.\n");
        return 1;
    }

    unsigned char *d_in = NULL;
    unsigned char *d_out = NULL;
    cudaMalloc(&d_in, nbytes_rgb);
    cudaMalloc(&d_out, npix);
    cudaMemcpy(d_in, h_rgb, nbytes_rgb, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((w + BLOCK_X - 1) / BLOCK_X, (h + BLOCK_Y - 1) / BLOCK_Y);
    colortoGrayscale<<<grid, block>>>(d_in, d_out, w, h);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel failed: %s\n", cudaGetErrorString(err));
        free(h_rgb);
        free(h_gray);
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }

    cudaMemcpy(h_gray, d_out, npix, cudaMemcpyDeviceToHost);

    if (write_pgm_p5(outpath, h_gray, w, h) != 0) {
        fprintf(stderr, "Could not write PGM: %s\n", outpath);
        free(h_rgb);
        free(h_gray);
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }

    printf("Read %s (%d x %d), wrote grayscale %s\n", inpath, w, h, outpath);

    free(h_rgb);
    free(h_gray);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

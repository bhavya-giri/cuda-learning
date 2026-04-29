#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BLURSIZE 1

__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int C = 3;

    if (col < w && row < h) {
        int idx = row * w + col;
        for (int ch = 0; ch < C; ++ch) {
            int pixVal = 0;
            int pixels = 0;
            for (int blurRow = -BLURSIZE; blurRow < BLURSIZE + 1; blurRow++) {
                for (int blurCol = -BLURSIZE; blurCol < BLURSIZE + 1; blurCol++) {
                    int currRow = row + blurRow;
                    int currCol = col + blurCol;
                    if (currRow >= 0 && currRow < h && currCol >= 0 && currCol < w) {
                        int i = currRow * w + currCol;
                        pixVal += in[i * C + ch];
                        ++pixels;
                    }
                }
            }
            out[idx * C + ch] = (unsigned char)(pixVal / pixels);
        }
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
    int wa, hb, maxv;
    if (fscanf(f, "%d %d %d", &wa, &hb, &maxv) != 3) {
        fclose(f);
        return -1;
    }
    unsigned char *buf = (unsigned char *)malloc((size_t)wa * hb * 3);
    if (!buf) {
        fclose(f);
        return -1;
    }
    for (int i = 0; i < wa * hb; i++) {
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
    *width = wa;
    *height = hb;
    return 0;
}

/** STB: load packed RGB (3 channels). Free with stbi_image_free when loaded via STB. */
static int load_rgb_via_stb(const char *path, unsigned char **rgb_out, int *width, int *height) {
    int comp = 0;
    unsigned char *img = stbi_load(path, width, height, &comp, 3);
    if (!img) {
        fprintf(stderr, "stbi_load failed (%s)\n", stbi_failure_reason());
        return -1;
    }
    *rgb_out = img;
    return 0;
}

static int save_rgb_png(const char *path, const unsigned char *rgb, int w, int h) {
    return stbi_write_png(path, w, h, 3, rgb, w * 3) ? 0 : -1;
}

static int path_suffix_icase(const char *path, const char *suffix_with_dot) {
    const char *dot = strrchr(path, '.');
    return dot != NULL && strcasecmp(dot, suffix_with_dot) == 0;
}

typedef enum { IMAGE_HEAP_MALLOC, IMAGE_HEAP_STBI } image_heap_kind;

static void image_free_rgb(unsigned char *p, image_heap_kind heap) {
    if (!p)
        return;
    if (heap == IMAGE_HEAP_STBI)
        stbi_image_free(p);
    else
        free(p);
}

static int image_load_rgb(const char *path, unsigned char **rgb_out, int *width, int *height,
                          image_heap_kind *heap_kind) {
    if (path_suffix_icase(path, ".ppm"))
        return read_ppm_p3(path, rgb_out, width, height) ? -1 : (*heap_kind = IMAGE_HEAP_MALLOC, 0);

    int r = load_rgb_via_stb(path, rgb_out, width, height);
    if (r == 0)
        *heap_kind = IMAGE_HEAP_STBI;
    return r;
}

static int image_save_rgb_file(const char *path, const unsigned char *rgb, int w, int h) {
    if (path_suffix_icase(path, ".png"))
        return save_rgb_png(path, rgb, w, h);
    fprintf(stderr, "unsupported output suffix for RGB (use .png)\n");
    return -1;
}

#define BLOCK_X 16
#define BLOCK_Y 16

int main(int argc, char **argv) {
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
    unsigned char *h_blur = (unsigned char *)malloc(nbytes_rgb);
    if (!h_blur) {
        image_free_rgb(h_in, heap_kind);
        fprintf(stderr, "Out of memory.\n");
        return 1;
    }

    unsigned char *d_in = NULL;
    unsigned char *d_out = NULL;
    cudaMalloc(&d_in, nbytes_rgb);
    cudaMalloc(&d_out, nbytes_rgb);
    cudaMemcpy(d_in, h_in, nbytes_rgb, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((w + BLOCK_X - 1) / BLOCK_X, (h + BLOCK_Y - 1) / BLOCK_Y);
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

    cudaMemcpy(h_blur, d_out, nbytes_rgb, cudaMemcpyDeviceToHost);

    if (image_save_rgb_file(outpath, h_blur, w, h) != 0) {
        fprintf(stderr, "Could not write output: %s\n", outpath);
        image_free_rgb(h_in, heap_kind);
        free(h_blur);
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }

    printf("Read %s (%d x %d), wrote blurred RGB %s\n", inpath, w, h, outpath);

    image_free_rgb(h_in, heap_kind);
    free(h_blur);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

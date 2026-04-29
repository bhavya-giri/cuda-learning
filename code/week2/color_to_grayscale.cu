#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

/** STB: load file as packed RGB (3 channels). Free with stbi_image_free. */
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

/** Write one channel 8-bit gray as PNG; stride equals width bytes. */
static int save_gray_png(const char *path, const unsigned char *gray, int w, int h) {
    if (!stbi_write_png(path, w, h, 1, gray, w)) {
        return -1;
    }
    return 0;
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

/** PPM ASCII (P3) uses malloc; any other suffix uses STB loaders (PNG, JPEG, BMP, GIF, PSD, PIC, HDR, TGA…). */
static int image_load_rgb(const char *path, unsigned char **rgb_out, int *width, int *height,
                            image_heap_kind *heap_kind) {
    if (path_suffix_icase(path, ".ppm"))
        return read_ppm_p3(path, rgb_out, width, height) ? -1 : (*heap_kind = IMAGE_HEAP_MALLOC, 0);

    int r = load_rgb_via_stb(path, rgb_out, width, height);
    if (r == 0)
        *heap_kind = IMAGE_HEAP_STBI;
    return r;
}

static int image_save_gray_file(const char *path, const unsigned char *gray, int w, int h) {
    if (path_suffix_icase(path, ".pgm"))
        return write_pgm_p5(path, gray, w, h);
    if (path_suffix_icase(path, ".png"))
        return save_gray_png(path, gray, w, h);

    fprintf(stderr, "unsupported output suffix (use .png or .pgm)\n");
    return -1;
}

#define BLOCK_X 16
#define BLOCK_Y 16

int main(int argc, char **argv) {
    const char *inpath = (argc >= 2) ? argv[1] : "image.png";
    const char *outpath = (argc >= 3) ? argv[2] : "image_gray.png";

    unsigned char *h_rgb = NULL;
    int w, h;
    image_heap_kind heap_kind = IMAGE_HEAP_MALLOC;
    if (image_load_rgb(inpath, &h_rgb, &w, &h, &heap_kind) != 0) {
        fprintf(stderr, "Could not load image: %s\n", inpath);
        return 1;
    }

    size_t npix = (size_t)w * (size_t)h;
    size_t nbytes_rgb = npix * 3;
    unsigned char *h_gray = (unsigned char *)malloc(npix);
    if (!h_gray) {
        image_free_rgb(h_rgb, heap_kind);
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
        image_free_rgb(h_rgb, heap_kind);
        free(h_gray);
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }

    cudaMemcpy(h_gray, d_out, npix, cudaMemcpyDeviceToHost);

    if (image_save_gray_file(outpath, h_gray, w, h) != 0) {
        fprintf(stderr, "Could not write output: %s\n", outpath);
        image_free_rgb(h_rgb, heap_kind);
        free(h_gray);
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }

    printf("Read %s (%d x %d), wrote grayscale %s\n", inpath, w, h, outpath);

    image_free_rgb(h_rgb, heap_kind);
    free(h_gray);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

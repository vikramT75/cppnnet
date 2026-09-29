#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cmath>

#define NN_IMPLEMENTATION
#include "../nn.hpp"

#define IMG_SIZE    28
#define NUM_INPUTS  (IMG_SIZE * IMG_SIZE)
#define NUM_CLASSES 3
#define NUM_COLS    (NUM_INPUTS + NUM_CLASSES)

#define NUM_TRAIN 50000
#define NUM_TEST   7000

#define CLASS_CIRCLE   0
#define CLASS_SQUARE   1
#define CLASS_TRIANGLE 2

#ifndef M_PI
#define M_PI 3.14159265359f
#endif

static void save_mat(FILE *out, Mat m)
{
    const char *magic = "nn.h.mat";
    fwrite(magic, strlen(magic), 1, out);
    fwrite(&m.rows, sizeof(m.rows), 1, out);
    fwrite(&m.cols, sizeof(m.cols), 1, out);
    for (size_t i = 0; i < m.rows; i++)
        fwrite(&MAT_AT(m, i, 0), sizeof(float), m.cols, out);
}

static float randf_range(float lo, float hi)
{
    return lo + ((float)rand() / (float)RAND_MAX) * (hi - lo);
}

static int point_in_triangle(float px, float py,
                              float x0, float y0,
                              float x1, float y1,
                              float x2, float y2)
{
    float d1 = (px - x1)*(y0 - y1) - (x0 - x1)*(py - y1);
    float d2 = (px - x2)*(y1 - y2) - (x1 - x2)*(py - y2);
    float d3 = (px - x0)*(y2 - y0) - (x2 - x0)*(py - y0);
    int has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0);
    int has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0);
    return !(has_neg & has_pos);
}

static void generate_sample(float pixels[NUM_INPUTS], int class_id)
{
    float target_size = randf_range(10.0f, 22.0f);
    float r = (class_id == CLASS_TRIANGLE) ? target_size/1.73205081f : target_size/2.0f;
    float spread  = (24.0f - target_size) / 2.0f;
    float cx      = randf_range(14.0f - spread, 14.0f + spread);
    float cy      = randf_range(14.0f - spread, 14.0f + spread);
    float theta   = randf_range(0.0f, 2.0f * (float)M_PI);
    float c_theta = cosf(theta), s_theta = sinf(theta);
    int   is_hollow = (rand() % 2 == 0);
    float inner_r   = r - randf_range(1.5f, 3.0f);
    if (inner_r < 1.0f) inner_r = 1.0f;

    for (int y = 0; y < IMG_SIZE; y++)
        for (int x = 0; x < IMG_SIZE; x++)
        {
            float px = (float)x + 0.5f, py = (float)y + 0.5f;
            int   inside = 0;
            float dx  = px - cx, dy = py - cy;
            float rdx = dx*c_theta - dy*s_theta;
            float rdy = dx*s_theta + dy*c_theta;

            switch (class_id)
            {
            case CLASS_CIRCLE:
            {
                int outer = (dx*dx + dy*dy) <= (r*r);
                int inner = (dx*dx + dy*dy) <= (inner_r*inner_r);
                inside = is_hollow ? (outer && !inner) : outer;
            } break;
            case CLASS_SQUARE:
            {
                int outer = (fabsf(rdx) <= r) & (fabsf(rdy) <= r);
                int inner = (fabsf(rdx) <= inner_r) & (fabsf(rdy) <= inner_r);
                inside = is_hollow ? (outer && !inner) : outer;
            } break;
            case CLASS_TRIANGLE:
            {
                float sqrt3 = 1.73205081f;
                float bx0=0,by0=-r, bx1=-r*sqrt3/2,by1=r/2, bx2=r*sqrt3/2,by2=r/2;
                float rx0=bx0*c_theta-by0*s_theta, ry0=bx0*s_theta+by0*c_theta;
                float rx1=bx1*c_theta-by1*s_theta, ry1=bx1*s_theta+by1*c_theta;
                float rx2=bx2*c_theta-by2*s_theta, ry2=bx2*s_theta+by2*c_theta;
                float min_x=fminf(fminf(rx0,rx1),rx2),max_x=fmaxf(fmaxf(rx0,rx1),rx2);
                float min_y=fminf(fminf(ry0,ry1),ry2),max_y=fmaxf(fmaxf(ry0,ry1),ry2);
                float bbc_x=(min_x+max_x)/2, bbc_y=(min_y+max_y)/2;
                float tx0=rx0-bbc_x,ty0=ry0-bbc_y,tx1=rx1-bbc_x,ty1=ry1-bbc_y,tx2=rx2-bbc_x,ty2=ry2-bbc_y;
                int   outer=point_in_triangle(dx,dy,tx0,ty0,tx1,ty1,tx2,ty2);
                float f=inner_r/r,cc_x=-bbc_x,cc_y=-bbc_y;
                float ix0=cc_x+f*(tx0-cc_x),iy0=cc_y+f*(ty0-cc_y);
                float ix1=cc_x+f*(tx1-cc_x),iy1=cc_y+f*(ty1-cc_y);
                float ix2=cc_x+f*(tx2-cc_x),iy2=cc_y+f*(ty2-cc_y);
                int   inner=point_in_triangle(dx,dy,ix0,iy0,ix1,iy1,ix2,iy2);
                inside=is_hollow?(outer&&!inner):outer;
            } break;
            }
            pixels[y*IMG_SIZE+x] = inside ? randf_range(0.80f,1.00f) : randf_range(0.00f,0.12f);
        }
}

static void fill_dataset(Mat t, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        int class_id = (int)(i % NUM_CLASSES);
        float pixels[NUM_INPUTS];
        generate_sample(pixels, class_id);
        for (size_t j = 0; j < NUM_INPUTS;  j++) MAT_AT(t, i, j) = pixels[j];
        for (size_t j = 0; j < NUM_CLASSES; j++)
            MAT_AT(t, i, NUM_INPUTS+j) = (j==(size_t)class_id) ? 1.0f : 0.0f;
    }
    mat_shuffle_rows(t);
}

static int write_dataset(Mat t, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: could not open '%s' for writing\n", path); return 0; }
    save_mat(f, t);
    fclose(f);
    printf("Saved %zu samples -> %s\n", t.rows, path);
    return 1;
}

int main()
{
    srand((unsigned)time(nullptr));
    printf("Generating shapes dataset (28x28, %d classes)...\n", NUM_CLASSES);
    Mat train = mat_alloc(nullptr, NUM_TRAIN, NUM_COLS);
    fill_dataset(train, NUM_TRAIN);
    if (!write_dataset(train, "demos/data/shapes_train.mat")) return 1;
    Mat test = mat_alloc(nullptr, NUM_TEST, NUM_COLS);
    fill_dataset(test, NUM_TEST);
    if (!write_dataset(test, "demos/data/shapes_test.mat")) return 1;
    printf("Done.\n");
    return 0;
}

#include <ctime>

#define NN_ACT ACT_SIG
#define NN_BACKPROP_TRADITIONAL
#define NN_IMPLEMENTATION
#include "../nn.hpp"

#define NN_UI_IMPLEMENTATION
#include "../nn_ui.hpp"

#define BITS 4

#define IMG_FACTOR 80
#define IMG_WIDTH  (16 * IMG_FACTOR)
#define IMG_HEIGHT (9  * IMG_FACTOR)

static size_t max_epoch        = 10000;
static size_t batches_per_frame = 20;
static size_t batch_size        = (1 << BITS) * (1 << BITS);
static bool   paused            = true;
static float  rate              = 1.0f;
static size_t arch[]            = {2 * BITS, 4 * BITS, BITS + 1};

int main()
{
    InitWindow(IMG_WIDTH, IMG_HEIGHT, "Adder NN");
    SetTargetFPS(120);

    srand((unsigned)time(nullptr));

    size_t n    = (1 << BITS);
    size_t rows = n * n;
    Mat t = mat_alloc(nullptr, rows, 2 * BITS + BITS + 1);

    for (size_t i = 0; i < t.rows; i++)
    {
        size_t x = i / n;
        size_t y = i % n;
        size_t z = x + y;
        for (size_t j = 0; j < BITS; j++)
        {
            MAT_AT(t, i, j)          = (float)((x >> j) & 1);
            MAT_AT(t, i, j + BITS)   = (float)((y >> j) & 1);
            MAT_AT(t, i, 2*BITS + j) = (float)((z >> j) & 1);
        }
        MAT_AT(t, i, 2 * BITS + BITS) = (float)(z >= n);
    }

    Region temp = region_alloc_allocator(256 * 1024 * 1024);
    NN nn = nn_alloc(nullptr, arch, ARRAY_LEN(arch));
    nn_rand(nn, -1, 1);
    NN_PRINT(nn);

    float     cost  = 0.0f;
    Cost_Plot plot  = {};
    size_t    epoch = 0;
    Batch     batch = {};

    while (!WindowShouldClose())
    {
        if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_P))
            paused = !paused;

        if (IsKeyPressed(KEY_R))
        {
            epoch          = 0;
            nn_rand(nn, -1, 1);
            plot.count     = 0;
            batch.finished = false;
            batch.begin    = 0;
            batch.cost     = 0;
        }

        for (size_t i = 0; i < batches_per_frame && !paused && epoch < max_epoch; i++)
        {
            batch_process(&temp, &batch, batch_size, nn, t, rate);
            if (batch.finished)
            {
                epoch++;
                cost = batch.cost;
                da_append(&plot, cost);
                mat_shuffle_rows(t);
            }
        }

        BeginDrawing();
        ClearBackground((Color){0x18, 0x18, 0x18, 0xFF});

        int w = GetRenderWidth();
        int h = GetRenderHeight();

        char buffer[256];
        snprintf(buffer, sizeof(buffer),
                 "Epoch : %zu/%zu\t\tRate : %f\t\tCost : %f",
                 epoch, max_epoch, rate, cost);
        DrawText(buffer, 0, 0, (int)(h * 0.04), WHITE);

        int rw, rh, rx, ry;
        rw = w / 2;
        rh = h * 2 / 3;
        rx = 0;
        ry = h / 2 - rh / 2;
        plot_cost(plot, rx, ry, rw, rh);

        int zero_y     = ry + rh;
        int font_size  = (int)(h * 0.04);
        int text_pad_x = 10;
        int text_pad_y = 5;
        DrawLine(rx, zero_y, rx + rw, zero_y, RAYWHITE);
        DrawText("0", rx + text_pad_x, zero_y - font_size - text_pad_y, font_size, RAYWHITE);

        rx += rw;
        nn_render_raylib(nn, rx, ry, rw, rh);

        EndDrawing();
        region_reset(&temp);
    }

    for (size_t x = 0; x < n; x++)
        for (size_t y = 0; y < n; y++)
        {
            printf("%zu + %zu = ", x, y);
            for (size_t j = 0; j < BITS; j++)
            {
                MAT_AT(NN_INPUT(nn), 0, j)        = (float)((x >> j) & 1);
                MAT_AT(NN_INPUT(nn), 0, j + BITS) = (float)((y >> j) & 1);
            }
            nn_forward(nn);
            if (MAT_AT(NN_OUTPUT(nn), 0, BITS) > 0.5f)
                printf("OVERFLOW\n");
            else
            {
                size_t z = 0;
                for (size_t j = 0; j < BITS; j++)
                {
                    size_t bit = (size_t)(MAT_AT(NN_OUTPUT(nn), 0, j) > 0.5f);
                    z |= bit << j;
                }
                printf("%zu\n", z);
            }
        }
    return 0;
}

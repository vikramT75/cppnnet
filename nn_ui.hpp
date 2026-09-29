#pragma once

#include <raylib.h>
#include <raymath.h>
#include <cassert>
#include <cfloat>
#include <cstdlib>
#include <cmath>
#include "nn.hpp"

struct Cost_Plot {
    float  *items;
    size_t  count;
    size_t  capacity;
};

#define DA_INIT_CAP 256
#define da_append(da, item)                                                             \
    do {                                                                                \
        if ((da)->count >= (da)->capacity) {                                            \
            (da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity * 2;   \
            (da)->items = (float*)realloc((da)->items,                                  \
                                 (da)->capacity * sizeof(*(da)->items));                \
            assert((da)->items != NULL && "Not enough memory");                         \
        }                                                                               \
        (da)->items[(da)->count++] = (item);                                            \
    } while (0)

void nn_render_raylib(NN nn, int rx, int ry, int rw, int rh);
void cost_plot_minmax(Cost_Plot plot, float *min, float *max);
void plot_cost(Cost_Plot plot, int rx, int ry, int rw, int rh);

#ifdef NN_UI_IMPLEMENTATION
#ifndef NN_UI_IMPLEMENTATION_DONE
#define NN_UI_IMPLEMENTATION_DONE

void nn_render_raylib(NN nn, int rx, int ry, int rw, int rh)
{
    Color low_colour        = {0xFF, 0x00, 0xFF, 0xFF};
    Color high_colour       = {0x00, 0xFF, 0x00, 0x00};
    Color background_colour = {0x18, 0x18, 0x18, 0xFF};
    (void)background_colour;

    float neuron_radius   = rh * 0.04f;
    int   layer_border_vpad = 50;
    int   layer_border_hpad = 50;
    int   nn_width  = rw - 2 * layer_border_hpad;
    int   nn_height = rh - 2 * layer_border_vpad;
    int   nn_x      = rx + rw / 2 - nn_width  / 2;
    int   nn_y      = ry + rh / 2 - nn_height / 2;
    int   layer_hpad = nn_width / (int)nn.arch_count;

    for (size_t l = 0; l < nn.arch_count; l++)
    {
        int layer_vpad1 = nn_height / (int)nn.as[l].cols;

        for (size_t i = 0; i < nn.as[l].cols; i++)
        {
            int cx1 = nn_x + (int)l * layer_hpad + layer_hpad / 2;
            int cy1 = nn_y + (int)i * layer_vpad1 + layer_vpad1 / 2;

            if (l + 1 < nn.arch_count)
            {
                int layer_vpad2 = nn_height / (int)nn.as[l + 1].cols;
                for (size_t j = 0; j < nn.as[l + 1].cols; j++)
                {
                    int cx2 = nn_x + (int)(l + 1) * layer_hpad + layer_hpad / 2;
                    int cy2 = nn_y + (int)j * layer_vpad2 + layer_vpad2 / 2;

                    float value = sigmoidf(MAT_AT(nn.ws[l], i, j));
                    high_colour.a = (unsigned char)floorf(255.f * value);
                    float   thickness = rh * 0.004f;
                    Vector2 start     = {(float)cx1, (float)cy1};
                    Vector2 end       = {(float)cx2, (float)cy2};
                    DrawLineEx(start, end, thickness,
                               ColorAlphaBlend(low_colour, high_colour, RAYWHITE));
                }
            }

            if (l > 0)
            {
                high_colour.a = (unsigned char)floorf(255.f * sigmoidf(MAT_AT(nn.bs[l - 1], 0, i)));
                DrawCircle(cx1, cy1, neuron_radius,
                           ColorAlphaBlend(low_colour, high_colour, RAYWHITE));
            }
            else
            {
                DrawCircle(cx1, cy1, neuron_radius, GRAY);
            }
        }
    }
}

void cost_plot_minmax(Cost_Plot plot, float *min, float *max)
{
    *max = FLT_MIN;
    *min = FLT_MAX;
    for (size_t i = 0; i < plot.count; i++)
    {
        if (*max < plot.items[i]) *max = plot.items[i];
        if (*min > plot.items[i]) *min = plot.items[i];
    }
}

void plot_cost(Cost_Plot plot, int rx, int ry, int rw, int rh)
{
    float max_val, min_val;
    cost_plot_minmax(plot, &min_val, &max_val);
    if (min_val > 0) min_val = 0;
    float  range = max_val - min_val;
    size_t n     = plot.count;
    if (n < 1000) n = 1000;

    for (size_t i = 0; i + 1 < plot.count; i++)
    {
        float x1 = rx + (float)rw * (float)i / (float)n;
        float y1 = ry + rh * (1.f - (plot.items[i]     - min_val) / range);
        float x2 = rx + (float)rw * (float)(i + 1) / (float)n;
        float y2 = ry + rh * (1.f - (plot.items[i + 1] - min_val) / range);
        DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, rh * 0.005f, RED);
    }
}

#endif
#endif

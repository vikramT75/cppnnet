#include <cstdio>
#include <cstdlib>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include "../thirdparty/stb_image.h"

#define NN_IMPLEMENTATION
#include "../nn.hpp"

#define IMG_SIZE    28
#define NUM_INPUTS  (IMG_SIZE * IMG_SIZE)  // 784
#define NUM_CLASSES 10
#define NUM_COLS    (NUM_INPUTS + NUM_CLASSES) // 794

static void save_mat(FILE *out, Mat m)
{
    const char *magic = "nn.h.mat";
    fwrite(magic, strlen(magic), 1, out);
    fwrite(&m.rows, sizeof(m.rows), 1, out);
    fwrite(&m.cols, sizeof(m.cols), 1, out);
    for (size_t i = 0; i < m.rows; i++)
        fwrite(&MAT_AT(m, i, 0), sizeof(float), m.cols, out);
}

struct SampleBuf {
    float  *data;
    size_t  count;
    size_t  capacity;
};

static void sb_push(SampleBuf *sb, const float row[NUM_COLS])
{
    if (sb->count >= sb->capacity)
    {
        sb->capacity = sb->capacity == 0 ? 4096 : sb->capacity * 2;
        sb->data = (float *)realloc(sb->data, sb->capacity * NUM_COLS * sizeof(float));
        if (!sb->data) { fprintf(stderr, "ERROR: out of memory\n"); exit(1); }
    }
    memcpy(sb->data + sb->count * NUM_COLS, row, NUM_COLS * sizeof(float));
    sb->count++;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s <manifest.txt> <output.mat>\n\n", argv[0]);
        fprintf(stderr, "Manifest: one line per sample - <path.png> <label 0-9>\n");
        return 1;
    }

    const char *manifest_path = argv[1];
    const char *output_path   = argv[2];

    FILE *mf = fopen(manifest_path, "r");
    if (!mf) { fprintf(stderr, "ERROR: cannot open manifest '%s'\n", manifest_path); return 1; }

    SampleBuf sb          = {};
    size_t    skipped     = 0;
    size_t    class_counts[NUM_CLASSES] = {};
    char      line[2048];
    size_t    line_num = 0;

    while (fgets(line, sizeof(line), mf))
    {
        line_num++;
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;

        char img_path[2000];
        int  label = -1;
        if (sscanf(line, "%1999s %d", img_path, &label) != 2 ||
            label < 0 || label >= NUM_CLASSES)
        {
            fprintf(stderr, "WARNING: line %zu malformed, skipping\n", line_num);
            skipped++; continue;
        }

        int w, h, comp;
        unsigned char *pixels = stbi_load(img_path, &w, &h, &comp, 1);
        if (!pixels) { skipped++; continue; }
        if (w != IMG_SIZE || h != IMG_SIZE) { stbi_image_free(pixels); skipped++; continue; }

        float row[NUM_COLS];
        for (int i = 0; i < NUM_INPUTS;  i++) row[i]              = (float)pixels[i] / 255.f;
        for (int j = 0; j < NUM_CLASSES; j++) row[NUM_INPUTS + j] = (j == label) ? 1.0f : 0.0f;
        stbi_image_free(pixels);
        sb_push(&sb, row);
        class_counts[label]++;
        if (sb.count % 5000 == 0) printf("  Loaded %zu samples...\n", sb.count);
    }
    fclose(mf);

    printf("Done reading manifest.\n");
    for (int c = 0; c < NUM_CLASSES; c++) printf("  Class %d: %zu samples\n", c, class_counts[c]);
    if (skipped > 0) printf("  Skipped: %zu entries\n", skipped);
    printf("  Total loaded: %zu samples\n", sb.count);

    if (sb.count == 0)
    {
        fprintf(stderr, "ERROR: no samples loaded\n");
        free(sb.data);
        return 1;
    }

    Mat t;
    t.rows   = sb.count;
    t.cols   = NUM_COLS;
    t.stride = NUM_COLS;
    t.es     = sb.data;
    mat_shuffle_rows(t);

    FILE *f = fopen(output_path, "wb");
    if (!f)
    {
        fprintf(stderr, "ERROR: cannot open '%s' for writing\n", output_path);
        free(sb.data);
        return 1;
    }
    save_mat(f, t);
    fclose(f);
    printf("Saved -> %s\n", output_path);
    free(sb.data);
    return 0;
}

#include <stdio.h>
#include <stdlib.h>

#define IMAGE_SIZE 28
#define IMAGE_AREA (IMAGE_SIZE * IMAGE_SIZE)
#define MNIST_HEADER_SIZE 16
#define LABEL_HEADER_SIZE 8
#define NUM_IMAGES 60000 // For the MNIST test dataset

int reverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_image(FILE* image_file, unsigned char image[IMAGE_SIZE][IMAGE_SIZE]) {
    fread(image, sizeof(unsigned char), IMAGE_AREA, image_file);
}

void load_mnist_dataset(const char* image_filename, const char* label_filename, unsigned char images[][IMAGE_SIZE][IMAGE_SIZE], unsigned char labels[], int* count) {
    FILE* image_file = fopen(image_filename, "rb");
    FILE* label_file = fopen(label_filename, "rb");

    if (image_file == NULL || label_file == NULL) {
        perror("Error opening file");
        exit(1);
    }

    // Read and handle headers
    fread(count, sizeof(int), 1, image_file);
    *count = reverseInt(*count);
    fseek(label_file, LABEL_HEADER_SIZE, SEEK_SET); // Skip label file header

    for (int i = 0; i < *count; ++i) {
        read_image(image_file, images[i]);
        fread(&labels[i], sizeof(unsigned char), 1, label_file);
    }

    fclose(image_file);
    fclose(label_file);
}

void print_image(unsigned char image[IMAGE_SIZE][IMAGE_SIZE]) {
    for (int y = 0; y < IMAGE_SIZE; ++y) {
        for (int x = 0; x < IMAGE_SIZE; ++x) {
            printf("%c", image[y][x] < 128 ? '.' : '*');
        }
        printf("\n");
    }
}

// int main() {
//     unsigned char (*images)[IMAGE_SIZE][IMAGE_SIZE] = malloc(sizeof(unsigned char) * NUM_IMAGES * IMAGE_AREA);
//     unsigned char *labels = malloc(sizeof(unsigned char) * NUM_IMAGES);

//     if (images == NULL || labels == NULL) {
//         perror("Error allocating memory");
//         exit(1);
//     }

//     int num_images = NUM_IMAGES;
//     load_mnist_dataset("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", images, labels, &num_images);

//     // Print the first image and its label
//     for(int i = 10; i < 20; i++){ 
//         printf("Label of the first image: %d\n", labels[i]);
//         printf("First image:\n");
//         print_image(images[i]);
//     }

//     // Free the allocated memory
//     free(images);
//     free(labels);

//     return 0;
// }

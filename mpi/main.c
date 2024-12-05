#include "lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#define FILE_TRAIN_IMAGE "train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL "train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE "t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL "t10k-labels-idx1-ubyte"
#define LENET_FILE "model.dat"
#define COUNT_TRAIN 60000
#define COUNT_TEST 10000

double wtime(void)
{
    double now_time;
    struct timeval etstart;

    if (gettimeofday(&etstart, NULL) == -1)
        perror("Error: calling gettimeofday() not successful.\n");

    now_time = ((etstart.tv_sec) * 1000 + etstart.tv_usec / 1000.0);
    return now_time;
}

int read_data(unsigned char (*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image || !fp_label)
        return 1;
    fseek(fp_image, 16, SEEK_SET);
    fseek(fp_label, 8, SEEK_SET);
    fread(data, sizeof(*data) * count, 1, fp_image);
    fread(label, count, 1, fp_label);
    fclose(fp_image);
    fclose(fp_label);
    return 0;
}

void training(LeNet5 *lenet, image *train_data, uint8 *train_label, int batch_size, int total_size, int rank, int size)
{
    for (int i = 0; i < total_size; i += batch_size)
    {
        int current_batch_size = (i + batch_size <= total_size) ? batch_size : (total_size - i);
        TrainBatch(lenet, train_data + i, train_label + i, current_batch_size, rank, size);
    }
}

int testing(LeNet5 *lenet, image *test_data, uint8 *test_label, int total_size)
{
    int right = 0, percent = 0;
    for (int i = 0; i < total_size; ++i)
    {
        uint8 l = test_label[i];
        int p = Predict(lenet, test_data[i], 10);
        right += l == p;
    }
    return right;
}

int save(LeNet5 *lenet, char filename[])
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
        return 1;
    fwrite(lenet, sizeof(LeNet5), 1, fp);
    fclose(fp);
    return 0;
}

int load(LeNet5 *lenet, char filename[])
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
        return 1;
    fread(lenet, sizeof(LeNet5), 1, fp);
    fclose(fp);
    return 0;
}

void foo()
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
    uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
    image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
    uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));

    if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
    {
        if (rank == 0)
            printf("ERROR!!!\nDataset File Not Found!\n");
        free(train_data);
        free(train_label);
    }
    if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
    {
        if (rank == 0)
            printf("ERROR!!!\nDataset File Not Found!\n");
        free(test_data);
        free(test_label);
    }

    LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
    if (rank == 0 && load(lenet, LENET_FILE))
        Initial(lenet);

    MPI_Bcast(lenet, sizeof(LeNet5) / sizeof(double), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Measure training time
    if (rank == 0)
        printf("Training started...\n");
    double start_time = wtime();
    training(lenet, train_data, train_label, 300, COUNT_TRAIN, rank, size);
    double training_time = (wtime() - start_time) / 1000.0;
    if (rank == 0)
        printf("Training complete.\n");

    // Measure testing time and precision
    if (rank == 0)
        printf("Testing started...\n");
    double testing_start_time = wtime();
    int correct_predictions = 0;
    if (rank == 0)
        correct_predictions = testing(lenet, test_data, test_label, COUNT_TEST);
    double testing_time = (wtime() - testing_start_time) / 1000.0;
    if (rank == 0)
        printf("Testing complete.\n");

    if (rank == 0)
    {
        double precision = (double)correct_predictions / COUNT_TEST;
        printf("Precision is %.6f\n", precision);
        printf("Training time is %.6fs\n", training_time);
        printf("Testing time is %.6fs\n", testing_time);

        save(lenet, LENET_FILE);
    }

    free(lenet);
    free(train_data);
    free(train_label);
    free(test_data);
    free(test_label);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    foo();
    MPI_Finalize();
    return 0;
}

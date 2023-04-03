#include "load_data.h"

int main(){
    // int data_num = 200;
    // int data_dim = 173762;
    // char **data = new char*[data_num];
    // LoadBinToSparseVector<float, uint32_t>("../../pysparnn/examples/20_newsgroups_querydata_sparse.bin", data, data_num, data_dim);
    // uint32_t vector_len = *(uint32_t *)data[51];
    // printf("vector_len: %u\n", vector_len);
    // float data_sum = 0;
    // for (int i = 0; i < vector_len; i++){
    //     float data_i = *(float *)(data[51] + sizeof(uint32_t) + i * sizeof(float));
    //     data_sum += data_i*data_i;
    //     printf("%f ", data_i);
    // }
    // printf("\n");
    // printf("data_sum: %f\n", data_sum);
    
    // for (int i = 0; i < data_num; i++)
    // {
    //     delete[] data[i];
    // }
    // delete[] data;

    uint32_t data_num = 200;
    uint32_t dense_dim = 100;
    char **data = new char*[data_num];
    char **attr = new char*[data_num];
    LoadHybridBinToArray<float, uint64_t>("../test_data/wiki.hybrid", data, attr, data_num, dense_dim);
    
    for (int i = 0; i < 1; i++)
    {
        float data_sum = 0;
        printf("dense_dim %d\n", dense_dim);
        for (int j = 0; j < dense_dim; j++){
            float data_i = *(float *)(data[i] + j * sizeof(float));
            data_sum += data_i*data_i;
            printf("%f ", data_i);
        }
        printf("\n");
        printf("dense data_sum: %f\n", data_sum);
    }

    for (int i = 0; i < data_num; i++)
    {
        delete[] data[i];
        delete[] attr[i];
    }
    delete[] data;
    delete[] attr;
    
    return 0;
}
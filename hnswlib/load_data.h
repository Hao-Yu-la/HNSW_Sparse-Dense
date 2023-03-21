#include <queue>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <fstream>

// load file. store format: (uint32_t)num, (uint32_t)dim, (data_T)num * dim.
template<typename data_T>
void LoadBinToArray(const std::string& file_path, data_T *data_m, uint32_t nums, uint32_t dims, bool non_header = false){
    std::ifstream file_reader(file_path.c_str(), std::ios::binary);
    if (!non_header){
        uint32_t nums_r, dims_r;
        file_reader.read((char *) &nums_r, sizeof(uint32_t));
        file_reader.read((char *) &dims_r, sizeof(uint32_t));
        if ((nums != nums_r) || (dims != dims_r)){
            printf("Error, file %s is error, nums_r: %u, dims_r: %u\n", file_path.c_str(), nums_r, dims_r);
            exit(1);
        }
    }

    uint32_t readsize = dims * sizeof(data_T);
    for (uint i = 0; i < nums; i++) {
        file_reader.read((char *) (data_m + dims * i), readsize);
        if (file_reader.gcount() != readsize) {
            printf("Read Error\n"); exit(1);
        }
    }
    file_reader.close();
    printf("Load %u * %u Data from %s done.\n", nums, dims, file_path.c_str());
}

template<typename data_T, typename index_T>
void LoadBinToSparseVector(const std::string& file_path, char **data_m, uint32_t nums, uint32_t dims, bool non_header = false){
    std::ifstream file_reader(file_path.c_str(), std::ios::binary);
    if (!non_header){
        uint32_t nums_r, dims_r;
        file_reader.read((char *) &nums_r, sizeof(int32_t));
        file_reader.read((char *) &dims_r, sizeof(int32_t));
        if ((nums != nums_r) || (dims != dims_r)){
            printf("Error, file %s is error, nums_r: %u, dims_r: %u\n", file_path.c_str(), nums_r, dims_r);
            exit(1);
        }
    }

    index_T vector_len = 0;
    for (uint i = 0; i < nums; i++) {
        file_reader.read((char *) &vector_len, sizeof(index_T));
        data_m[i] = (char *) malloc(sizeof(index_T) + vector_len * (sizeof(data_T) + sizeof(index_T)));
        memcpy(data_m[i], &vector_len, sizeof(index_T));
        file_reader.read(data_m[i] + sizeof(index_T), vector_len * (sizeof(data_T) + sizeof(index_T)));
    }

    file_reader.close();
    printf("Load %u * %u Data from %s done.\n", nums, dims, file_path.c_str());
}

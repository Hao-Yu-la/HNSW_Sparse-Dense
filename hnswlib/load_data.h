#include <queue>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <fstream>
#include <cmath>

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

// load file. store format: (uint32_t)num, (uint32_t)dim, (index_T)vector_len, vector_len * data_T, vector_len * index_T.
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

// load hybird file. store format: 
// hybrid文件自描述信息头（每个字段为一个 int） magic_version is 73280002
//         [magic_version][label_size][embedding_type_size][dense_embedding_dim][place_holder_0][has_sparse]
//         [place_holder_1][place_holder_2][has_attr_bytes]
//         因此文件头长度为 9 * 4 = 36 bytes
//         label_size 为 sizeof(uint64);
//         embedding_type_size 为 sizeof(float);
//         embedding_dim 为存储向量维度，例如 128 或 384 等;
//         has_sparse, has_attr_bytes 为 0 或 1;
//     向量条目
//         [label_K][float: embedding_V x dim]
//             [int: sparse_dim]
//             [u64: sparse_id][float: sparse_value] x sparse_dim
//     附加信息
//         [int: attr_bytes_len][attr_bytes x alen]
template<typename data_T, typename index_T>
void LoadHybridBinToArray(const std::string& file_path, char **data_m, char **attr_m, uint32_t nums, uint32_t &dims, bool non_header = false){
    uint32_t has_sparse, has_attr_bytes;
    std::ifstream file_reader(file_path.c_str(), std::ios::binary);
    if (!non_header){
        uint32_t magic_version;
        file_reader.read((char *) &magic_version, sizeof(uint32_t));
        if (magic_version != 73280002){
            printf("Error, file %s is error, magic_version: %u\n", file_path.c_str(), magic_version);
            exit(1);
        }
        uint32_t label_size, embedding_type_size, dense_embedding_dim, place_holder_0, \
            place_holder_1, place_holder_2;
        file_reader.read((char *) &label_size, sizeof(uint32_t));
        file_reader.read((char *) &embedding_type_size, sizeof(uint32_t));
        file_reader.read((char *) &dense_embedding_dim, sizeof(uint32_t));
        file_reader.read((char *) &place_holder_0, sizeof(uint32_t));
        file_reader.read((char *) &has_sparse, sizeof(uint32_t));
        file_reader.read((char *) &place_holder_1, sizeof(uint32_t));
        file_reader.read((char *) &place_holder_2, sizeof(uint32_t));
        file_reader.read((char *) &has_attr_bytes, sizeof(uint32_t));
        dims = dense_embedding_dim;
        if ((label_size != sizeof(uint64_t)) || (embedding_type_size != sizeof(float)) \
            || (dense_embedding_dim != dims) || (has_sparse != 1) || (has_attr_bytes != 1)){
            printf("Error, file %s is error, label_size: %u, embedding_type_size: %u, \
                dense_embedding_dim: %u, has_sparse: %u, has_attr_bytes: %u\n", file_path.c_str(), \
                label_size, embedding_type_size, dense_embedding_dim, has_sparse, has_attr_bytes);
            exit(1);
        }
    }

    uint64_t label = 0;
    float dense_data[dims];
    uint32_t sparse_dim = 0;
    index_T sparse_dim_64 = 0;
    uint32_t attr_bytes_len = 0;
    for (uint i = 0; i < nums; i++) {
        //read label
        file_reader.read((char *) &label, sizeof(uint64_t));
        //read dense data
        file_reader.read((char *) dense_data, dims * sizeof(float));
        //normalize dense data
        float norm = 0.0;
        for (uint j = 0; j < dims; j++){
            norm += dense_data[j] * dense_data[j];
        }
        norm = sqrt(norm);
        for (uint j = 0; j < dims; j++){
            dense_data[j] /= norm;
        }
        //read sparse data
        file_reader.read((char *) &sparse_dim, sizeof(uint32_t));
        sparse_dim_64 = sparse_dim;
        uint64_t sparse_id[sparse_dim];
        float sparse_value[sparse_dim];
        for (uint j = 0; j < sparse_dim; j++){
            file_reader.read((char *) &sparse_id[j], sizeof(uint64_t));
            file_reader.read((char *) &sparse_value[j], sizeof(float));
        }
        //normalize sparse data
        norm = 0.0;
        for (uint j = 0; j < sparse_dim; j++){
            norm += sparse_value[j] * sparse_value[j];
        }
        norm = sqrt(norm);
        for (uint j = 0; j < sparse_dim; j++){
            sparse_value[j] /= norm;
        }
        //read attr bytes
        if (has_attr_bytes){
            file_reader.read((char *) &attr_bytes_len, sizeof(uint32_t));
            attr_m[i] = (char *) malloc(attr_bytes_len);
            file_reader.read(attr_m[i], attr_bytes_len);
        }
        //write to data_m
        data_m[i] = (char *) malloc(dims * sizeof(float) + sizeof(index_T) + sparse_dim * (sizeof(uint64_t) + sizeof(float)));
        memcpy(data_m[i], dense_data, dims * sizeof(float));
        memcpy(data_m[i] + dims * sizeof(float), &sparse_dim_64, sizeof(index_T));
        memcpy(data_m[i] + dims * sizeof(float) + sizeof(index_T), sparse_value, sparse_dim * sizeof(float));
        memcpy(data_m[i] + dims * sizeof(float) + sizeof(index_T) + sparse_dim * sizeof(float), sparse_id, sparse_dim * sizeof(uint64_t));
    }
    printf("Load %u data from %s done.\n", nums, file_path.c_str());
}

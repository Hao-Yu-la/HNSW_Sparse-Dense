#include "../../hnswlib/hnswlib.h"
#include <set>


int main() {
    int dense_dim = 16;               // Dimension of the elements
    int max_elements = 10;   // Maximum number of elements, should be known beforehand
    int M = 16;                // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 5;  // Controls index search speed/build speed tradeoff

    int sparse_dim = 100000;
    int none_zero_num_bound = 10;
    int max_value = 10;

    int query_num = 10;

    // Initing index
    hnswlib::InnerProductSpace space(dense_dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    alg_hnsw->setEf(5);

    // generate dense-sparse data
    srand((unsigned)time(NULL));
    char **data = new char*[max_elements];
    uint dense_data_size = dense_dim * sizeof(hnswlib::vectordata_t);
    for (int i = 0; i < max_elements; i++)
    {
        //generate sparse vector none_zero_num
        hnswlib::vectorsizeint none_zero_num = rand() % none_zero_num_bound + 1;
        data[i] = new char[dense_data_size + none_zero_num * (sizeof(hnswlib::vectordata_t) + \
            sizeof(hnswlib::vectorsizeint)) + sizeof(hnswlib::vectorsizeint)];
        hnswlib::vectorsizeint len = none_zero_num;
        memcpy(data[i] + dense_data_size, &len, sizeof(hnswlib::vectorsizeint));

        //generate dense data
        for (int j = 0; j < dense_dim; j++)
        {
            hnswlib::vectordata_t value = rand() % max_value;
            memcpy(data[i] + j * sizeof(hnswlib::vectordata_t), &value, sizeof(hnswlib::vectordata_t));
        }

        //generate sparse data
        for (int j = 0; j < none_zero_num; j++)
        {
            hnswlib::vectordata_t value = rand() % max_value + 1;
            memcpy(data[i] + dense_data_size + sizeof(hnswlib::vectorsizeint) + j * sizeof(hnswlib::vectordata_t), \
                &value, sizeof(hnswlib::vectordata_t));
        }

        //generate sparse index
        std::set<hnswlib::vectorsizeint> index_set;
        while (index_set.size() < none_zero_num)
        {
            index_set.insert(rand() % sparse_dim);
        }
        std::set<hnswlib::vectorsizeint>::iterator iter = index_set.begin();
        for (int k = 0; k < none_zero_num; k++)
        {
            hnswlib::vectorsizeint index = *iter;
            memcpy(data[i] + dense_data_size + sizeof(hnswlib::vectorsizeint) + none_zero_num * sizeof(hnswlib::vectordata_t) \
                + k * sizeof(hnswlib::vectorsizeint), &index, sizeof(hnswlib::vectorsizeint));
            iter++;
        }
    }

    // Normalize data
    for (int i = 0; i < max_elements; i++)
    {
        // Normalize sparse data
        float norm = 0;
        hnswlib::vectorsizeint len;
        memcpy(&len, data[i] + dense_data_size, sizeof(hnswlib::vectorsizeint));
        for (int j = 0; j < len; j++)
        {
            hnswlib::vectordata_t value;
            memcpy(&value, data[i] + dense_data_size + sizeof(hnswlib::vectorsizeint) + j * sizeof(hnswlib::vectordata_t), \
                sizeof(hnswlib::vectordata_t));
            norm += value * value;
        }
        norm = sqrt(norm);
        for (int j = 0; j < len; j++)
        {
            hnswlib::vectordata_t value;
            memcpy(&value, data[i] + dense_data_size + sizeof(hnswlib::vectorsizeint) + j * sizeof(hnswlib::vectordata_t), \
                sizeof(hnswlib::vectordata_t));
            value /= norm;
            memcpy(data[i] + dense_data_size + sizeof(hnswlib::vectorsizeint) + j * sizeof(hnswlib::vectordata_t), \
                &value, sizeof(hnswlib::vectordata_t));
        }

        // Normalize dense data
        norm = 0;
        for (int j = 0; j < dense_dim; j++)
        {
            hnswlib::vectordata_t value;
            memcpy(&value, data[i] + j * sizeof(hnswlib::vectordata_t), sizeof(hnswlib::vectordata_t));
            norm += value * value;
        }
        norm = sqrt(norm);
        for (int j = 0; j < dense_dim; j++)
        {
            hnswlib::vectordata_t value;
            memcpy(&value, data[i] + j * sizeof(hnswlib::vectordata_t), sizeof(hnswlib::vectordata_t));
            value /= norm;
            memcpy(data[i] + j * sizeof(hnswlib::vectordata_t), &value, sizeof(hnswlib::vectordata_t));
        }
    }

    clock_t start1, finish1;
    clock_t start2, finish2;

    // Add data to index
    std::cout << "addPoint start\n";
    start1 = clock();
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data[i], i);
    }
    finish1 = clock();
    std::cout << "addPoint finish\n";
    
    // Query the elements for themselves and measure recall
    start2 = clock();
    float correct = 0;
    for (int i = 0; i < query_num; i++) {
        std::vector<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnnCloserFirst(data[i], 1);
        hnswlib::labeltype label = result[0].second;
        float dist = result[0].first;
        if (label == i) correct++;
        // if (label != i){
        //     std::cout << "i: " << i << "\n";
        //     std::cout << "label: " << label << "\n";
        //     std::cout << "dist: " << dist << "\n";
        // }
    }
    finish2 = clock();

    float recall = correct / query_num;
    float time1 = (float)(finish1 - start1) / CLOCKS_PER_SEC;
    float time2 = (float)(finish2 - start2) / CLOCKS_PER_SEC;
    std::cout << "Recall: " << recall << "\n";
    std::cout << "Time of addPoint: " << time1 << "\n";
    std::cout << "Time of searchKnn: " << time2 << "\n";
    std::cout << "Time of average searchKnn: " << time2 / query_num << "\n";

    // Serialize index
    char hnsw_path[100];
    sprintf(hnsw_path, "random_%d_dim_%d_ef_%d_M_%d.bin", max_elements, dense_dim, ef_construction, M);
    alg_hnsw->saveIndex(hnsw_path);
    delete alg_hnsw;

    // Deserialize index and check recall
    // hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);

    // count sparse rate
    // float non_zero_sum = 0;
    // for (int i = 0; i < max_elements; i++) {
    //     char* data_search = alg_hnsw->getDataByInternalId(i);
    //     hnswlib::vectorsizeint len;
    //     memcpy(&len, data_search, sizeof(hnswlib::vectorsizeint));
    //     non_zero_sum += len;
    // }
    // std::cout << "non_zero_sum: " << non_zero_sum << "\n";
    // float sparse_rate = (float)non_zero_sum / ((float)dense_dim * max_elements);
    // std::cout << "sparse_rate: " << sparse_rate << "\n";

    for (int test_i = 0; test_i < 3; test_i++){
        alg_hnsw->setEf(60 + test_i * 5);
        alg_hnsw->metric_distance_computations = 0;
        printf("ef: %d\n", 60 + test_i * 5);
        start2 = clock();
        float correct = 0;
        for (int i = 0; i < query_num; i++) {
            char* data_search = alg_hnsw->getDataByInternalId(i);
            std::vector<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnnCloserFirst(data_search, 1);
            hnswlib::labeltype label = result[0].second;
            float dist = result[0].first;
            if (label == i) correct++;
            // if (label != i){
            //     std::cout << "i: " << i << "\n";
            //     std::cout << "label: " << label << "\n";
            //     std::cout << "dist: " << dist << "\n";
            // }
        }
        finish2 = clock();
        float recall = (float)correct / query_num;
        float time2 = (float)(finish2 - start2) / CLOCKS_PER_SEC;
        std::cout << "Recall of deserialized index: " << recall << "\n";
        std::cout << "Time of searchKnn: " << time2 << "\n";
        std::cout << "metric_distance_computations: " << alg_hnsw->metric_distance_computations / query_num << "\n";
    }

    for (int i = 0; i < max_elements; i++)
    {
        delete[] data[i];
    }
    delete[] data;
    delete alg_hnsw;
    
    return 0;
}

#include "data_loader.h"

using DATALOADER::DataLoader;
DataLoader::DataLoader(std::string data_type_name, uint32_t _max_elements, std::string _data_path) {
    // float
    data_type_len = 4;
    if (data_type_name == "u8")
        data_type_len = 1;

    data_path = _data_path;
    FILE *file = fopen(data_path.c_str(), "rb");
    if (file == nullptr) {
        perror("Error opening file");
        return ;
    }
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    uint32_t elements_in_file;
    fread(&elements_in_file, sizeof(elements_in_file), 1, file);
    fread(&dim, sizeof(dim), 1, file);

    if (_max_elements == 0) elements = elements_in_file;
    else elements = std::min(_max_elements, elements_in_file);

    tot_data_size = 1ll * elements * dim * data_type_len;

    data = malloc(tot_data_size);

    size_t result = fread(data, 1, tot_data_size, file);
    if (result != tot_data_size) {
        perror("Reading error");
        free(data);
        fclose(file);
        return ;
    }

    fclose(file);
}

DataLoader::~DataLoader() {
    if (data != nullptr) {
        free(data);
        data = nullptr;
    }
}

uint32_t DataLoader::get_elements() { return elements; }

uint32_t DataLoader::get_dim() { return dim; }

void DataLoader::free_data() {
    if (data != nullptr) {
        free(data);
        data = nullptr;
    }

}
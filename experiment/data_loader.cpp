#include "data_loader.h"

namespace DATALOADER
{
    DataLoader::DataLoader(std::string data_type_name, uint32_t _max_elements, std::string _data_path, std::string _data_name)
    {
        data_name = _data_name;
        // float
        data_type_len = 4;
        if (data_type_name == "u8")
            data_type_len = 1;

        data_path = _data_path;
        FILE *file = fopen(data_path.c_str(), "rb");
        if (file == nullptr)
        {
            perror("Error opening file");
            return;
        }

        fseek(file, 0, SEEK_END);
        size_t file_size = ftell(file);
        fseek(file, 0, SEEK_SET);

        if (data_name == "gist")
        {
            // 每个点前面有4byte的dim
            offset_per_elem = 4;
            uint32_t elements_in_file = file_size / (960 * 4 + 4);
            if (_max_elements == 0)
                elements = elements_in_file;
            else
                elements = std::min(_max_elements, elements_in_file);
            dim = 960;
            fseek(file, 4, SEEK_SET);

            tot_data_size = 1ll * elements * (dim * data_type_len + offset_per_elem) - offset_per_elem;
        }
        else if (data_name == "sift")
        {
            offset_per_elem = 4;
            uint32_t elements_in_file = file_size / (128 * 4 + 4);
            if (_max_elements == 0)
                elements = elements_in_file;
            else
                elements = std::min(_max_elements, elements_in_file);
            dim = 128;
            fseek(file, 4, SEEK_SET);

            tot_data_size = 1ll * elements * (dim * data_type_len + offset_per_elem) - offset_per_elem;
        }
        else
        {

            uint32_t elements_in_file;
            fread(&elements_in_file, sizeof(elements_in_file), 1, file);
            fread(&dim, sizeof(dim), 1, file);

            if (_max_elements == 0)
                elements = elements_in_file;
            else
                elements = std::min(_max_elements, elements_in_file);

            tot_data_size = 1ll * elements * dim * data_type_len;
        }

        data = malloc(tot_data_size);

        size_t result = fread(data, 1, tot_data_size, file);
        if (result != tot_data_size)
        {
            perror("Reading error");
            free(data);
            fclose(file);
            return;
        }

        fclose(file);
    }

    DataLoader::~DataLoader()
    {
        if (data != nullptr)
        {
            free(data);
            data = nullptr;
        }
    }

    uint32_t DataLoader::get_elements() { return elements; }

    uint32_t DataLoader::get_dim() { return dim; }

    void DataLoader::free_data()
    {
        if (data != nullptr)
        {
            free(data);
            data = nullptr;
        }
    }

}
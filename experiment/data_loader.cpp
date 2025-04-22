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
        else if (data_name == "mnist")
        {
            offset_per_elem = 4;
            uint32_t elements_in_file = file_size / (784 * 4 + 4);
            cout << elements_in_file << endl;
            if (_max_elements == 0)
                elements = elements_in_file;
            else
                elements = std::min(_max_elements, elements_in_file);
            dim = 784;
            fseek(file, 4, SEEK_SET);

            tot_data_size = 1ll * elements * (dim * data_type_len + offset_per_elem) - offset_per_elem;
        }
        else if (data_name == "deep")
        {
            offset_per_elem = 4;
            uint32_t elements_in_file = file_size / (256 * 4 + 4);
            if (_max_elements == 0)
                elements = elements_in_file;
            else
                elements = std::min(_max_elements, elements_in_file);
            dim = 256;
            fseek(file, 4, SEEK_SET);

            tot_data_size = 1ll * elements * (dim * data_type_len + offset_per_elem) - offset_per_elem;
        }
        else if (data_name == "opai")
        {
            offset_per_elem = 4;
            uint32_t elements_in_file = file_size / (1536 * 4 + 4);
            if (_max_elements == 0)
                elements = elements_in_file;
            else
                elements = std::min(_max_elements, elements_in_file);
            dim = 1536;
            fseek(file, 4, SEEK_SET);

            tot_data_size = 1ll * elements * (dim * data_type_len + offset_per_elem) - offset_per_elem;
        }
        else if(data_name == "msmarco"){
            offset_per_elem = 4;
            uint32_t elements_in_file = file_size / (768 * 4 + 4);
            if (_max_elements == 0)
                elements = elements_in_file;
            else
                elements = std::min(_max_elements, elements_in_file);
            dim = 768;

            tot_data_size = 1ll * elements * (dim * data_type_len + offset_per_elem) - offset_per_elem;
        }
        else if (data_name == "nuswide")
        {
            offset_per_elem = 4;
            uint32_t elements_in_file = file_size / (500 * 4 + 4);
            if (_max_elements == 0)
                elements = elements_in_file;
            else
                elements = std::min(_max_elements, elements_in_file);
            cout << "elements in file: " << elements_in_file << endl;
            dim = 500;
            tot_data_size = 1ll * elements * (dim * data_type_len + offset_per_elem) - offset_per_elem;
        }
        else if(data_name == "tiny")
        {
            offset_per_elem = 4;
            uint32_t elements_in_file = file_size / (384 * 4 + 4);
            if (_max_elements == 0)
                elements = elements_in_file;
            else
                elements = std::min(_max_elements, elements_in_file);
            dim = 384;

            tot_data_size = 1ll * elements * (dim * data_type_len + offset_per_elem) - offset_per_elem;
        }
        else if (data_name == "yahoo")
        {
            offset_per_elem = 0;
            uint32_t elements_in_file = file_size / (128 * 4);
            if (_max_elements == 0)
                elements = elements_in_file;
            else
                elements = std::min(_max_elements, elements_in_file);
            dim = 128;

            tot_data_size = 1ll * elements * (dim * data_type_len + offset_per_elem) - offset_per_elem;
        }
        else if (data_name == "yandex")
        {
            offset_per_elem = 0;
            uint32_t elements_in_file = file_size / (128 * 4);
            if (_max_elements == 0)
                elements = elements_in_file;
            else
                elements = std::min(_max_elements, elements_in_file);
            dim = 128;

            tot_data_size = 1ll * elements * (dim * data_type_len + offset_per_elem) - offset_per_elem;
        }
        else if (data_name == "glove")
        {
            offset_per_elem = 0;
            uint32_t elements_in_file = file_size / (300 * 4);
            if (_max_elements == 0)
                elements = elements_in_file;
            else
                elements = std::min(_max_elements, elements_in_file);
            dim = 300;

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
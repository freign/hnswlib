#include "ArgParser.h"

// Default constructor definition
CommandLineOptions::CommandLineOptions() : maxElements(0) {}

// Parameterized constructor definition
CommandLineOptions::CommandLineOptions(const std::string& dir, const std::string& name, int maxElems)
    : dataDir(dir), dataName(name), maxElements(maxElems) {}

// ArgParser function definition
CommandLineOptions ArgParser(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " [point_data_path] [query_data_path] [data_name] [max_elements] [pq cfg file]" << std::endl;
        return CommandLineOptions();
    }
    std::string point_data_path = argv[1];
    std::string query_data_path = argv[2];
    std::string data_name = argv[3];
    int max_elements = std::atoi(argv[4]);
    // int use_pq = std::atoi(argv[5]);

    CommandLineOptions opt;
    opt.dataName = data_name;
    opt.point_data_path = point_data_path;
    opt.query_data_path = query_data_path;
    opt.maxElements = max_elements;
    // opt.use_pq = use_pq;
    opt.pq_cfg_file = argv[5];

    fs::path pathObj(opt.point_data_path);
    opt.dataDir = pathObj.parent_path().string(); // Use .string() instead of .c_str()

    std::cout << "Data Directory: " << opt.dataDir << std::endl;
    std::cout << "Point Data Path: " << opt.point_data_path << std::endl;
    std::cout << "Query Data Path: " << opt.query_data_path << std::endl;
    std::cout << "Data Name: " << opt.dataName << std::endl;
    std::cout << "Max Elements: " << opt.maxElements << std::endl;
    std::cout << "PQ cfg file: " << opt.pq_cfg_file << std::endl;

    return opt;
}

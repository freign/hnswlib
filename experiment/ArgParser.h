#pragma once
#include <bits/stdc++.h>
#include <filesystem>

namespace fs = std::filesystem;

struct CommandLineOptions {
    std::string dataDir;    // 数据目录
    std::string dataName;   // 数据名称
    std::string point_data_path;
    std::string query_data_path;
    int maxElements;        // 最大元素数量

    // 构造函数，可以提供默认值
    CommandLineOptions() : maxElements(0) {} // 默认构造函数

    // 带参数的构造函数，用于直接初始化所有成员变量
    CommandLineOptions(const std::string& dir, const std::string& name, int maxElems)
        : dataDir(dir), dataName(name), maxElements(maxElems) {}
};

CommandLineOptions ArgParser(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " [point_data_path] [query_data_path] [data_name] [max_elements]" << std::endl;
        return CommandLineOptions();
    }
    std::string point_data_path = argv[1];
    std::string query_data_path = argv[2];
    std::string data_name = argv[3];
    int max_elements = std::atoi(argv[4]);


    CommandLineOptions opt;
    opt.dataName = data_name;
    opt.point_data_path = point_data_path;
    opt.query_data_path = query_data_path;
    opt.maxElements = max_elements;

    fs::path pathObj(opt.point_data_path);
    opt.dataDir = pathObj.parent_path().c_str();

    std::cout << "Data Directory: " << opt.dataDir << std::endl;
    std::cout << "Point Data Path: " << opt.point_data_path << std::endl;
    std::cout << "Query Data Path: " << opt.query_data_path << std::endl;
    std::cout << "Data Name: " << opt.dataName << std::endl;
    std::cout << "Max Elements: " << opt.maxElements << std::endl;


    return opt;
}
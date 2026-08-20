#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Module.hpp>

#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace MNN::Express;

namespace {

std::vector<int> parseShape(const std::string& text) {
    std::vector<int> shape;
    std::stringstream stream(text);
    std::string part;
    while (std::getline(stream, part, ',')) {
        int value = std::stoi(part);
        if (value <= 0) throw std::runtime_error("shape dimensions must be positive");
        shape.push_back(value);
    }
    if (shape.empty()) throw std::runtime_error("shape is empty");
    return shape;
}

size_t elementCount(const std::vector<int>& shape) {
    size_t count = 1;
    for (int value : shape) count *= static_cast<size_t>(value);
    return count;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "Usage: moss_mnn_graph_probe model input_name output_name input_shape input.bin output.bin threads\n";
        return 2;
    }
    try {
        auto shape = parseShape(argv[4]);
        size_t inputCount = elementCount(shape);
        std::vector<float> values(inputCount);
        std::ifstream inputFile(argv[5], std::ios::binary);
        inputFile.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(float));
        if (!inputFile || inputFile.peek() != std::ifstream::traits_type::eof()) {
            throw std::runtime_error("input file size differs from input shape");
        }

        MNN::BackendConfig backendConfig;
        int threads = std::stoi(argv[7]);
        std::shared_ptr<Executor> executor(Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, threads));
        ExecutorScope scope(executor);
        Module::Config moduleConfig;
        moduleConfig.rearrange = true;
        std::unique_ptr<Module, decltype(&Module::destroy)> module(
            Module::load({argv[2]}, {argv[3]}, argv[1], &moduleConfig), Module::destroy);
        if (!module) throw std::runtime_error("failed to load MNN module");

        auto input = _Input(shape, NCHW, halide_type_of<float>());
        std::memcpy(input->writeMap<float>(), values.data(), values.size() * sizeof(float));
        auto outputs = module->onForward({input});
        if (outputs.size() != 1 || outputs[0] == nullptr || outputs[0]->getInfo() == nullptr) {
            throw std::runtime_error("MNN module did not produce one materialized output");
        }
        const auto* info = outputs[0]->getInfo();
        const float* outputValues = outputs[0]->readMap<float>();
        if (outputValues == nullptr) throw std::runtime_error("failed to read MNN output");
        std::ofstream outputFile(argv[6], std::ios::binary);
        outputFile.write(
            reinterpret_cast<const char*>(outputValues),
            static_cast<size_t>(info->size) * sizeof(float));
        if (!outputFile) throw std::runtime_error("failed to write MNN output");
        std::cout << "{\"elements\":" << info->size << ",\"shape\":[";
        for (size_t index = 0; index < info->dim.size(); ++index) {
            if (index) std::cout << ',';
            std::cout << info->dim[index];
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

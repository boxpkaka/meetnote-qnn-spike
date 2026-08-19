#include <audio/audio.hpp>

#include <fstream>

int main(int argc, char** argv) {
    if (argc != 3) {
        return 2;
    }
    auto loaded = MNN::AUDIO::load(argv[1], 16000, 0, 30 * 16000);
    if (loaded.first == nullptr || loaded.second != 16000) {
        return 3;
    }
    auto features = MNN::AUDIO::whisper_fbank(loaded.first, 16000, 80, 400, 160, 30);
    auto info = features == nullptr ? nullptr : features->getInfo();
    auto values = features == nullptr ? nullptr : features->readMap<float>();
    if (info == nullptr || values == nullptr || info->size != 80 * 3000) {
        return 4;
    }
    std::ofstream output(argv[2], std::ios::binary);
    output.write(reinterpret_cast<const char*>(values), info->size * sizeof(float));
    return output.good() ? 0 : 5;
}

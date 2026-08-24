#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "tokenizer.hpp"

using MNN::Transformer::Tokenizer;

namespace {

constexpr int kSampleStride = 1280;
constexpr int kAudioPad = 151671;
constexpr int kAudioStart = 151669;
constexpr int kAudioEnd = 151670;
constexpr float kAudioTokensPerSecond = 12.5f;
constexpr int kMarkerSeconds = 5;

const char* kDefaultPrompt =
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。\n";

std::vector<int> audioSpan(Tokenizer* tokenizer, int audioTokens) {
    const int tokensPerMarker = static_cast<int>(kAudioTokensPerSecond * kMarkerSeconds);
    const int duration = static_cast<int>(audioTokens / kAudioTokensPerSecond);
    std::vector<int> ids;
    int consumed = 0;
    for (int second = kMarkerSeconds; second <= duration; second += kMarkerSeconds) {
        const int position = (second / kMarkerSeconds) * tokensPerMarker;
        ids.insert(ids.end(), position - consumed, kAudioPad);
        consumed = position;
        for (char digit : std::to_string(second)) {
            auto digitIds = tokenizer->encode(std::string(1, digit));
            if (digitIds.size() != 1) {
                throw std::runtime_error("time marker digit is not one tokenizer token");
            }
            ids.push_back(digitIds[0]);
        }
    }
    ids.insert(ids.end(), audioTokens - consumed, kAudioPad);
    return ids;
}

std::vector<int> expand(
    Tokenizer* tokenizer,
    const std::string& rendered,
    const std::string& placeholder,
    const std::vector<int>& replacement) {
    auto position = rendered.find(placeholder);
    if (position == std::string::npos || rendered.find(placeholder, position + placeholder.size()) != std::string::npos) {
        throw std::runtime_error("rendered prompt must contain exactly one audio placeholder");
    }
    auto before = tokenizer->encode(rendered.substr(0, position));
    auto after = tokenizer->encode(rendered.substr(position + placeholder.size()));
    before.insert(before.end(), replacement.begin(), replacement.end());
    before.insert(before.end(), after.begin(), after.end());
    return before;
}

void writeIds(std::ostream& output, const std::vector<int>& ids) {
    output << '[';
    for (size_t index = 0; index < ids.size(); ++index) {
        if (index > 0) output << ',';
        output << ids[index];
    }
    output << ']';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: moss_tokenizer_probe tokenizer.mtok sample_count" << std::endl;
        return 2;
    }
    std::unique_ptr<Tokenizer> tokenizer(Tokenizer::createTokenizer(argv[1]));
    if (!tokenizer) {
        std::cerr << "failed to load tokenizer" << std::endl;
        return 1;
    }
    int64_t samples = std::stoll(argv[2]);
    if (samples <= 0 || samples > 300LL * 16000) {
        std::cerr << "sample_count must represent 1 to 300 seconds" << std::endl;
        return 2;
    }
    int audioTokens = static_cast<int>((samples - 1) / kSampleStride + 1);
    auto span = audioSpan(tokenizer.get(), audioTokens);

    const std::string runtimePlaceholder = "<audio>input</audio>";
    auto runtimeRendered = tokenizer->apply_chat_template(
        runtimePlaceholder + "\n" + std::string(kDefaultPrompt));
    std::vector<int> runtimeReplacement = {kAudioStart};
    runtimeReplacement.insert(runtimeReplacement.end(), span.begin(), span.end());
    runtimeReplacement.push_back(kAudioEnd);
    auto runtimeIds = expand(
        tokenizer.get(), runtimeRendered, runtimePlaceholder, runtimeReplacement);

    const std::string officialPlaceholder = "<|audio_pad|>";
    auto officialRendered = tokenizer->apply_chat_template(
        "<|audio_start|>" + officialPlaceholder + "<|audio_end|>\n" +
        std::string(kDefaultPrompt));
    auto officialIds = expand(tokenizer.get(), officialRendered, officialPlaceholder, span);

    auto mismatch = std::mismatch(
        officialIds.begin(), officialIds.end(), runtimeIds.begin(), runtimeIds.end());
    size_t firstMismatch = mismatch.first == officialIds.end()
        ? std::min(officialIds.size(), runtimeIds.size())
        : static_cast<size_t>(mismatch.first - officialIds.begin());
    bool exact = officialIds == runtimeIds;
    std::cout << "{\"format\":\"meetnote.moss_full_prompt_contract.v1\","
              << "\"status\":\"" << (exact ? "valid" : "invalid") << "\","
              << "\"sample_count\":" << samples << ','
              << "\"audio_embedding_count\":" << audioTokens << ','
              << "\"official_token_count\":" << officialIds.size() << ','
              << "\"runtime_token_count\":" << runtimeIds.size() << ','
              << "\"first_mismatch\":" << firstMismatch << ','
              << "\"official_input_ids\":";
    writeIds(std::cout, officialIds);
    std::cout << ",\"runtime_input_ids\":";
    writeIds(std::cout, runtimeIds);
    std::cout << "}" << std::endl;
    return exact ? 0 : 1;
}

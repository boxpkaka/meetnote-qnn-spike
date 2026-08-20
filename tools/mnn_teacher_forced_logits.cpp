#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "llm/llm.hpp"

using MNN::Transformer::Llm;
using MNN::Transformer::LlmContext;
using MNN::Transformer::LlmStatus;
using MNN::Transformer::MultimodalPrompt;
using MNN::Transformer::PromptAudioPart;

namespace {

std::string readFile(const std::string& path) {
    std::ifstream input(path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::string qwenChatPrompt(const std::string& userPrompt) {
    std::ostringstream out;
    out << "<|im_start|>system\n"
        << "你是 MeetNote 的端侧会议纪要结构化抽取引擎。不要输出思考过程，不要输出 Markdown，只输出用户要求的 JSON。"
        << "<|im_end|>\n"
        << "<|im_start|>user\n"
        << userPrompt
        << "<|im_end|>\n"
        << "<|im_start|>assistant\n"
        << "<think>\n\n</think>\n\n";
    return out.str();
}

std::string escapeJson(const std::string& value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(ch)
                        << std::dec << std::setw(0);
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    return out.str();
}

void appendJsonFloat(std::ostream& out, float value) {
    if (std::isfinite(value)) {
        out << std::setprecision(9) << value;
    } else {
        out << "null";
    }
}

void writeJsonIntArray(std::ostream& output, const std::vector<int>& values) {
    output << "[";
    for (size_t index = 0; index < values.size(); ++index) {
        if (index > 0) {
            output << ",";
        }
        output << values[index];
    }
    output << "]";
}

bool writeTokenIds(
        const std::string& path,
        const std::vector<int>& promptIds,
        const std::vector<int>& referenceIds) {
    if (path.empty()) {
        return true;
    }
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output.is_open()) {
        return false;
    }
    output << "{\"format\":\"meetnote.token_ids.v1\",\"prompt_ids\":";
    writeJsonIntArray(output, promptIds);
    output << ",\"reference_ids\":";
    writeJsonIntArray(output, referenceIds);
    output << "}\n";
    return output.good();
}

int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool endsWith(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
            value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int parseLayer(const std::string& name) {
    const std::vector<std::string> prefixes = {"/blocks.", "/layers."};
    size_t start = std::string::npos;
    std::string prefix;
    for (const auto& candidate : prefixes) {
        start = name.find(candidate);
        if (start != std::string::npos) {
            prefix = candidate;
            break;
        }
    }
    if (prefix.empty()) {
        return -1;
    }
    const size_t numberStart = start + prefix.size();
    const size_t end = name.find("/", numberStart);
    if (end == std::string::npos || end == numberStart) {
        return -1;
    }
    const std::string number = name.substr(numberStart, end - numberStart);
    if (!std::all_of(number.begin(), number.end(), [](unsigned char ch) {
            return std::isdigit(ch);
        })) {
        return -1;
    }
    return std::stoi(number);
}

class BoundaryTraceWriter {
public:
    explicit BoundaryTraceWriter(const std::string& prefix)
        : metadata_(prefix + ".layers.jsonl", std::ios::out | std::ios::trunc),
          tensors_(prefix + ".layers.f32", std::ios::out | std::ios::binary | std::ios::trunc) {
    }

    bool isOpen() const {
        return metadata_.is_open() && tensors_.is_open();
    }

    bool capture(const std::vector<MNN::Tensor*>& outputs, const MNN::OperatorInfo* info) {
        if (outputs.empty() || info == nullptr) {
            return true;
        }
        const std::string& name = info->name();
        const int layer = parseLayer(name);
        if (layer >= 0) {
            currentLayer_ = layer;
        }
        const int effectiveLayer = layer >= 0 ? layer : currentLayer_;
        if (effectiveLayer < 0 || effectiveLayer > 35) {
            return true;
        }
        const std::string block = "/blocks." + std::to_string(effectiveLayer);
        if (name == block + "/Reshape_output_0") {
            return write("hidden", effectiveLayer, name, outputs[0]);
        }
        if (endsWith(name, "/input_layernorm/Mul_1_output_0")) {
            return write("q_proj_input", effectiveLayer, name, outputs[0]);
        }
        if (name == "/Add_output_0") {
            return write("attn_q", effectiveLayer, name, outputs[0]);
        }
        if (name == "/Add_1_output_0") {
            return write("attn_k", effectiveLayer, name, outputs[0]);
        }
        if (endsWith(name, "/self_attn/v_proj/Linear")) {
            return write("attn_v", effectiveLayer, name, outputs[0]);
        }
        if (endsWith(name, "/self_attn/Add_output_0")) {
            return write("attn_q", effectiveLayer, name, outputs[0]);
        }
        if (endsWith(name, "/self_attn/Add_1_output_0")) {
            return write("attn_k", effectiveLayer, name, outputs[0]);
        }
        if (endsWith(name, "/self_attn/Reshape_2_output_0")) {
            return write("attn_v", effectiveLayer, name, outputs[0]);
        }
        if (endsWith(name, "/self_attn/q_proj/FakeLinear_output_0") ||
            endsWith(name, "/self_attn/q_proj/Linear/post_reshape") ||
            endsWith(name, "/self_attn/q_proj/Linear")) {
            return write("q_proj", effectiveLayer, name, outputs[0]);
        }
        if (endsWith(name, "/self_attn/q_norm/Mul_1_output_0") ||
            name == "/q_norm/Mul_1_output_0") {
            return write("q_norm", effectiveLayer, name, outputs[0]);
        }
        return true;
    }

    bool captureInputs(
            const std::vector<MNN::Tensor*>& inputs,
            const MNN::OperatorInfo* info) {
        if (inputs.empty() || info == nullptr) {
            return true;
        }
        if (endsWith(info->name(), "/self_attn/q_proj/Linear")) {
            const int layer = parseLayer(info->name());
            if (layer >= 0 && layer <= 35) {
                currentLayer_ = layer;
                return write("q_proj_input", layer, info->name(), inputs[0]);
            }
        }
        if (endsWith(info->name(), "/input_layernorm/Mul_1_output_0")) {
            const int parsedLayer = parseLayer(info->name());
            currentLayer_ = parsedLayer >= 0 ? parsedLayer : currentLayer_ + 1;
            if (currentLayer_ > 35) {
                return false;
            }
            return write("hidden", currentLayer_, info->name(), inputs[0]);
        }
        if (endsWith(info->name(), "/self_attn/q_norm/Mul_1_output_0")) {
            return write("q_proj", 0, info->name(), inputs[0]);
        }
        return true;
    }

    bool good() const {
        return records_ >= 3 && metadata_.good() && tensors_.good();
    }

private:
    bool write(
            const char* kind,
            int layer,
            const std::string& opName,
            MNN::Tensor* deviceTensor) {
        const std::string key = std::to_string(layer) + ":" + kind;
        if (!captured_.insert(key).second) {
            return true;
        }
        if (deviceTensor == nullptr || deviceTensor->elementSize() <= 0) {
            return false;
        }
        std::shared_ptr<MNN::Tensor> hostTensor(
                new MNN::Tensor(deviceTensor, deviceTensor->getDimensionType()));
        if (!deviceTensor->copyToHostTensor(hostTensor.get())) {
            return false;
        }
        const auto type = hostTensor->getType();
        const float* values = hostTensor->host<float>();
        if (type.code != halide_type_float || type.bits != 32 || values == nullptr) {
            return false;
        }

        const int elements = hostTensor->elementSize();
        metadata_ << "{\"kind\":\"" << kind
                  << "\",\"layer\":" << layer
                  << ",\"op\":\"" << escapeJson(opName)
                  << "\",\"offset_floats\":" << offsetFloats_
                  << ",\"elements\":" << elements
                  << ",\"shape\":[";
        for (int dimension = 0; dimension < hostTensor->dimensions(); ++dimension) {
            if (dimension > 0) {
                metadata_ << ",";
            }
            metadata_ << hostTensor->length(dimension);
        }
        metadata_ << "]}\n";
        tensors_.write(
                reinterpret_cast<const char*>(values),
                static_cast<std::streamsize>(elements) * sizeof(float));
        offsetFloats_ += elements;
        ++records_;
        return metadata_.good() && tensors_.good();
    }

    std::ofstream metadata_;
    std::ofstream tensors_;
    int64_t offsetFloats_ = 0;
    int records_ = 0;
    int currentLayer_ = -1;
    std::set<std::string> captured_;
};

void writeStep(
        std::ostream& output,
        Llm* llm,
        int step,
        int inputToken,
        int targetToken,
        const float* logits,
        int vocabSize,
        int topK,
        int64_t forwardMs) {
    std::vector<int> indices(static_cast<size_t>(vocabSize));
    std::iota(indices.begin(), indices.end(), 0);
    const int selected = std::min(topK, vocabSize);
    std::partial_sort(
            indices.begin(),
            indices.begin() + selected,
            indices.end(),
            [logits](int lhs, int rhs) {
                return logits[lhs] > logits[rhs];
            });

    const float targetLogit = logits[targetToken];
    int targetRank = 1;
    int finiteCount = 0;
    for (int token = 0; token < vocabSize; ++token) {
        if (std::isfinite(logits[token])) {
            ++finiteCount;
        }
        if (logits[token] > targetLogit) {
            ++targetRank;
        }
    }

    output << "{\"type\":\"step\""
           << ",\"step\":" << step
           << ",\"input_token\":" << inputToken
           << ",\"input_piece\":\"" << escapeJson(llm->tokenizer_decode(inputToken)) << "\""
           << ",\"target_token\":" << targetToken
           << ",\"target_piece\":\"" << escapeJson(llm->tokenizer_decode(targetToken)) << "\""
           << ",\"target_logit\":";
    appendJsonFloat(output, targetLogit);
    output << ",\"target_rank\":" << targetRank
           << ",\"finite_logits\":" << finiteCount
           << ",\"forward_ms\":" << forwardMs
           << ",\"top\":[";
    for (int rank = 0; rank < selected; ++rank) {
        if (rank > 0) {
            output << ",";
        }
        const int token = indices[rank];
        output << "{\"token\":" << token
               << ",\"piece\":\"" << escapeJson(llm->tokenizer_decode(token)) << "\""
               << ",\"logit\":";
        appendJsonFloat(output, logits[token]);
        output << "}";
    }
    output << "]}\n";
}

int parsePositiveInt(const char* value, const char* label) {
    std::istringstream input(value);
    int parsed = 0;
    input >> parsed;
    if (!input || parsed <= 0) {
        std::cerr << label << " must be a positive integer\n";
        return -1;
    }
    return parsed;
}

int parseNonNegativeInt(const char* value, const char* label) {
    std::istringstream input(value);
    int parsed = -1;
    input >> parsed;
    if (!input || parsed < 0) {
        std::cerr << label << " must be a non-negative integer\n";
        return -1;
    }
    return parsed;
}

}  // namespace

int main(int argc, const char* argv[]) {
    if (argc < 5 || argc > 11) {
        std::cerr
                << "Usage: " << argv[0]
                << " <config.json> <prompt.txt> <reference.txt> <output-prefix>"
                << " [max-steps=48; 0=tokenize-only] [top-k=20] [threads=4]"
                << " [trace-step=-1]"
                << " [token-ids-output]"
                << " [moss-wav; reference file contains whitespace-separated token IDs]\n";
        return 2;
    }

    const int maxSteps = argc >= 6 ? parseNonNegativeInt(argv[5], "max-steps") : 48;
    const int topK = argc >= 7 ? parsePositiveInt(argv[6], "top-k") : 20;
    const int threads = argc >= 8 ? parsePositiveInt(argv[7], "threads") : 4;
    const int traceStep = argc >= 9 ? std::stoi(argv[8]) : -1;
    const std::string tokenIdsOutput = argc >= 10 ? argv[9] : "";
    const std::string mossWav = argc >= 11 ? argv[10] : "";
    if (maxSteps < 0 || topK < 0 || threads < 0) {
        return 2;
    }
    if (maxSteps == 0 && tokenIdsOutput.empty()) {
        std::cerr << "token-ids-output is required when max-steps is 0\n";
        return 2;
    }
    if (traceStep >= maxSteps || traceStep < -1) {
        std::cerr << "trace-step must be -1 or less than max-steps\n";
        return 2;
    }

    const std::string prompt = readFile(argv[2]);
    const std::string reference = readFile(argv[3]);
    if (prompt.empty() || reference.empty()) {
        std::cerr << "prompt and reference files must be non-empty\n";
        return 2;
    }

    std::unique_ptr<Llm> llm(Llm::createLLM(argv[1]));
    if (!llm) {
        std::cerr << "failed to create MNN LLM\n";
        return 1;
    }
    const std::string prefix = argv[4];
    const char* hiddenPrefixEnv = std::getenv("MOSS_TEACHER_HIDDEN_PREFIX");
    const std::string hiddenPrefix = hiddenPrefixEnv == nullptr ? "" : hiddenPrefixEnv;
    std::ofstream hiddenMetadata;
    std::ofstream hiddenBinary;
    if (!hiddenPrefix.empty()) {
        hiddenMetadata.open(hiddenPrefix + ".jsonl", std::ios::out | std::ios::trunc);
        hiddenBinary.open(hiddenPrefix + ".f32", std::ios::out | std::ios::binary | std::ios::trunc);
        if (!hiddenMetadata.is_open() || !hiddenBinary.is_open()) {
            std::cerr << "failed to open hidden-state output files\n";
            return 1;
        }
    }
    std::unique_ptr<BoundaryTraceWriter> boundaryTrace;
    bool traceActive = false;
    std::ostringstream config;
    config << "{\"async\":false,\"enable_debug\":"
           << (traceStep >= 0 ? "true" : "false")
           << ",\"backend_type\":\"cpu\",\"thread_num\":" << threads;
    if (!hiddenPrefix.empty()) {
        config << ",\"hidden_states\":true";
    }
    if (mossWav.empty()) {
        config << ",\"jinja\":{\"context\":{\"enable_thinking\":false}}";
    }
    config << "}";
    llm->set_config(config.str());
    llm->set_config("{\"tmp_path\":\"/tmp/meetnote-mnn-teacher\"}");
    if (traceStep >= 0) {
        boundaryTrace.reset(new BoundaryTraceWriter(prefix));
        if (!boundaryTrace->isOpen()) {
            std::cerr << "failed to open boundary trace files\n";
            return 1;
        }
        llm->setDebugCallback(
                [&boundaryTrace, &traceActive](
                        const std::vector<MNN::Tensor*>& tensors,
                        const MNN::OperatorInfo* info) {
                    return !traceActive || boundaryTrace->captureInputs(tensors, info);
                },
                [&boundaryTrace, &traceActive](
                        const std::vector<MNN::Tensor*>& tensors,
                        const MNN::OperatorInfo* info) {
                    return !traceActive || boundaryTrace->capture(tensors, info);
                });
    }
    if (!llm->load()) {
        std::cerr << "failed to load MNN LLM\n";
        return 1;
    }

    std::ofstream metadata(prefix + ".jsonl", std::ios::out | std::ios::trunc);
    std::ofstream binary(prefix + ".f32", std::ios::out | std::ios::binary | std::ios::trunc);
    if (!metadata.is_open() || !binary.is_open()) {
        std::cerr << "failed to open output files\n";
        return 1;
    }

    llm->reset();
    llm->generate_init();
    std::vector<int> inputIds;
    std::vector<int> referenceIds;
    if (mossWav.empty()) {
        inputIds = llm->tokenizer_encode(qwenChatPrompt(prompt));
        referenceIds = llm->tokenizer_encode(reference);
    } else {
        MultimodalPrompt multimodal;
        multimodal.prompt_template = llm->apply_chat_template(
            "<audio>input</audio>\n" + prompt);
        PromptAudioPart audio;
        audio.file_path = mossWav;
        multimodal.audios["input"] = audio;
        inputIds = llm->tokenizer_encode(multimodal);
        std::istringstream tokenStream(reference);
        int token = 0;
        while (tokenStream >> token) referenceIds.push_back(token);
    }
    if (!writeTokenIds(tokenIdsOutput, inputIds, referenceIds)) {
        std::cerr << "failed to write token IDs: " << tokenIdsOutput << "\n";
        return 1;
    }
    if (maxSteps == 0) {
        std::cout << "token IDs written to " << tokenIdsOutput << "\n";
        return 0;
    }
    if (referenceIds.size() < 2) {
        std::cerr << "reference must encode to at least two tokens\n";
        return 2;
    }

    llm->generate(inputIds, 0);
    LlmContext* context = const_cast<LlmContext*>(llm->getContext());
    if (context->status == LlmStatus::INTERNAL_ERROR) {
        std::cerr << "prefill failed\n";
        return 1;
    }

    const int steps = std::min(static_cast<int>(referenceIds.size()) - 1, maxSteps);
    int vocabSize = 0;
    for (int step = 0; step < steps; ++step) {
        const int inputToken = referenceIds[step];
        const int targetToken = referenceIds[step + 1];
        context->history_tokens.push_back(inputToken);
        context->gen_seq_len += 1;
        if (step == traceStep) {
            traceActive = true;
        }
        const int64_t startMs = nowMs();
        auto logits = llm->forward({inputToken}, false);
        const int64_t forwardMs = nowMs() - startMs;
        if (step == traceStep) {
            traceActive = false;
        }
        context->gen_seq_len -= 1;
        if (logits == nullptr || logits->getInfo() == nullptr) {
            std::cerr << "decode returned no logits at step " << step << "\n";
            return 1;
        }

        const auto* info = logits->getInfo();
        if (info->dim.empty()) {
            std::cerr << "decode logits have no dimensions\n";
            return 1;
        }
        vocabSize = info->dim.back();
        if (vocabSize <= 0 || targetToken < 0 || targetToken >= vocabSize) {
            std::cerr << "invalid vocabulary or target token at step " << step << "\n";
            return 1;
        }
        const float* values = logits->readMap<float>();
        if (values == nullptr || info->size < vocabSize) {
            std::cerr << "decode logits are not readable at step " << step << "\n";
            return 1;
        }
        values += info->size - vocabSize;

        if (!hiddenPrefix.empty()) {
            auto outputs = llm->getOutputs();
            int hiddenIndex = llm->getOutputIndex("hidden_states");
            if (hiddenIndex < 0 || hiddenIndex >= static_cast<int>(outputs.size()) ||
                outputs[hiddenIndex] == nullptr || outputs[hiddenIndex]->getInfo() == nullptr) {
                std::cerr << "decode returned no hidden_states at step " << step << "\n";
                return 1;
            }
            auto hidden = outputs[hiddenIndex];
            const auto* hiddenInfo = hidden->getInfo();
            const float* hiddenValues = hidden->readMap<float>();
            if (hiddenValues == nullptr) {
                std::cerr << "hidden_states are not readable at step " << step << "\n";
                return 1;
            }
            hiddenBinary.write(
                reinterpret_cast<const char*>(hiddenValues),
                static_cast<std::streamsize>(hiddenInfo->size) * sizeof(float));
            hiddenMetadata << "{\"step\":" << step << ",\"elements\":" << hiddenInfo->size
                           << ",\"shape\":[";
            for (size_t index = 0; index < hiddenInfo->dim.size(); ++index) {
                if (index) hiddenMetadata << ',';
                hiddenMetadata << hiddenInfo->dim[index];
            }
            hiddenMetadata << "]}\n";
            if (!hiddenBinary.good() || !hiddenMetadata.good()) {
                std::cerr << "failed while writing hidden_states at step " << step << "\n";
                return 1;
            }
        }

        if (step == 0) {
            metadata << "{\"type\":\"header\""
                     << ",\"format\":\"meetnote.teacher_logits.v1\""
                     << ",\"dtype\":\"float32_le\""
                     << ",\"prompt_tokens\":" << inputIds.size()
                     << ",\"reference_tokens\":" << referenceIds.size()
                     << ",\"prompt_ids\":";
            writeJsonIntArray(metadata, inputIds);
            metadata << ",\"reference_ids\":";
            writeJsonIntArray(metadata, referenceIds);
            metadata << ",\"steps\":" << steps
                     << ",\"vocab_size\":" << vocabSize
                     << ",\"top_k\":" << topK
                     << "}\n";
        }
        binary.write(
                reinterpret_cast<const char*>(values),
                static_cast<std::streamsize>(vocabSize) * sizeof(float));
        writeStep(
                metadata,
                llm.get(),
                step,
                inputToken,
                targetToken,
                values,
                vocabSize,
                topK,
                forwardMs);
        if (!metadata.good() || !binary.good()) {
            std::cerr << "failed while writing step " << step << "\n";
            return 1;
        }
        std::cerr << "step " << (step + 1) << "/" << steps
                  << " forward_ms=" << forwardMs << "\n";
    }

    metadata << "{\"type\":\"footer\",\"steps_written\":" << steps << "}\n";
    if (boundaryTrace != nullptr && !boundaryTrace->good()) {
        std::cerr << "boundary trace did not capture all expected tensors\n";
        return 1;
    }
    std::cout << "prompt_tokens=" << inputIds.size()
              << " reference_tokens=" << referenceIds.size()
              << " steps=" << steps
              << " vocab_size=" << vocabSize << "\n";
    return 0;
}

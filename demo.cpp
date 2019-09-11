#include <iostream>
#include <map>
#include <neuralnetwork.hpp>
#include <string>
#include <vector>

using NetworkType = NeuralNetwork<1, 4, 8, 16, 8, 4, 1>;

constexpr double pi = 3.141592654;

namespace domain {
    constexpr double min = -5.0;
    constexpr double max = 5.0;
}

double sine(double x) noexcept {
    const auto t = (x - domain::min) / (domain::max - domain::min);
    return std::sin(t * 2 * pi) * 0.5 + 0.5;
}

double cosine(double x) noexcept {
    const auto t = (x - domain::min) / (domain::max - domain::min);
    return std::cos(t * 2 * pi) * 0.5 + 0.5;
}

double linear(double x) noexcept {
    return (x - domain::min) / (domain::max - domain::min);
}

double saw(double x) noexcept {
    x = linear(x);
    return x - std::floor(x);
}

double square(double x) noexcept {
    return std::round(saw(x));
}

double triangle(double x) noexcept {
    return 1.0 - 2.0 * std::abs(saw(x) - 0.5);
}

using FunctionSignature = double (*)(double) noexcept;
FunctionSignature theFunction = nullptr;

constexpr std::size_t graphWidth = 60;
constexpr std::size_t graphHeight = 30;

void printNetwork(const NetworkType& nn){
    std::array<std::size_t, graphWidth> heights;
    for (std::size_t gx = 0; gx < graphWidth; ++gx){
        const double t = (static_cast<double>(gx) / static_cast<double>(graphWidth - 1));
        const double x = t * (domain::max - domain::min) + domain::min;
        const auto y = nn.compute(gsl::span{&x, 1})[0];
        heights[gx] = static_cast<std::size_t>(std::round((1.0 - y) * graphHeight));
    }
    
    for (std::size_t gy = 0; gy < graphHeight; ++gy){
        for (std::size_t gx = 0; gx < graphWidth; ++gx){
            std::cout << (heights[gx] > gy ? ' ' : '#');
        }
        std::cout << '\n';
    }
}

void printFunction(){
    std::array<std::size_t, graphWidth> heights;
    for (std::size_t gx = 0; gx < graphWidth; ++gx){
        const double t = (static_cast<double>(gx) / static_cast<double>(graphWidth - 1));
        const double x = t * (domain::max - domain::min) + domain::min;
        const auto y = theFunction(x);
        heights[gx] = static_cast<std::size_t>(std::round((1.0 - y) * graphHeight));
    }
    
    for (std::size_t gy = 0; gy < graphHeight; ++gy){
        for (std::size_t gx = 0; gx < graphWidth; ++gx){
            std::cout << (heights[gx] > gy ? ' ' : '#');
        }
        std::cout << '\n';
    }
}

int main(int argc, char** argv){
    
    auto args = std::vector<std::string>{argv, argv + argc};

    assert(args.size() > 1);
    args.erase(args.begin());

    constexpr std::size_t ExamplesPerBatch = 1 << 15;
    std::size_t Batches = 256;

    if (args.size() > 0){
        static const std::map<std::string, FunctionSignature> fns = {
            {"linear", linear},
            {"sine", sine},
            {"sine2", [](double x) noexcept { return sine(2.0 * x); }},
            {"sine3", [](double x) noexcept { return sine(3.0 * x); }},
            {"cosine", cosine},
            {"cosine2", [](double x) noexcept { return cosine(2.0 * x); }},
            {"cosine3", [](double x) noexcept { return cosine(3.0 * x); }},
            {"square", square},
            {"square2", [](double x) noexcept { return square(2.0 * x); }},
            {"square3", [](double x) noexcept { return square(3.0 * x); }},
            {"triangle", triangle},
            {"triangle2", [](double x) noexcept { return triangle(2.0 * x); }},
            {"triangle3", [](double x) noexcept { return triangle(3.0 * x); }},
            {"saw2", [](double x) noexcept { return saw(2.0 * x); }},
            {"saw3", [](double x) noexcept { return saw(3.0 * x); }},
        };
        if (auto it = fns.find(args[0]); it != fns.end()){
            theFunction = it->second;
            args.erase(args.begin());
        } else {
            std::cout << "Whoops! \"" << args[0] << "\" is not a known function.\n";
            std::cout << "Please choose from one of the following.";
            for (const auto& nameFn : fns){
                std::cout << " - " << nameFn.first << '\n';
            }
            return 1;
        }
    } else {
        theFunction = sine;
    }

    if (args.size() > 0){
        // TODO: choose number of iterations
        // TODO: choose learning rate
    }

    auto rd = std::random_device{};
    auto re = std::default_random_engine{rd()};
    auto dist = std::uniform_real_distribution<double>{domain::min, domain::max};
    
    auto nn = NetworkType{};
    nn.randomizeWeights();

    std::cout << "\nStarting training...\n";
    
    for (std::size_t b = 0; b < Batches; ++b){
        //std::array<std::array<double, 2>, ExamplesPerBatch> inputs;
        std::array<double, ExamplesPerBatch> inputs;
        std::array<double, ExamplesPerBatch> expected_outputs;
        std::vector<std::pair<NetworkType::InputType, NetworkType::OutputType>> batch;

        for (std::size_t e = 0; e < ExamplesPerBatch; ++e){
            //inputs[e][0] = dist(re);
            //inputs[e][1] = dist(re);
            inputs[e] = dist(re);
            //expected_outputs[e] = inputs[e][0] + inputs[e][1];
            expected_outputs[e] = theFunction(inputs[e]);

            //auto i = gsl::span{&inputs[e][0], 2};
            auto i = gsl::span{&inputs[e], 1};
            auto o = gsl::span{&expected_outputs[e], 1};
            batch.push_back(std::pair{i, o});
        }

        auto loss = nn.train(batch, 0.01);

        std::cout << "Training batch " << b << ", loss: " << loss << "\n\n";
        printNetwork(nn);
        std::cout << '\n';
        std::cout.flush();
    }

    std::cout << "Training done.\n";

    std::cout << "\nExpected behavior:\n\n";
    printFunction();
    std::cout << "\nActual Behavior:\n\n";
    printNetwork(nn);

    std::cout << '\n';
}
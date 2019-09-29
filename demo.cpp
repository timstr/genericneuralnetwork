#include <iostream>
#include <functional>
#include <map>
#include <neuralnetwork.hpp>
#include <string>
#include <vector>

using NetworkType = NeuralNetwork<1, 16, 16, 16, 1>;

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

constexpr std::size_t graphWidth = 70;
constexpr std::size_t graphHeight = 30;

void print(std::function<double(double)> fn){
    std::array<double, graphWidth> values;
    for (std::size_t gx = 0; gx < graphWidth; ++gx){
        const double t = (static_cast<double>(gx) / static_cast<double>(graphWidth - 1));
        const double x = t * (domain::max - domain::min) + domain::min;
        values[gx] = fn(x);
    }

    for (std::size_t gy = 0; gy < graphHeight; ++gy){
        const auto y = static_cast<double>(gy);
        for (std::size_t gx = 0; gx < graphWidth; ++gx){
            const auto h = (1.0 - values[gx]) * static_cast<double>(graphHeight);
            if (h > y + 1.0){
                std::cout << ' ';
            } else if (h > y + (2.0 / 3.0)){
                std::cout << '_';
            } else if (h > y + (1.0 / 3.0)){
                std::cout << '=';
            } else {
                std::cout << '#';
            }
        }
        std::cout << '\n';
    }
}

void printNetwork(NetworkType& nn){
    const auto fn = [&](double x){
        std::array<double, 1> in = {x};
        auto out = nn.compute(in);
        return out[0];
    };
    print(fn);
}

void printFunction(){
    print(theFunction);
}

int main(int argc, char** argv){

    auto args = std::vector<std::string>{argv, argv + argc};

    assert(args.size() > 1);
    args.erase(args.begin());

    constexpr std::size_t ExamplesPerBatch = 64;
    std::size_t BatchesPerRun = 1 << 10;
    std::size_t NumRuns = 1 << 8;

    if (args.size() > 0){
        static const std::map<std::string, FunctionSignature> fns = {
            {"sigmoid", sigmoid},
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
            std::cout << "Please choose from one of the following.\n";
            for (const auto& nameFn : fns){
                std::cout << " - " << nameFn.first << '\n';
            }
            return 1;
        }
    } else {
        theFunction = linear;
    }

    double learningRate = 0.1;
    if (args.size() > 0){
        try {
            learningRate = std::stod(args[0]);
        } catch (...){
            std::cout << "Whoops! \"" << args[0] << "\" is not a valid learning rate.\n";
            std::cout << "Please choose a valid learning rate, for example 0.1 or 0.03\n";
            return 2;
        }
    }

    double momentumRatio = 0.75;

    // TODO: choose number of iterations

    auto rd = std::random_device{};
    auto re = std::default_random_engine{rd()};
    auto dist = std::uniform_real_distribution<double>{domain::min, domain::max};
    
    auto nn = NetworkType{};
    nn.randomizeWeights();

    std::cout << "Initial output:\n\n";

    printNetwork(nn);

    std::cout << "\nStarting training...\n";
    
    std::size_t iterationCount = 0;
    for (std::size_t r = 0; r < NumRuns; ++r){
        double loss = 0.0;
        for (std::size_t b = 0; b < BatchesPerRun; ++b){
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

            loss += nn.takeStep(batch, learningRate, momentumRatio);
            ++iterationCount;
        }
        loss /= static_cast<double>(BatchesPerRun);

        std::cout << "Training run " << r << ", loss: " << loss << ", " << iterationCount << " iterations so far\n\n";
        printNetwork(nn);
        std::cout << '\n';
        std::cout.flush();
    }

    std::cout << "Training done.\n";

    std::cout << "\nExpected behavior:\n\n";
    printFunction();
    std::cout << "\nActual Behavior:\n\n";
    printNetwork(nn);

    std::cout << "File save test:\n";

    nn.saveWeights("weights.dat");

    {
        auto nn2 = NetworkType{};
        nn2.loadWeights("weights.dat");

        std::cout << "Behavior after saving to and reading from file:\n\n";
        printNetwork(nn2);

        if (nn != nn2){
            std::cout << "Whoops, that is not the same network anymore. Drat.";
        } else {
            std::cout << "Nice! The network has identical weight values.";
        }
    }

    std::cout << '\n';
}

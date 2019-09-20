#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <random>
#include <gsl/span>

namespace detail {
    std::random_device rand_dev;
    std::default_random_engine rand_eng {rand_dev()};
    auto normal_dist = std::normal_distribution<double>{};

    template<std::size_t N>
    double loss(gsl::span<const double, N> v1, gsl::span<const double, N> v2) noexcept {
        double acc = 0;
        for (size_t i = 0; i < N; ++i){
            const auto d = v1[i] - v2[i];
            acc += d * d;
        }
        return acc;
    }

    template<typename T>
    void write(std::ofstream& ofs, const T& val){
        ofs.write(reinterpret_cast<const char*>(&val), sizeof(T));
        if (!ofs){
            throw std::runtime_error("Failed to write to file");
        }
    };

    template<typename T>
    T read(std::ifstream& ifs){
        T ret;
        ifs.read(reinterpret_cast<char*>(&ret), sizeof(T));
        if (!ifs){
            throw std::runtime_error("Failed to read from file");
        }
        return ret;
    }
}

double sigmoid(double x) noexcept {
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoidDerivative(double x) noexcept {
    const auto s = sigmoid(x);
    return s * (1.0 - s);
}

template<std::size_t Inputs, std::size_t Outputs>
class Weights {
public:
    double& operator()(std::size_t input, std::size_t output) noexcept {
        return const_cast<double&>(const_cast<const Weights&>(*this)(input, output));
    }
    const double& operator()(std::size_t input, std::size_t output) const noexcept {
        assert(input < Inputs);
        assert(output < Outputs);
        return values[input][output];
    }

    void randomize() noexcept {
        for (auto& a : values){
            for (auto& x : a){
                x = detail::normal_dist(detail::rand_eng);
            }
        }
    }

private:
    std::array<std::array<double, Outputs>, Inputs> values = {};
};


// Generic class representing a stack of neural layers
template<std::size_t...>
class LayerStack;

// Specialization of LayerStack for exactly one layer
// This is the input layer of the network
template<std::size_t Layer0Size>
class LayerStack<Layer0Size> {
public:
    LayerStack() noexcept
        : m_rawOutputs{}
        , m_transferredOutputs{} {
    }

    static constexpr std::size_t Neurons = Layer0Size;
    static constexpr std::size_t RootNeurons = Layer0Size;
    using InputType = gsl::span<const double, Neurons>;
    using RootInputType = InputType;
    using OutputType = InputType;

    OutputType compute(InputType inputs) const noexcept {
        std::copy(
            inputs.begin(),
            inputs.end(),
            m_rawOutputs.begin()
        );
        std::transform(
            inputs.begin(),
            inputs.end(),
            m_transferredOutputs.begin(),
            sigmoid
        );
        return m_transferredOutputs;
    }

    OutputType rawOutputs() const noexcept {
        return m_rawOutputs;
    }

    OutputType transferredOutputs() const noexcept {
        return m_transferredOutputs;
    }

private:
    mutable std::array<double, Neurons> m_rawOutputs;
    mutable std::array<double, Neurons> m_transferredOutputs;
};


// Specialization of LayerStack for two or more layers
// This class derives from LayerStack instantiated with the previous layers
template<std::size_t Layer0Size, std::size_t Layer1Size, std::size_t... OtherLayerSizes>
class LayerStack<Layer0Size, Layer1Size, OtherLayerSizes...> : public LayerStack<Layer1Size, OtherLayerSizes...> {
public:
    LayerStack() noexcept
        : m_weights{}
        , m_weightGradients{}
        , m_rawOutputs{}
        , m_transferredOutputs{} {
    }

    using Previous = LayerStack<Layer1Size, OtherLayerSizes...>;

    constexpr static std::size_t Neurons = Layer0Size;
    constexpr static std::size_t RootNeurons = Previous::RootNeurons;

    using InputType = typename Previous::OutputType;
    using RootInputType = typename Previous::RootInputType;
    using OutputType = gsl::span<const double, Neurons>;

    OutputType compute(RootInputType rootInputs) const noexcept {
        const auto inputs = this->Previous::compute(rootInputs);

        for (std::size_t o = 0; o < Neurons; ++o){
            double x = 0.0;
            for (std::size_t i = 0; i < Previous::Neurons; ++i){
                assert(i < static_cast<std::size_t>(inputs.size()));
                x += m_weights(i, o) * inputs[i];
            }
            x += m_weights(Previous::Neurons, o) * 1.0;
            assert(o < m_rawOutputs.size());
            assert(o < m_transferredOutputs.size());
            m_rawOutputs[o] = x;
            m_transferredOutputs[o] = sigmoid(x);
        }
        return m_transferredOutputs;
    }

    // outputDerivatives is the derivative of the cost w.r.t. each output neuron (including sigmoid)
    void backPropagateAdd(OutputType outputDerivatives){
        std::array<double, Previous::Neurons> inputDerivatives = {};
        const auto rawInputs = this->Previous::rawOutputs();
        const auto transferredInputs = this->Previous::transferredOutputs();

        for (std::size_t o = 0; o < Neurons; ++o){
            for (std::size_t i = 0; i < Previous::Neurons; ++i){
                inputDerivatives[i] += m_weights(i, o) * outputDerivatives[o];
                m_weightGradients(i, o) += outputDerivatives[o] * transferredInputs[i];
            }
            m_weightGradients(Previous::Neurons, o) += outputDerivatives[o] * 1.0;
        }
        for (std::size_t i = 0; i < Previous::Neurons; ++i){
            inputDerivatives[i] *= sigmoidDerivative(rawInputs[i]);
        }
        
        if constexpr (sizeof...(OtherLayerSizes) > 0){
            this->Previous::backPropagateAdd(inputDerivatives);
        }
    }

    void scaleGradients(double scale){
        for (std::size_t i = 0; i < Previous::Neurons + 1; ++i){
            for (std::size_t j = 0; j < Neurons; ++j){
                m_weightGradients(i, j) *= scale;
            }
        }
        if constexpr (sizeof...(OtherLayerSizes) > 0){
            this->Previous::scaleGradients(scale);
        }
    }

    void adjustWeights(double stepSize){
        for (std::size_t i = 0; i < Previous::Neurons + 1; ++i){
            for (std::size_t j = 0; j < Neurons; ++j){
                m_weights(i, j) += stepSize * m_weightGradients(i, j);
            }
        }
        if constexpr (sizeof...(OtherLayerSizes) > 0){
            this->Previous::adjustWeights(stepSize);
        }
    }
    
    template<std::size_t Layer>
    decltype(auto) weights() noexcept {
        if constexpr (Layer == 0){
            auto& ret = m_weights;
            return ret;
        } else {
            return this->Previous::template weights<Layer - 1>();
        }
    }

    void randomizeWeights() noexcept {
        m_weights.randomize();
        if constexpr (sizeof...(OtherLayerSizes) > 0){
            this->Previous::randomizeWeights();
        }
    }

    OutputType rawOutputs() const noexcept {
        return m_rawOutputs;
    }

    OutputType transferredOutputs() const noexcept {
        return m_transferredOutputs;
    }

    void writeLayerSizes(std::ofstream& ofs) const {
        detail::write<std::size_t>(ofs, Neurons);
        if constexpr (sizeof...(OtherLayerSizes) > 0){
            this->Previous::writeLayerSizes(ofs);
        } else {
            detail::write<std::size_t>(ofs, Previous::Neurons);
        }
    }

    void writeWeights(std::ofstream& ofs) const {
        for (std::size_t i = 0; i < Previous::Neurons + 1; ++i){
            for (std::size_t j = 0; j < Neurons; ++j){
                detail::write<double>(ofs, m_weights(i, j));
            }
        }
        if constexpr (sizeof...(OtherLayerSizes) > 0){
            this->Previous::writeWeights(ofs);
        }
    }

    void checkLayerSizes(std::ifstream& ifs) const {
        auto n = detail::read<std::size_t>(ifs);
        if (n != Neurons){
            throw std::runtime_error("Network size mismatch while reading from file");
        }
        if constexpr (sizeof...(OtherLayerSizes) > 0){
            this->Previous::checkLayerSizes(ifs);
        } else {
            auto n = detail::read<std::size_t>(ifs);
            if (n != Previous::Neurons){
                throw std::runtime_error("Network size mismatch while reading from file");
            }    
        }
    }

    void readWeights(std::ifstream& ifs){
        for (std::size_t i = 0; i < Previous::Neurons + 1; ++i){
            for (std::size_t j = 0; j < Neurons; ++j){
                m_weights(i, j) = detail::read<double>(ifs);
            }
        }
        if constexpr (sizeof...(OtherLayerSizes) > 0){
            this->Previous::readWeights(ifs);
        }
    }

    bool equals(const LayerStack& other) const noexcept {
        for (std::size_t i = 0; i < Previous::Neurons + 1; ++i){
            for (std::size_t j = 0; j < Neurons; ++j){
                if (m_weights(i, j) != other.m_weights(i, j)){
                    return false;
                }
            }
        }
        if constexpr (sizeof...(OtherLayerSizes) > 0){
            return this->Previous::equals(other);
        } else {
            return true;
        }
    }

private:
    Weights<Previous::Neurons + 1, Neurons> m_weights;
    Weights<Previous::Neurons + 1, Neurons> m_weightGradients;
    mutable std::array<double, Neurons> m_rawOutputs;
    mutable std::array<double, Neurons> m_transferredOutputs;
};


template<std::size_t Layer0Size, std::size_t... OtherLayerSizes>
class NeuralNetwork {
public:
    using LayersType = LayerStack<Layer0Size, OtherLayerSizes...>;
    static constexpr std::size_t NumLayers = 1ul + sizeof...(OtherLayerSizes);

    static constexpr std::size_t InputNeurons = LayersType::RootNeurons;
    static constexpr std::size_t OutputNeurons = LayersType::Neurons;
    using InputType = typename LayersType::RootInputType;
    using OutputType = typename LayersType::OutputType;

    OutputType compute(InputType inputs) const noexcept {
        return layers.compute(inputs);
    }

    double takeStep(gsl::span<std::pair<InputType, OutputType>> examples, double stepSize, double momentumRatio = 0.0) noexcept {
        double lossAcc = 0.0;
        layers.scaleGradients(momentumRatio);
        for (const auto& [input, expectedOutput] : examples){
            // layers.zeroGradients(); // testing
            compute(input);
            const auto transferredOutputs = layers.transferredOutputs();
            const auto rawOutputs = layers.rawOutputs();
            
            lossAcc += detail::loss<OutputNeurons>(transferredOutputs, expectedOutput);
            
            std::array<double, OutputNeurons> outputDerivatives;

            for (std::size_t o = 0; o < OutputNeurons; ++o){
                outputDerivatives[o] = sigmoidDerivative(rawOutputs[o]) * (expectedOutput[o] - transferredOutputs[o]);
            }

            layers.backPropagateAdd(outputDerivatives);
            // layers.adjustWeights(stepSize); // testing
        }
        layers.adjustWeights(stepSize / static_cast<double>(examples.size()));
        return lossAcc / static_cast<double>(examples.size());
    }

    /*double train(gsl::span<std::pair<InputType, OutputType>> examples) noexcept {
        // TODO
        return 0.0;
    }*/
    
    void randomizeWeights() noexcept {
        layers.randomizeWeights();
    }

    template<std::size_t Layer>
    double& weights(std::size_t input, std::size_t output) noexcept {
        return layers.template weights<Layer>()(input, output);
    }

    void saveWeights(std::string filepath) const {
        std::ofstream ofs{filepath, std::ios::out | std::ios::binary};
        if (!ofs){
            throw std::runtime_error("Could not open file");
        }

        detail::write<std::size_t>(ofs, NumLayers);

        layers.writeLayerSizes(ofs);
        layers.writeWeights(ofs);
    }

    void loadWeights(std::string filepath){
        std::ifstream ifs{filepath, std::ios::in | std::ios::binary};
        if (!ifs){
            throw std::runtime_error("Could not open file");
        }

        auto n = detail::read<std::size_t>(ifs);
        if (n != NumLayers){
            throw std::runtime_error("Wrong number of layers");
        }

        layers.checkLayerSizes(ifs);
        layers.readWeights(ifs);
    }

    bool operator==(const NeuralNetwork& other) const noexcept {
        return layers.equals(other.layers);
    }

    bool operator!=(const NeuralNetwork& other) const noexcept {
        return !this->operator==(other);
    }

private:
    LayersType layers;
};

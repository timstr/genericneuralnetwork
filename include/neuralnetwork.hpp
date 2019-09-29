#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <random>
#include <gsl/span>

namespace detail {
    // Random number generator
    std::random_device rand_dev;
    std::default_random_engine rand_eng {rand_dev()};
    
    // Computes the sum of the squared error between two vectors of equal size
    // Effectively returns the inner product of those vectors
    template<std::size_t N>
    double loss(gsl::span<const double, N> v1, gsl::span<const double, N> v2) noexcept {
        double acc = 0.0;
        for (size_t i = 0; i < N; ++i){
            const auto d = v1[i] - v2[i];
            acc += d * d;
        }
        return acc;
    }

    // Writes a binary value of type T to an output stream
    template<typename T>
    void write(std::ofstream& ofs, const T& val){
        ofs.write(reinterpret_cast<const char*>(&val), sizeof(T));
        if (!ofs){
            throw std::runtime_error("Failed to write to file");
        }
    };

    // Reads a binary value of type T from an input stream
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

// The sigmoid activation function used by the network
double sigmoid(double x) noexcept {
    return 1.0 / (1.0 + std::exp(-x));
}

// Derivative of the sigmoid activation function
double sigmoidDerivative(double x) noexcept {
    const auto s = sigmoid(x);
    return s * (1.0 - s);
}

// Weights is a class for managing the weights between two consecutive
// fully connected layers of a neural network.
// - Inputs is the number of neurons in the previous layer
// - Outputs is the number of neurons in the subsequent layer
template<std::size_t Inputs, std::size_t Outputs>
class Weights {
public:
    // Overload of operator() to provide array-like access in two dimensions
    // Returns the weight between the previous neuron `input` and the
    // subsequent neuron `output`
    double& operator()(std::size_t input, std::size_t output) noexcept {
        return const_cast<double&>(const_cast<const Weights&>(*this)(input, output));
    }
    const double& operator()(std::size_t input, std::size_t output) const noexcept {
        assert(input < Inputs);
        assert(output < Outputs);
        return values[input][output];
    }

    // Assigns a random value to each weight
    void randomize() noexcept {
        const auto uniform_dist = std::uniform_real_distribution<double>{-0.5, 0.5};
        for (auto& a : values){
            for (auto& x : a){
                x = uniform_dist(detail::rand_eng);
            }
        }
    }

private:
    std::array<std::array<double, Outputs>, Inputs> values = {};
};


// LayerStack is a generic class representing a stack of fully-connected
// neural layers. The template arguments are the number of neurons in each
// layer, in reverse order.
// This a forward declaration. Specializations for 
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

    // The number of neurons in this layer
    static constexpr std::size_t Neurons = Layer0Size;

    // The number of input neurons for the entire layer stack to come
    // (defined here and inherited by other layers)
    static constexpr std::size_t RootNeurons = Layer0Size;

    // The input type accepted by this layer is a view to an array of doubles
    using InputType = gsl::span<const double, Neurons>;

    // The input type for the entire layer stack
    // (defined here and inherited by other layers)
    using RootInputType = InputType;

    // The output type of this layer is the same as the input type
    // (The first layer does only trivial work)
    using OutputType = InputType;

    // Template function for computing the network's result starting from
    // a given layer. Since this will always be the input layer, the given
    // layer must always be the zero'th.
    template<std::size_t Layer>
    OutputType computePartial() const noexcept {
        static_assert(Layer == 0);
        // No work to do, just return the transferred outputs which are
        // assumed to have been modified by setActivation() (see below)
        return m_transferredOutputs;
    }

    // Returns the non-transferred outputs (the sigmoid function has
    // not been applied)
    OutputType rawOutputs() const noexcept {
        return m_rawOutputs;
    }

    // Returns the transferred outputs (the raw outputs after applying the
    // sigmoid activation function)
    OutputType transferredOutputs() const noexcept {
        return m_transferredOutputs;
    }

    // Modifies the (raw) activation of a given neuron
    template<std::size_t Layer>
    void setActivation(std::size_t neuron, double value) noexcept {
        static_assert(Layer == 0);
        assert(neuron < Neurons);
        m_rawOutputs[neuron] = value;
        m_transferredOutputs[neuron] = sigmoid(value);
    }

    // Returns the (raw) activation of a given neuron
    template<std::size_t Layer>
    double getActivation(std::size_t neuron) const noexcept {
        static_assert(Layer == 0);
        assert(neuron < Neurons);
        return m_rawOutputs[neuron];
    }

private:
    mutable std::array<double, Neurons> m_rawOutputs;
    mutable std::array<double, Neurons> m_transferredOutputs;
};


// Specialization of LayerStack for two or more layers
// This class derives from LayerStack instantiated with all the previous layers
// This specialization of LayerStack includes a Weights member object
// for storing the weights between this and the previous layer,
// which is the base class. Most methods in this class are recursive
// and use the direct base class to continue the recursion.
template<std::size_t Layer0Size, std::size_t Layer1Size, std::size_t... OtherLayerSizes>
class LayerStack<Layer0Size, Layer1Size, OtherLayerSizes...> : public LayerStack<Layer1Size, OtherLayerSizes...> {
public:
    LayerStack() noexcept
        : m_weights{}
        , m_weightGradients{}
        , m_rawOutputs{}
        , m_transferredOutputs{} {
    }

    // Helper type alias for referring to the previous layer
    // (which is the direct base class)
    using Previous = LayerStack<Layer1Size, OtherLayerSizes...>;

    // The number of neurons in this layer
    constexpr static std::size_t Neurons = Layer0Size;

    // The input type for this layer (same as the output of the previous layer)
    using InputType = typename Previous::OutputType;

    // The output type for this layer
    using OutputType = gsl::span<const double, Neurons>;

    // Template function for computing the network's result starting from
    // a given layer.
    template<std::size_t Layer>
    OutputType computePartial() const noexcept {
        // A reference to the outputs of the previous layer
        auto inputs = this->Previous::transferredOutputs();
        if constexpr (Layer > 0){
            // If there are more layers to compute, compute them
            inputs = this->Previous::template computePartial<Layer - 1>();
        }

        // For every output neuron
        for (std::size_t o = 0; o < Neurons; ++o){
            double activationAcc = 0.0;
            // For every input neuron
            for (std::size_t i = 0; i < Previous::Neurons; ++i){
                assert(i < static_cast<std::size_t>(inputs.size()));
                // add the weighted input to the activation
                activationAcc += m_weights(i, o) * inputs[i];
            }
            // Add the bias term
            activationAcc += m_weights(Previous::Neurons, o) * 1.0;

            assert(o < m_rawOutputs.size());
            assert(o < m_transferredOutputs.size());

            // update the raw and transferred output
            m_rawOutputs[o] = activationAcc;
            m_transferredOutputs[o] = sigmoid(activationAcc);
        }

        // Return a reference to the transferred outputs
        return m_transferredOutputs;
    }

    // Perform back-propagation and accumulate the derivative (with respect to a single training example)
    // To calculate the derivative for the sum of the loss over many training examples,
    // simply call scaleGradients(0.0) and then call this function for each example.
    // - outputDerivatives is the derivative of the cost w.r.t. each output neuron (including sigmoid)
    void backPropagateAdd(OutputType outputDerivatives){
        // Vector to hold derivates of input neurons
        auto inputDerivatives = std::vector<double>(Previous::Neurons, 0.0);

        // Raw outputs of the previous layer
        const auto rawInputs = this->Previous::rawOutputs();

        // Transferred (passed through activation function) outputs of the previous layer
        const auto transferredInputs = this->Previous::transferredOutputs();

        // For every output neuron
        for (std::size_t o = 0; o < Neurons; ++o){
            // For every input neuron
            for (std::size_t i = 0; i < Previous::Neurons; ++i){
                // Add a weight term to the input derivative
                inputDerivatives[i] += m_weights(i, o) * outputDerivatives[o];
                // Accumulate the gradient for this weight
                m_weightGradients(i, o) += outputDerivatives[o] * transferredInputs[i];
            }
            // Account for bias derivative
            m_weightGradients(Previous::Neurons, o) += outputDerivatives[o] * 1.0;
        }
        // Multiply each input derivate by sigmoid derivative
        for (std::size_t i = 0; i < Previous::Neurons; ++i){
            inputDerivatives[i] *= sigmoidDerivative(rawInputs[i]);
        }
        
        // Back-propagate through previous layers (if any)
        if constexpr (sizeof...(OtherLayerSizes) > 0){
            this->Previous::backPropagateAdd(inputDerivatives);
        }
    }

    // Multiply every the gradient of every weight in the stack by a scalar value
    // Call scaleGradients(0.0) to reset the gradient
    // Call scaleGradients with a value between 0.0 and 1.0 between steps and
    // derivative calculations to enable momentum
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

    // Adjust each weight in the stack proportional to its accumulated
    // derivative (see backPropagateAdd) and proportional to the given step size
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
    
    // Returns the weight at a given layer, connecting a given input neuron
    // to a given output neuron
    // - Layer is the index of the layer (starting at 0)
    // - input is the index of the input neuron (output neuron of the previous layer)
    // - output is the index of the output neuron (neuron in the given layer)
    template<std::size_t Layer>
    double getWeight(std::size_t input, std::size_t output) const noexcept {
        if constexpr (Layer == 0){
            return m_weights(input, output);
        } else {
            return this->Previous::template getWeight<Layer - 1>(intput, output);
        }
    }
    
    // Modifies the weight at a given layer, connecting a given input neuron
    // to a given output neuron
    // - Layer is the index of the layer (starting witht the output layer at 0)
    // - input is the index of the input neuron (output neuron of the previous layer)
    // - output is the index of the output neuron (neuron in the given layer)
    template<std::size_t Layer>
    void setWeight(std::size_t input, std::size_t output, double value) const noexcept {
        if constexpr (Layer == 0){
            assert(input < Previous::Neurons);
            assert(output < Neurons);
            m_weights(input, output) = value;
        } else {
            return this->Previous::template setWeight<Layer - 1>(intput, output, value);
        }
    }

    // Returns the activation of a given neuron in a given layer
    // NOTE: this is the raw activation, which is before the sigmoid function is applied
    // - Layer is the index of the layer (starting with the output layer at 0)
    // - neuron is the index of a neuron in that layer
    template<std::size_t Layer>
    double getActivation(std::size_t neuron) const noexcept {
        if constexpr (Layer == 0){
            assert(neuron < Neurons);
            return m_rawOutputs[neuron];
        } else {
            return this->Previous::template getActivation<Layer - 1>(neuron);
        }
    }

    // Modifies the activation of a given neuron in a given layer
    // NOTE: this is the raw activation, which is before the sigmoid function is applied
    // - Layer is the index of the layer (starting with the output layer at 0)
    // - neuron is the index of a neuron in that layer
    template<std::size_t Layer>
    void setActivation(std::size_t neuron, double value) noexcept {
        if constexpr (Layer == 0){
            assert(neuron < Neurons);
            m_rawOutputs[neuron] = value;
            m_transferredOutputs[neuron] = sigmoid(value);
        } else {
            this->Previous::template setActivation<Layer - 1>(neuron, value);
        }
    }

    // Randomizes the weights of all layers in the stack
    void randomizeWeights() noexcept {
        m_weights.randomize();
        if constexpr (sizeof...(OtherLayerSizes) > 0){
            this->Previous::randomizeWeights();
        }
    }

    // Returns a reference to the raw (before sigmoid function) outputs of each neuron
    OutputType rawOutputs() const noexcept {
        return m_rawOutputs;
    }
    
    // Returns a reference to the transferred (after sigmoid function) outputs of each neuron
    OutputType transferredOutputs() const noexcept {
        return m_transferredOutputs;
    }

    // Writes the sizes of this and all previous layers to an output stream
    void writeLayerSizes(std::ofstream& ofs) const {
        detail::write<std::size_t>(ofs, Neurons);
        if constexpr (sizeof...(OtherLayerSizes) > 0){
            this->Previous::writeLayerSizes(ofs);
        } else {
            detail::write<std::size_t>(ofs, Previous::Neurons);
        }
    }

    // Writes the weights of this and all previous layers to an output stream
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

    // Reads in the sizes for this and all previous layers and asserts
    // that they match (the network size is fixed at compile time)
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

    // Reads in the weights of this and all previous layers from an input stream
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

    // Returns true if all weight values in this layer stack exactly
    // match those in the other layer stack
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

// NeuralNetwork is a an implementation of a fully-connected neural network
// that uses the sigmoid activation function.
// This is a template class, and the number and sizes of layers can be chosen
// arbitrarily using template parameters. Each integer in the list of template
// arguments defines a new layer with that many neurons.
// NOTE: the neurons per layer are specified with the output neurons first
// and the input neurons last. This is contrary to the common left-to-right
// presentation of neural networks, but is a consequence of the way templates
// are being used here (for now).
// For example:
// - NeuralNetwork<1, 4, 4, 4, 16> is a network with 1 output neuron, 16 input
//   neurons, and 3 hidden layers containing 4 neurons each.
template<std::size_t Layer0Size, std::size_t... OtherLayerSizes>
class NeuralNetwork {
public:
    static_assert(sizeof...(OtherLayerSizes) > 0, "You must create a NeuralNetwork with at least two layers");

    // Type alias for the layer stack used internally
    using LayersType = LayerStack<Layer0Size, OtherLayerSizes...>;

    // The number of layers in the network, which includes the input,
    // output, and any hidden layers.
    static constexpr std::size_t NumLayers = 1ul + sizeof...(OtherLayerSizes);

    // The number of neurons in the input layer
    static constexpr std::size_t InputNeurons = LayersType::RootNeurons;

    // The number of neurons in the output layer
    static constexpr std::size_t OutputNeurons = LayersType::Neurons;

    // The input type accepted by the network
    using InputType = typename LayersType::RootInputType;

    // The output type accepted by the network
    using OutputType = typename LayersType::OutputType;

    // Computes the network's output for a given input
    OutputType compute(InputType inputs) noexcept {
        for (std::size_t i = 0; i < InputNeurons; ++i){
            this->template setActivation<NumLayers - 1>(i, inputs[i]);
        }
        return layers.template computePartial<NumLayers - 1>();
    }

    // Computes the network's output, not from any input,
    // but starting from a specific layer.
    // For example, computePartial<1>() computes the output using
    // just the activations of the second-to-last layer.
    // computePartial<numLayers - 1> computes the output using
    // the entire network, starting from the input layer's activations.
    template<std::size_t Layer>
    OutputType computePartial() const noexcept {
        static_assert(Layer < NumLayers);
        static_assert(Layer > 0);
        return layers.template computePartial<Layer - 1>();
    }

    // Trains the network for a single step and returns the average loss.
    // - examples is a list of input-output pairs which are the training data
    // - stepSize is the factor by which the network's weights are adjusted
    // - momentumRatio is the ratio of the previous step which is added onto
    //   this step. Omit this parameter if you do not want to use momentum
    // NOTE: examples can be a list containing one item, it can be a random
    // subset of the training data, or it can be the entire training data.
    // Its use is up to you.
    double takeStep(gsl::span<std::pair<InputType, OutputType>> examples, double stepSize, double momentumRatio = 0.0) noexcept {
        double lossAcc = 0.0;
        // scale gradients down before accumulating (or set to zero)
        layers.scaleGradients(momentumRatio);

        // for every training example and its label
        for (const auto& [input, expectedOutput] : examples){
            // Compute the network's prediction
            compute(input);

            // Get the network's output
            const auto transferredOutputs = layers.transferredOutputs();
            const auto rawOutputs = layers.rawOutputs();
            
            // Calculate the loss and accumulate
            lossAcc += detail::loss(transferredOutputs, expectedOutput);
            
            // array to hold derivates of output neurons as needed for backpropagation
            auto outputDerivatives = std::array<double, OutputNeurons>{};

            // For every output neuron
            for (std::size_t o = 0; o < OutputNeurons; ++o){
                // Its derivative is the derivative of the sigmoid function on the actual output,
                // times the expected output minus the actual output
                outputDerivatives[o] = sigmoidDerivative(rawOutputs[o]) * (expectedOutput[o] - transferredOutputs[o]);
            }

            // Calculate and accumulate the derivative through the network
            layers.backPropagateAdd(outputDerivatives);
        }
        // Take a step
        // NOTE: the step size is divided by the number of examples here because
        // every call to backPropagateAdd() adds the derivative with respect to
        // a single training example. This division effectively gives the derivative
        // with respect to the average loss
        layers.adjustWeights(stepSize / static_cast<double>(examples.size()));

        // Return the average loss
        return lossAcc / static_cast<double>(examples.size());
    }
    
    // Randomizes the weights in the entire network
    void randomizeWeights() noexcept {
        layers.randomizeWeights();
    }
    
    // Returns the weight for a given layer between a neuron in that layer
    // neuron and a input from the previous layer
    // - Layer is the index of the output layer, starting with the
    //   network's final layer at 0
    // - input is the index of a neuron in the layer before that
    // - output is the index of a neuron in the given layer
    template<std::size_t Layer>
    double getWeight(std::size_t input, std::size_t output) const noexcept {
        static_assert(Layer < NumLayers);
        return layers.template weights<Layer>()(input, output);
    }
    
    // Modifies the weight for a given layer between a neuron in that layer
    // neuron and a input from the previous layer
    // - Layer is the index of the output layer, starting with the
    //   network's final output layer at index 0
    // - input is the index of a neuron in the layer before that
    // - output is the index of a neuron in the given layer
    template<std::size_t Layer>
    void setWeight(std::size_t input, std::size_t output, double value) noexcept {
        static_assert(Layer < NumLayers);
        return layers.template weights<Layer>()(input, output);
    }
    
    // Returns the activation of a single neuron.
    // Call compute, computePartial, or setActivation to see this value change.
    // - Layer is the index of the layer, starting with the network's final
    //   final output layer at index 0.
    // - neuron is the index of a neuron in that layer
    template<std::size_t Layer>
    double getActivation(std::size_t neuron) const noexcept {
        static_assert(Layer < NumLayers);
        return layers.template getActivation<Layer>(neuron);
    }
    
    // Modifies the activation of a single neuron.
    // Calling computePartial for the next (Layer - 1) output layer will use this value.
    // Calling compute will overwrite this value completely.
    // - Layer is the index of the layer, starting with the network's final
    //   final output layer at index 0.
    // - neuron is the index of a neuron in that layer
    template<std::size_t Layer>
    void setActivation(std::size_t neuron, double value) noexcept {
        static_assert(Layer < NumLayers);
        layers.template setActivation<Layer>(neuron, value);
    }

    // Writes the network's weights to a file
    // NOTE: this file will be specific to this network's structure and
    // cannot be used on any other network structure. The number and sizes
    // of layers must match.
    void saveWeights(std::string filepath) const {
        std::ofstream ofs{filepath, std::ios::out | std::ios::binary};
        if (!ofs){
            throw std::runtime_error("Could not open file");
        }

        detail::write<std::size_t>(ofs, NumLayers);

        layers.writeLayerSizes(ofs);
        layers.writeWeights(ofs);
    }
    
    // Loads the network's weights from a file
    // NOTE: if the number and sizes of layers in the file do not match
    // this network's structure, an exception is thrown.
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

    // Overload of operator == to test whether the weights in
    // two networks are equal
    bool operator==(const NeuralNetwork& other) const noexcept {
        return layers.equals(other.layers);
    }
    
    // Overload of operator != to test whether the weights in
    // two networks are not equal
    bool operator!=(const NeuralNetwork& other) const noexcept {
        return !this->operator==(other);
    }

private:
    LayersType layers;
};

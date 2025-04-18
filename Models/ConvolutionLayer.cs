using ComputeShaders;
using ComputeShaders.Windows;
using Neuran.Activations;
using Neuran.GradientDescent;
using Neuran.Utilities;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Models
{
    /// <summary>
    /// The class for convolution layers in CNNs.
    /// </summary>
    public class ConvolutionLayer : IGradientDescent
    {
        /// <summary>
        /// The side length of the kernel.
        /// </summary>
        public int FilterSize { get; private set; }
        /// <summary>
        /// The additional length to add (virtually) to the input tensor.
        /// </summary>
        public int Padding { get; private set; }
        /// <summary>
        /// The amount of pixels the filter move every iteration.
        /// </summary>
        public int Stride { get; private set; }

        IGradientDescent layer;

        /// <summary>
        /// Creates a new instance.
        /// </summary>
        /// <param name="inputWidth"></param>
        /// <param name="inputHeight"></param>
        /// <param name="inputDepth"></param>
        /// <param name="filterSize"></param>
        /// <param name="filtersNum"></param>
        /// <param name="stride"></param>
        /// <param name="padding"></param>
        /// <param name="activation"></param>
        /// <param name="random"></param>
        /// <param name="device"></param>
        public ConvolutionLayer(int inputWidth, int inputHeight, int inputDepth, int filterSize, int filtersNum, int stride, bool padding, IActivation activation, Random random, CSDevice device = null)
        {
            FilterSize = filterSize;
            Padding = padding ? FilterSize - 1 : 0;
            Stride = stride;

            if (device == null)
            {
                layer = new ConvCPU(inputWidth, inputHeight, inputDepth, filterSize, filtersNum, stride, padding, activation, random);
            }
            else
            {
                layer = new ConvGPU(inputWidth, inputHeight, inputDepth, filterSize, filtersNum, stride, padding, activation, random, device);
            }
        }

        /// <inheritdoc/>
        public Tensor[] Output { get => layer.Output; }

        /// <inheritdoc/>
        public Tensor[] Input { get => layer.Input; }

        /// <inheritdoc/>
        public Tensor[] PreLayerDer { get => layer.PreLayerDer; }

        /// <inheritdoc/>
        public IGradientDescent ConnectedFrom { get => layer.ConnectedFrom; }

        /// <inheritdoc/>
        public void PrepareGD(int maxTruncatedLength)
        {
            layer.PrepareGD(maxTruncatedLength);
        }

        /// <inheritdoc/>
        public void EndGD()
        {
            layer.EndGD();
        }

        /// <inheritdoc/>
        public void AddParameters(List<Tensor> parameters)
        {
            layer.AddParameters(parameters);
        }

        /// <inheritdoc/>
        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            layer.Backpropagate(lossDer, pastTime);
        }

        /// <inheritdoc/>
        public void Connect(IGradientDescent model)
        {
            layer.Connect(model);
        }

        /// <inheritdoc/>
        public void ResetGradients()
        {
            layer.ResetGradients();
        }

        /// <inheritdoc/>
        public Tensor[] Run(Tensor[] input)
        {
            return layer.Run(input);
        }

        /// <inheritdoc/>
        public void Reset()
        {
            layer.Reset();
        }

        /// <inheritdoc/>
        public void SaveRandomState()
        {
            layer.SaveRandomState();
        }

        /// <inheritdoc/>
        public void LoadRandomState()
        {
            layer.LoadRandomState();
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            layer.Dispose();
        }
    }

    public class ConvCPU : IGradientDescent
    {
        public Tensor[] PreLayerDer { get => preLayerDer; }

        public IGradientDescent ConnectedFrom { get; private set; }

        public Tensor[] Output { get; private set; }

        public Tensor[] Input { get; private set; }

        public int InputWidth { get => Input[0].Dimensions[0]; }
        public int InputHeight { get => Input[0].Dimensions[1]; }
        public int InputDepth { get => Input[0].Dimensions[2]; }
        public int OutputWidth { get => Output[0].Dimensions[0]; }
        public int OutputHeight { get => Output[0].Dimensions[1]; }
        public int OutputDepth { get => Output[0].Dimensions[2]; }

        /// <summary>
        /// The side length of a filter.
        /// </summary>
        public int FilterSize { get; }
        /// <summary>
        /// Tne number of filters in the layer.
        /// </summary>
        public int Filters { get; }
        /// <summary>
        /// The layer's stride.
        /// </summary>
        public int Stride { get; }
        /// <summary>
        /// The amount of padding used on the layer.
        /// </summary>
        public int Padding { get; }

        public IActivation ActivationFunction { get; private set; }

        private Tensor beforeActivation;
        private Tensor filterAndBias;

        private Tensor[] preLayerDer;
        private Tensor beforeActivationDer;
        private Tensor[] inputs;
        private Tensor[] outputs;

        public ConvCPU(int inputWidth, int inputHeight, int inputDepth, int filterSize, int filtersNum, int stride, bool padding, IActivation activation, Random random)
        {
            Padding = padding ? filterSize - 1 : 0;
            Filters = filtersNum;
            FilterSize = filterSize;
            Stride = stride;

            int outputWidth = (inputWidth - filterSize + Padding * 2) / Stride + 1;
            int outputHeight = (inputHeight - filterSize + Padding * 2) / Stride + 1;
            int outputDepth = filtersNum;

            Input = new Tensor[] { new Tensor(null, inputWidth, inputHeight, inputDepth) };
            Output = new Tensor[] { new Tensor(null, outputWidth, outputHeight, outputDepth) };
            beforeActivation = new Tensor(null, outputWidth, outputHeight, outputDepth);

            ActivationFunction = activation;

            filterAndBias = new Tensor(null, filterSize * filterSize * filtersNum * inputDepth + 1);

            //https://www.geeksforgeeks.org/weight-initialization-techniques-for-deep-neural-networks/
            // "He Normal Initialization"
            float std = (float)Math.Sqrt(2.0 / (filterSize * filterSize * inputDepth)); //In Convolution layers, the number of input channels (fan_in) to a single neuron (output pixel) (in other words, the number of input pixels that affect a single output pixel) = input depth * filter size * filter size

            for (int i = 0; i < filterAndBias.TensorLength; i++)
            {
                filterAndBias[i] = (float)UtilitiesFuncs.RandomGaussain(0, std, random);
            }
        }


        public void AddParameters(List<Tensor> parameters)
        {
            parameters.Add(filterAndBias);
        }
        private float SingleFilterBack(int filterX, int filterY, int filterZ, int filterW, int pastTime)
        {
            int xmax = Math.Min(OutputWidth - 1, (int)Div(InputWidth - 1 + Padding - filterX, Stride));
            int ymax = Math.Min(OutputHeight - 1, (int)Div(InputHeight - 1 + Padding - filterY, Stride));
            int xmin = (int)Math.Max(Math.Ceiling(Div(Padding - filterX, Stride)), 0);
            int ymin = (int)Math.Max(Math.Ceiling(Div(Padding - filterY, Stride)), 0);

            float sum = 0;

            for (int j = ymin; j <= ymax; j++)
            {
                for (int i = xmin; i <= xmax; i++)
                {
                    sum += beforeActivationDer[i, j, filterW] * inputs[pastTime][Stride * i + filterX - Padding, Stride * j + filterY - Padding, filterZ];
                }
            }

            return sum;
        }
        private float Div(int num, int den)
        {
            if (num % den == 0)
                return num / den;
            else
                return num / (float)den;
        }
        private float SingleBack(int inputX, int inputY, int inputZ)
        {
            float sum = 0;

            //int ymax = Math.Min((inputY + Padding) / Stride, Output.Dimensions[1] - 1);
            //int xmax = Math.Min((inputX + Padding) / Stride, Output.Dimensions[0] - 1);
            //int ymin = (int)Math.Max(Math.Ceiling((inputY + Padding + 1 - FilterSize) / (float)Stride), 0);
            //int xmin = (int)Math.Max(Math.Ceiling((inputX + Padding + 1 - FilterSize) / (float)Stride), 0);

            int ymax = Math.Min((int)Div(inputY + Padding, Stride), OutputHeight - 1);
            int xmax = Math.Min((int)Div(inputX + Padding, Stride), OutputWidth - 1);
            int ymin = (int)Math.Max(Math.Ceiling(Div(inputY + Padding + 1 - FilterSize, Stride)), 0);
            int xmin = (int)Math.Max(Math.Ceiling(Div(inputX + Padding + 1 - FilterSize, Stride)), 0);

            for (int h = 0; h < OutputDepth; h++)
            {
                for (int j = ymin; j <= ymax; j++)
                {
                    for (int i = xmin; i <= xmax; i++)
                    {
                        int filterX = inputX - i * Stride + Padding;
                        int filterY = inputY - j * Stride + Padding;
                        int filterZ = inputZ;
                        int filterW = h;
                        int filterIndex = filterX + filterY * FilterSize + filterZ * FilterSize * FilterSize + filterW * InputDepth * FilterSize * FilterSize;

                        sum += beforeActivationDer[i, j, h] * filterAndBias[filterIndex];

                    }
                }
            }

            return sum;
        }
        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            lossDer[0].CopyTo(beforeActivationDer);

            ActivationFunction.GetDerivative(beforeActivation, outputs[pastTime], beforeActivationDer);

            for (int z = 0; z < InputDepth; z++)
            {
                for (int y = 0; y < InputHeight; y++)
                {
                    for (int x = 0; x < InputWidth; x++)
                    {
                        PreLayerDer[0][x, y, z] = SingleBack(x, y, z);
                    }
                }
            }

            float sum = 0;
            for (int i = 0; i < beforeActivationDer.TensorLength; i++)
            {
                sum += beforeActivationDer[i];
            }
            filterAndBias.Gradient[filterAndBias.TensorLength - 1] = sum;

            for (int w = 0; w < OutputDepth; w++)
            {
                for (int z = 0; z < InputDepth; z++)
                {
                    for (int y = 0; y < FilterSize; y++)
                    {
                        for (int x = 0; x < FilterSize; x++)
                        {
                            int index = x + y * FilterSize + z * FilterSize * FilterSize + w * FilterSize * FilterSize * InputDepth;

                            float der = SingleFilterBack(x, y, z, w, pastTime);

                            filterAndBias.Gradient[index] += der;
                        }
                    }
                }
            }

            ConnectedFrom?.Backpropagate(preLayerDer, pastTime);
        }

        public void Connect(IGradientDescent model)
        {
            ConnectedFrom = model;
        }

        public void Dispose()
        {

        }

        public void EndGD()
        {
            preLayerDer = null;
            beforeActivationDer = null;
            inputs = null;
            outputs = null;
        }

        public void LoadRandomState()
        {

        }

        public void PrepareGD(int maxTruncatedLength)
        {
            preLayerDer = new Tensor[] { Input[0].EmptyClone() };
            beforeActivationDer = Output[0].EmptyClone();

            inputs = new Tensor[maxTruncatedLength];
            outputs = new Tensor[maxTruncatedLength];

            for (int i = 0; i < maxTruncatedLength; i++)
            {
                inputs[i] = Input[0].EmptyClone();
                outputs[i] = Output[0].EmptyClone();
            }
        }

        public void Reset()
        {

        }

        public void ResetGradients()
        {

        }

        private float Convolute(int outputX, int outputY, int outputZ)
        {
            float sum = 0;

            int ymin = Math.Max(Padding - Stride * outputY, 0);
            int xmin = Math.Max(Padding - Stride * outputX, 0);
            int ymax = Math.Min(InputHeight - 1 + Padding - Stride * outputY, FilterSize - 1);
            int xmax = Math.Min(InputWidth - 1 + Padding - Stride * outputX, FilterSize - 1);

            for (int h = 0; h < InputDepth; h++)
            {
                for (int j = ymin; j <= ymax; j++)
                {
                    for (int i = xmin; i <= xmax; i++)
                    {
                        int filterIndex = outputZ * InputDepth * FilterSize * FilterSize + h * FilterSize * FilterSize + j * FilterSize + i;
                        sum += Input[0][Stride * outputX + i - Padding, Stride * outputY + j - Padding, h] * filterAndBias[filterIndex];
                    }
                }
            }

            return sum + filterAndBias[filterAndBias.TensorLength - 1];
        }
        public Tensor[] Run(Tensor[] input)
        {
            input[0].CopyTo(Input[0]);

            for (int z = 0; z < OutputDepth; z++)
            {
                for (int y = 0; y < OutputHeight; y++)
                {
                    for (int x = 0; x < OutputWidth; x++)
                    {
                        float ba = Convolute(x, y, z);

                        beforeActivation[x, y, z] = ba;

                        if (ActivationFunction.ElementWise)
                            Output[0][x, y, z] = ActivationFunction.ActivateElementWise(ba);
                    }
                }
            }

            if (!ActivationFunction.ElementWise)
                ActivationFunction.Activate(beforeActivation, Output[0]);

            if (inputs != null)
            {
                for (int i = inputs.Length - 2; i >= 0; i--)
                {
                    inputs[i].CopyTo(inputs[i + 1]);
                    outputs[i].CopyTo(outputs[i + 1]);
                }

                Input[0].CopyTo(inputs[0]);
                Output[0].CopyTo(outputs[0]);
            }

            return Output;
        }

        public void SaveRandomState()
        {

        }
    }

    public class ConvGPU : IGradientDescent
    {
        public Tensor[] PreLayerDer { get => preLayerDer; }

        public IGradientDescent ConnectedFrom { get; private set; }

        public Tensor[] Output { get; private set; }

        public Tensor[] Input { get; private set; }

        /// <summary>
        /// The side length of a filter.
        /// </summary>
        public int FilterSize { get; }
        /// <summary>
        /// Tne number of filters in the layer.
        /// </summary>
        public int Filters { get; }
        /// <summary>
        /// The layer's stride.
        /// </summary>
        public int Stride { get; }
        /// <summary>
        /// The amount of padding used on the layer.
        /// </summary>
        public int Padding { get; }

        public IActivation ActivationFunction { get; private set; }

        private Tensor beforeActivation;
        private Tensor filterAndBias;

        private Tensor[] preLayerDer;
        private Tensor beforeActivationDer;
        private Tensor[] inputs;
        private Tensor[] outputs;

        private CSDevice device;
        private ComputeShader conv;
        private CSCBuffer<Int12> info;

        private ConvDerCalculator derCalculator;
        
        public int InputWidth { get => Input[0].Dimensions[0]; }
        public int InputHeight { get => Input[0].Dimensions[1]; }
        public int InputDepth { get => Input[0].Dimensions[2]; }
        public int OutputWidth { get => Output[0].Dimensions[0]; }
        public int OutputHeight { get => Output[0].Dimensions[1]; }
        public int OutputDepth { get => Output[0].Dimensions[2]; }

        public ConvGPU(int inputWidth, int inputHeight, int inputDepth, int filterSize, int filtersNum, int stride, bool padding, IActivation activation, Random random, CSDevice device)
        {
            this.device = device;
            conv = device.CreateComputeShader(ConvolutionShaders.Convolution);

            Padding = padding ? filterSize - 1 : 0;
            Filters = filtersNum;
            FilterSize = filterSize;
            Stride = stride;

            int outputWidth = (inputWidth - filterSize + Padding * 2) / Stride + 1;
            int outputHeight = (inputHeight - filterSize + Padding * 2) / Stride + 1;
            int outputDepth = filtersNum;

            Input = new Tensor[] { new Tensor(device, inputWidth, inputHeight, inputDepth) };
            Output = new Tensor[] { new Tensor(device, outputWidth, outputHeight, outputDepth) };
            beforeActivation = new Tensor(device, outputWidth, outputHeight, outputDepth);

            ActivationFunction = activation;

            float[] f = new float[filterSize * filterSize * filtersNum * inputDepth + 1];

            //https://www.geeksforgeeks.org/weight-initialization-techniques-for-deep-neural-networks/
            // "He Normal Initialization"
            float std = (float)Math.Sqrt(2.0 / (filterSize * filterSize * inputDepth)); //In Convolution layers, the number of input channels (fan_in) to a single neuron (output pixel) (in other words, the number of input pixels that affect a single output pixel) = input depth * filter size * filter size

            for (int i = 0; i < f.Length; i++)
            {
                f[i] = (float)UtilitiesFuncs.RandomGaussain(0, std, random);
            }

            filterAndBias = new Tensor(device, f);

            info = device.CreateBuffer(new Int12()
            {
                int1 = inputWidth,
                int2 = inputHeight,
                int3 = inputDepth,
                int4 = outputWidth,
                int5 = outputHeight,
                int6 = outputDepth,
                int7 = filterSize,
                int8 = stride,
                int9 = Padding,
                int10 = filterAndBias.TensorLength,
            }, Int12.Size);
        }

        public void PrepareGD(int maxTruncatedLength)
        {
            preLayerDer = new Tensor[] { Input[0].EmptyClone() };
            beforeActivationDer = Output[0].EmptyClone();

            inputs = new Tensor[maxTruncatedLength];
            outputs = new Tensor[maxTruncatedLength];

            for (int i = 0; i < maxTruncatedLength; i++)
            {
                inputs[i] = Input[0].EmptyClone();
                outputs[i] = Output[0].EmptyClone();
            }

            derCalculator = new ConvDerCalculator(Output[0], Input[0], filterAndBias, FilterSize, Stride, Padding);
        }

        public void EndGD()
        {
            preLayerDer[0].Dispose();
            beforeActivationDer.Dispose();

            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i].Dispose();
                outputs[i].Dispose();
            }

            inputs = null;
            outputs = null;

            derCalculator.Dispose();
        }

        public void AddParameters(List<Tensor> parameters)
        {
            parameters.Add(filterAndBias);
        }

        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            lossDer[0].CopyTo(beforeActivationDer);

            ActivationFunction.GetDerivative(beforeActivation, outputs[pastTime], beforeActivationDer);

            derCalculator.CalcPreLayer(beforeActivationDer, filterAndBias, preLayerDer[0]);
            derCalculator.CalcFiltersDer(inputs[pastTime], beforeActivationDer, filterAndBias);

            ConnectedFrom?.Backpropagate(preLayerDer, pastTime);
        }

        public void Connect(IGradientDescent model)
        {
            ConnectedFrom = model;
        }

        public void ResetGradients()
        {

        }

        public Tensor[] Run(Tensor[] input)
        {
            input[0].CopyTo(Input[0]);

            device.SetComputeShader(conv);

            Input[0].SetUAV(0);
            beforeActivation.SetUAV(1);
            filterAndBias.SetUAV(2);

            device.SetBuffer(info, 0);

            device.Dispatch((int)Math.Ceiling(OutputWidth / 8f), (int)Math.Ceiling(OutputHeight / 8f), OutputDepth);

            ActivationFunction.Activate(beforeActivation, Output[0]);

            if (inputs != null)
            {
                for (int i = inputs.Length - 2; i >= 0; i--)
                {
                    inputs[i].CopyTo(inputs[i + 1]);
                    outputs[i].CopyTo(outputs[i + 1]);
                }

                Input[0].CopyTo(inputs[0]);
                Output[0].CopyTo(outputs[0]);
            }

            return Output;
        }

        public void Reset()
        {
            Input[0].Zero();
            Output[0].Zero();
        }

        public void SaveRandomState()
        {

        }

        public void LoadRandomState()
        {

        }

        public void Dispose()
        {
            conv.Dispose();
            Input[0].Dispose();
            Output[0].Dispose();
            beforeActivation.Dispose();
            filterAndBias.Dispose();

            if (inputs != null)
                EndGD();
        }
    }
}

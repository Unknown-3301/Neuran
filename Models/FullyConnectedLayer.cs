using ComputeShaders;
using Neuran.Activations;
using Neuran.GradientDescent;
using Neuran.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Models
{
    /// <summary>
    /// A dense neural network layer.
    /// </summary>
    public class FullyConnectedLayer : IGradientDescent
    {
        IGradientDescent layer;

        /// <inheritdoc/>
        public Tensor[] PreLayerDer { get => layer.PreLayerDer; }
        /// <inheritdoc/>
        public IGradientDescent ConnectedFrom { get => layer.ConnectedFrom; }

        /// <summary>
        /// Creates a new instance.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        /// <param name="activationFunction"></param>
        /// <param name="random"></param>
        /// <param name="device"></param>
        public FullyConnectedLayer(int input, int output, IActivation activationFunction, Random random, CSDevice device = null)
        {
            if (device == null)
            {
                layer = new CPUFCL(input, output, activationFunction, random);
            }
            else
            {
                layer = new GPUFCL(input, output, activationFunction, random, device);
            }
        }

        /// <inheritdoc/>
        public Tensor[] Output { get => layer.Output; }

        /// <inheritdoc/>
        public Tensor[] Input { get => layer.Input; }

        /// <inheritdoc/>
        public void EndGD() => layer.EndGD();


        /// <inheritdoc/>
        public void PrepareGD(int maxTruncatedLength) => layer.PrepareGD(maxTruncatedLength);

        /// <inheritdoc/>
        public void Reset() => layer.Reset();

        /// <inheritdoc/>
        public void ResetGradients() => layer.ResetGradients();

        /// <inheritdoc/>
        public Tensor[] Run(Tensor[] input) => layer.Run(input);

        /// <inheritdoc/>
        public void Dispose() => layer.Dispose();

        /// <inheritdoc/>
        public void AddParameters(List<Tensor> parameters) => layer.AddParameters(parameters);

        /// <inheritdoc/>
        public void Backpropagate(Tensor[] lossDer, int pastTime) => layer.Backpropagate(lossDer, pastTime);

        /// <inheritdoc/>
        public void Connect(IGradientDescent model) => layer.Connect(model);

        /// <inheritdoc/>
        public void SaveRandomState() { }

        /// <inheritdoc/>
        public void LoadRandomState() { }
    }
    // cpu fully connected layer
    internal class CPUFCL : IGradientDescent
    {
        private Tensor[] input;
        private Tensor weight;
        private Tensor bias;
        private Tensor beforeActivation;
        private Tensor[] output;

        private bool added;
        private Tensor[] preLayerDer;
        private Tensor beforeActivationDer;

        private Tensor[] inputs;
        private Tensor[] outputs;

        IActivation function;

        public Tensor[] PreLayerDer { get => preLayerDer; }
        public Tensor[] Output { get => output; }

        public Tensor[] Input { get => input; }

        public bool IsRecurrent { get => false; }

        public IGradientDescent ConnectedFrom { get; private set; }

        public CPUFCL(int input, int output, IActivation function, Random random)
        {
            this.function = function;
            weight = new Tensor(null, input, output);
            bias = new Tensor(null, output);
            beforeActivation = new Tensor(null, output);
            this.output = new Tensor[] { new Tensor(null, output) };
            this.input = new Tensor[] { new Tensor(null, input) };

            for (int y = 0; y < output; y++)
            {
                for (int x = 0; x < input; x++)
                {
                    weight[x, y] = (float)UtilitiesFuncs.RandomGaussain(0, 1f / input, random); //what about 1f / Math.Sqrt(input)??)
                }

                bias[y] = (float)UtilitiesFuncs.RandomGaussain(0, 1f / input, random);
            }
        }

        public Tensor[] Run(Tensor[] input)
        {
            input[0].CopyTo(this.input[0]);

            for (int y = 0; y < weight.Dimensions[1]; y++)
            {
                float sum = 0;

                for (int x = 0; x < weight.Dimensions[0]; x++)
                {
                    sum += weight[x, y] * this.input[0][x];
                }

                beforeActivation[y] = sum + bias[y];

                if (function.ElementWise)
                    output[0][y] = function.ActivateElementWise(beforeActivation[y]);
            }

            if (!function.ElementWise)
                function.Activate(beforeActivation, output[0]);

            if (inputs != null)
            {
                for (int i = inputs.Length - 2; i >= 0; i--)
                {
                    inputs[i].CopyTo(inputs[i + 1]);
                    outputs[i].CopyTo(outputs[i + 1]);
                }

                this.input[0].CopyTo(inputs[0]);
                output[0].CopyTo(outputs[0]);
            }

            return output;
        }

        public void PrepareGD(int maxTruncatedLength)
        {
            weight.CreateGradient();
            bias.CreateGradient();
            beforeActivationDer = new Tensor(null, output[0].Dimensions);
            preLayerDer = new Tensor[] { new Tensor(null, input[0].Dimensions) };

            outputs = new Tensor[maxTruncatedLength];
            inputs = new Tensor[maxTruncatedLength];

            for (int i = 0; i < maxTruncatedLength; i++)
            {
                inputs[i] = new Tensor(null, input[0].Dimensions);
                outputs[i] = new Tensor(null, output[0].Dimensions);
            }
        }

        public void EndGD()
        {
            weight.DisposeGradient();
            bias.DisposeGradient();
            beforeActivationDer = null;
            preLayerDer = null;
            outputs = null;
            inputs = null;
        }

        public void AddParameters(List<Tensor> parameters)
        {
            if (added)
                return;

            parameters.Add(weight);
            parameters.Add(bias);
        }
        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            lossDer[0].CopyTo(beforeActivationDer);

            if (!function.ElementWise)
            {
                function.GetDerivative(beforeActivation, outputs[pastTime], beforeActivationDer);
            }

            for (int i = 0; i < Input[0].TensorLength; i++)
            {
                preLayerDer[0][i] = 0;

                for (int j = 0; j < Output[0].TensorLength; j++)
                {
                    if (i == 0)
                    {
                        if (function.ElementWise)
                            beforeActivationDer[j] *= function.GetDerivativeElementWise(beforeActivation[j], outputs[pastTime][j]);

                        bias.Gradient[j] += beforeActivationDer[j];
                    }

                    preLayerDer[0][i] += weight[j * Input[0].TensorLength + i] * beforeActivationDer[j];

                    weight.Gradient[j * Input[0].TensorLength + i] += inputs[pastTime][i] * beforeActivationDer[j];
                }
            }

            ConnectedFrom?.Backpropagate(preLayerDer, pastTime);
        }

        public void ResetGradients()
        {
            weight.Gradient.Zero();
            bias.Gradient.Zero();
            beforeActivationDer.Zero();
        }
        public void Reset()
        {
            if (inputs == null)
                return;

            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i].Zero();
                outputs[i].Zero();
            }
        }

        public void Connect(IGradientDescent model)
        {
            ConnectedFrom = model;
        }

        /// <inheritdoc/>
        public void Dispose() { }

        /// <inheritdoc/>
        public void SaveRandomState() { }

        /// <inheritdoc/>
        public void LoadRandomState() { }
    }
    internal class GPUFCL : IGradientDescent, IDisposable
    {
        private Tensor[] input;
        private Tensor weight;
        private Tensor bias;
        private Tensor beforeActivation;
        private Tensor[] output;
        IActivation function;

        private CSDevice device;
        private ComputeShader forward;
        private CSCBuffer<Int4> info;

        private Tensor[] preLayerDer;
        private Tensor beforeActivationDer;
        private ComputeShader backward;

        private Tensor[] inputs;
        private Tensor[] outputs;

        public Tensor[] PreLayerDer { get => preLayerDer; }
        public Tensor[] Output { get => output; }

        public Tensor[] Input { get => input; }

        public IGradientDescent ConnectedFrom { get; private set; }

        public GPUFCL(int input, int output, IActivation function, Random random, CSDevice device)
        {
            this.device = device;
            this.function = function;

            float[] weights = new float[input * output];
            float[] bias = new float[output];

            for (int y = 0; y < output; y++)
            {
                for (int x = 0; x < input; x++)
                {
                    weights[x + y * input] = (float)UtilitiesFuncs.RandomGaussain(0, 1f / input, random);
                }

                bias[y] = (float)UtilitiesFuncs.RandomGaussain(0, 1f / input, random); //It appears that just init them as 0 is better? https://cs231n.github.io/neural-networks-2/
            }

            weight = new Tensor(device, weights);
            this.bias = new Tensor(device, bias);
            beforeActivation = new Tensor(device, output);

            info = device.CreateBuffer(new Int4(), Int4.Size);

            this.output = new Tensor[] { new Tensor(device, output) };
            this.input = new Tensor[] { new Tensor(device, input) };

            forward = device.CreateComputeShader(FCLShaders.Forward);
            backward = device.CreateComputeShader(FCLShaders.Backpropgation);
        }

        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            lossDer[0].CopyTo(beforeActivationDer);

            //float[] d1 = lossDer[0].GetData(); //DEBUG
            //float[] d2 = beforeActivationDer.GetData();//DEBUG
            //float[] d3 = outputs[pastTime].GetData();//DEBUG

            function.GetDerivative(beforeActivation, outputs[pastTime], beforeActivationDer);

            //float[] d4 = beforeActivationDer.GetData();//DEBUG

            bias.Gradient.Add(beforeActivationDer);

            inputs[pastTime].SetUAV(0);
            preLayerDer[0].SetUAV(1);
            weight.SetUAV(2);
            weight.Gradient.SetUAV(3);
            beforeActivationDer.SetUAV(4);

            info.UpdateBuffer(new Int4()
            {
                int1 = input[0].TensorLength,
                int2 = output[0].TensorLength,
            });

            device.SetBuffer(info, 0);

            device.SetComputeShader(backward);
            device.Dispatch((int)Math.Ceiling(input[0].TensorLength / 16f), 1, 1);

            ConnectedFrom?.Backpropagate(preLayerDer, pastTime);
        }

        public void EndGD()
        {
            weight.DisposeGradient();
            bias.DisposeGradient();
            beforeActivationDer.Dispose();
            preLayerDer[0].Dispose();

            for (int i = 0; i < outputs.Length; i++)
            {
                outputs[i].Dispose();
                inputs[i].Dispose();
            }
        }

        public void AddParameters(List<Tensor> list)
        {
            list.Add(weight);
            list.Add(bias);
        }


        public void PrepareGD(int maxTruncatedLength)
        {
            weight.CreateGradient();
            bias.CreateGradient();
            beforeActivationDer = new Tensor(device, output[0].Dimensions);
            preLayerDer = new Tensor[] { new Tensor(device, input[0].Dimensions) };

            outputs = new Tensor[maxTruncatedLength];
            inputs = new Tensor[maxTruncatedLength];

            for (int i = 0; i < maxTruncatedLength; i++)
            {
                inputs[i] = new Tensor(device, input[0].Dimensions);
                outputs[i] = new Tensor(device, output[0].Dimensions);
            }
        }

        public void ResetGradients()
        {
            weight.Gradient.Zero();
            bias.Gradient.Zero();
            beforeActivationDer.Zero();
        }
        public void Reset()
        {
            if (inputs == null)
                return;

            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i].Zero();
                outputs[i].Zero();
            }
        }

        public Tensor[] Run(Tensor[] input)
        {
            input[0].CopyTo(this.input[0]);

            this.input[0].SetUAV(0);
            weight.SetUAV(1);
            bias.SetUAV(2);
            beforeActivation.SetUAV(3);

            info.UpdateBuffer(new Int4()
            {
                int1 = this.input[0].TensorLength,
                int2 = output[0].TensorLength,
            });

            device.SetBuffer(info, 0);

            device.SetComputeShader(forward);
            device.Dispatch((int)Math.Ceiling(output[0].TensorLength / 16f), 1, 1);

            function.Activate(beforeActivation, output[0]);

            if (inputs != null)
            {
                for (int i = inputs.Length - 2; i >= 0; i--)
                {
                    inputs[i].CopyTo(inputs[i + 1]);
                    outputs[i].CopyTo(outputs[i + 1]);
                }

                this.input[0].CopyTo(inputs[0]);
                output[0].CopyTo(outputs[0]);
            }

            return output;
        }

        public void Dispose()
        {
            weight.Dispose();
            bias.Dispose();

            if (weight.Gradient != null)
                EndGD();

            forward.Dispose();
            info.Dispose();
            input[0].Dispose();
            output[0].Dispose();
            beforeActivation.Dispose();
        }

        public void AddParameters(List<Tensor> parameters, List<Tensor> derivatives)
        {
            parameters.Add(weight);
            parameters.Add(bias);
        }

        public void Connect(IGradientDescent model)
        {
            ConnectedFrom = model;
        }

        /// <inheritdoc/>
        public void SaveRandomState() { }

        /// <inheritdoc/>
        public void LoadRandomState() { }
    }
}

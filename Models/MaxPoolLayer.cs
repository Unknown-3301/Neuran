using ComputeShaders;
using Neuran.GradientDescent;
using Neuran.Utilities;
using SharpDX.DXGI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Models
{
    /// <summary>
    /// The max pool layer
    /// </summary>
    public class MaxPoolLayer : IGradientDescent
    {
        public int PoolSize { get; private set; }
        public int Stride { get; private set; }

        IGradientDescent layer;

        /// <summary>
        /// Creates a new instance
        /// </summary>
        /// <param name="inputWidth"></param>
        /// <param name="inputHeight"></param>
        /// <param name="inputDepth"></param>
        /// <param name="poolSize"></param>
        /// <param name="stride"></param>
        /// <param name="device"></param>
        public MaxPoolLayer(int inputWidth, int inputHeight, int inputDepth, int poolSize, int stride, CSDevice device = null)
        {
            if (device == null)
            {
                layer = new CPUMP(inputWidth, inputHeight, inputDepth, poolSize, stride);
            }
            else
            {
                layer = new GPUMP(inputWidth, inputHeight, inputDepth, poolSize, stride, device);
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

    internal class CPUMP : IGradientDescent
    {
        public Tensor[] Output { get; private set; }

        public Tensor[] Input { get; private set; }
        public Tensor[] PreLayerDer { get => preLayerDer; }

        public IGradientDescent ConnectedFrom { get; private set; }

        private int poolSize, stride;
        Tensor[] preLayerDer;

        Tensor[] inputs;
        Tensor[] outputs;

        public CPUMP(int inputWidth, int inputHeight, int inputDepth, int poolSize, int stride)
        {
            Input = new Tensor[] { new Tensor(null, inputWidth, inputHeight, inputDepth) };
            Output = new Tensor[] { new Tensor(null, inputWidth / stride, inputHeight / stride, inputDepth) };

            this.stride = stride;
            this.poolSize = poolSize;
        }

        int CeilDiv(int num, int den)
        {
            if (num % den == 0)
                return num / den;
            return (int)Math.Ceiling(num / (float)den); //dont think about doing num / den + 1 this will break for negative numbers :(
        }
        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            for (int z = 0; z < Input[0].Dimensions[2]; z++)
            {
                for (int y = 0; y < Input[0].Dimensions[1]; y++)
                {
                    for (int x = 0; x < Input[0].Dimensions[0]; x++)
                    {
                        int xmin = Math.Max(CeilDiv(x - poolSize + 1, stride), 0);
                        int ymin = Math.Max(CeilDiv(y - poolSize + 1, stride), 0);
                        int xmax = Math.Min(x / stride, outputs[pastTime].Dimensions[0] - 1);
                        int ymax = Math.Min(y / stride, outputs[pastTime].Dimensions[1] - 1);

                        float sum = 0;
                        float ivalue = inputs[pastTime][x, y, z];

                        for (int j = ymin; j <= ymax; j++)
                        {
                            for (int i = xmin; i <= xmax; i++)
                            {
                                if (ivalue == outputs[pastTime][i, j, z])
                                {
                                    sum += lossDer[0][i, j, z];
                                }
                            }
                        }

                        preLayerDer[0][x, y, z] = sum;
                    }
                }
            }

            ConnectedFrom?.Backpropagate(preLayerDer, pastTime);
        }

        public void EndGD()
        {
            preLayerDer = null;
            inputs = null;
            outputs = null;
        }

        public void PrepareGD(int maxTruncatedLength)
        {
            preLayerDer = new Tensor[] { Input[0].EmptyClone() };

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
            Input[0].Zero();
            Output[0].Zero();
        }

        public void ResetGradients() { }

        public Tensor[] Run(Tensor[] input)
        {
            input[0].CopyTo(Input[0]);

            for (int z = 0; z < Output[0].Dimensions[2]; z++)
            {
                for (int y = 0; y < Output[0].Dimensions[1]; y++)
                {
                    for (int x = 0; x < Output[0].Dimensions[0]; x++)
                    {
                        int startInputIndexX = x * stride;
                        int startInputIndexY = y * stride;
                        int startInputIndexZ = z;
                        float m = Input[0][startInputIndexX, startInputIndexY, startInputIndexZ];

                        int ymin = startInputIndexY;
                        int xmin = startInputIndexX;
                        int ymax = Math.Min(Input[0].Dimensions[1] - 1, poolSize + startInputIndexY - 1);
                        int xmax = Math.Min(Input[0].Dimensions[0] - 1, poolSize + startInputIndexX - 1);

                        for (int j = ymin; j <= ymax; j++)
                        {
                            for (int i = xmin; i <= xmax; i++)
                            {
                                m = Math.Max(m, Input[0][i, j, startInputIndexZ]);
                            }
                        }

                        Output[0][x, y, z] = m;
                    }
                }
            }

            if (inputs != null)
            {
                for (int i = inputs.Length - 2; i >= 0; i--)
                {
                    inputs[i].CopyTo(inputs[i + 1]);
                    outputs[i].CopyTo(outputs[i + 1]);
                }

                input[0].CopyTo(inputs[0]);
                Output[0].CopyTo(outputs[0]);
            }

            return Output;
        }
        public void Connect(IGradientDescent model)
        {
            ConnectedFrom = model;
        }

        public void AddParameters(List<Tensor> parameters) { }

        public void SaveRandomState() { }

        public void LoadRandomState() { }

        public void Dispose() { }
    }
    internal class GPUMP : IGradientDescent
    {
        public Tensor[] Output { get; private set; }

        public Tensor[] Input { get; private set; }

        public Tensor[] PreLayerDer { get => preLayerDer; }

        public IGradientDescent ConnectedFrom { get; private set; }

        private int poolSize, stride;
        Tensor[] preLayerDer;

        Tensor[] inputs;
        Tensor[] outputs;

        private CSDevice device;
        private ComputeShader maxPool, derMaxPool;
        private CSCBuffer<Int8> info;

        public GPUMP(int inputWidth, int inputHeight, int inputDepth, int poolSize, int stride, CSDevice device)
        {
            this.device = device;

            Input = new Tensor[] { new Tensor(device, inputWidth, inputHeight, inputDepth) };
            Output = new Tensor[] { new Tensor(device, inputWidth / stride, inputHeight / stride, inputDepth) };

            maxPool = device.CreateComputeShader(MaxPoolShaders.MaxPooling);

            this.poolSize = poolSize;
            this.stride = stride;

            info = device.CreateBuffer(new Int8()
            {
                int1 = Input[0].Dimensions[0],
                int2 = Input[0].Dimensions[1],
                int3 = Input[0].Dimensions[2],

                int4 = Output[0].Dimensions[0],
                int5 = Output[0].Dimensions[1],
                int6 = Output[0].Dimensions[2],

                int7 = poolSize,
                int8 = stride,
            }, Int8.Size);
        }
        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            inputs[pastTime].SetUAV(0);
            outputs[pastTime].SetUAV(1);
            preLayerDer[0].SetUAV(2);
            lossDer[0].SetUAV(3);

            device.SetBuffer(info, 0);

            device.SetComputeShader(derMaxPool);
            device.Dispatch((int)Math.Ceiling(Input[0].Dimensions[0] / 8f), (int)Math.Ceiling(Input[0].Dimensions[1] / 8f), Input[0].Dimensions[2]);

            float[] d1 = inputs[pastTime].GetData(); //DEBUG
            float[] d2 = outputs[pastTime].GetData(); //DEBUG
            float[] d3 = preLayerDer[0].GetData(); //DEBUG
            float[] d4 = lossDer[0].GetData(); //DEBUG

            ConnectedFrom?.Backpropagate(preLayerDer, pastTime);
        }

        public void EndGD()
        {
            preLayerDer[0].Dispose();
            preLayerDer = null;
            derMaxPool.Dispose();

            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i].Dispose();
                outputs[i].Dispose();
            }

            inputs = null;
            outputs = null;
        }

        public void PrepareGD(int maxTruncatedLength)
        {
            preLayerDer = new Tensor[] { Input[0].EmptyClone() };
            derMaxPool = device.CreateComputeShader(MaxPoolShaders.PLDerMaxPooling);

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
            Input[0].Zero();
            Output[0].Zero();
        }

        public void ResetGradients() { }

        public Tensor[] Run(Tensor[] input)
        {
            input[0].CopyTo(Input[0]);

            Input[0].SetUAV(0);
            Output[0].SetUAV(1);

            device.SetBuffer(info, 0);

            device.SetComputeShader(maxPool);

            device.Dispatch((int)Math.Ceiling(Output[0].Dimensions[0] / 8f), (int)Math.Ceiling(Output[0].Dimensions[1] / 8f), Output[0].Dimensions[2]);

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

        public void Dispose()
        {
            Input[0].Dispose();
            Output[0].Dispose();
            maxPool.Dispose();

            if (preLayerDer != null)
                EndGD();
        }

        public void Connect(IGradientDescent model)
        {
            ConnectedFrom = model;
        }

        public void AddParameters(List<Tensor> parameters) { }

        public void SaveRandomState() { }

        public void LoadRandomState() { }
    }
}

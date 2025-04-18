using ComputeShaders;
using ComputeShaders.Windows;
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
    /// The dropout layer.
    /// </summary>
    public class Dropout : IGradientDescent
    {
        /// <inheritdoc/>
        public Tensor[] PreLayerDer { get => layer.PreLayerDer; }

        IGradientDescent layer;

        /// <summary>
        /// Whether to use dropout.
        /// </summary>
        public bool Enable { get; set; } = true;

        private int saveSeed;
        private SaveRandom random;

        /// <summary>
        /// Creates a new instance
        /// </summary>
        /// <param name="size">The size of the layer (number of neurons)</param>
        /// <param name="dropoutRate">The dropout rate [0, 1]</param>
        /// <param name="random"></param>
        /// <param name="device">d3d11 device</param>
        public Dropout(int size, float dropoutRate, Random random, CSDevice device = null)
        {
            saveSeed = random.Next();
            random = new Random(saveSeed);
            this.random = new SaveRandom(random);

            if (device == null)
                layer = new DCPU(size, 1, dropoutRate, random);
            else
                layer = new DGPU(size, 1, dropoutRate, device, random);
        }

        /// <inheritdoc/>
        public IGradientDescent ConnectedFrom => layer.ConnectedFrom;

        /// <inheritdoc/>
        public Tensor[] Output => layer.Output;

        /// <inheritdoc/>
        public Tensor[] Input => layer.Input;

        /// <inheritdoc/>
        public void AddParameters(List<Tensor> parameters) => layer.AddParameters(parameters);

        /// <inheritdoc/>
        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            if (!Enable)
            {
                ConnectedFrom?.Backpropagate(lossDer, pastTime);
                return;
            }

            layer.Backpropagate(lossDer, pastTime);
        }

        /// <inheritdoc/>
        public void Connect(IGradientDescent model) => layer.Connect(model);

        /// <inheritdoc/>
        public void Dispose() => layer.Dispose();

        /// <inheritdoc/>
        public void EndGD() => layer.EndGD();

        /// <inheritdoc/>
        public void PrepareGD(int maxTruncatedLength) => layer.PrepareGD(maxTruncatedLength);

        /// <inheritdoc/>
        public void Reset() => layer.Reset();

        /// <inheritdoc/>
        public void ResetGradients() => layer.ResetGradients();

        /// <inheritdoc/>
        public Tensor[] Run(Tensor[] input)
        {
            if (!Enable)
                return input;

            return layer.Run(input);
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
    }

    internal class DCPU : IGradientDescent
    {
        public Tensor[] PreLayerDer { get => preLayer; }
        public IGradientDescent ConnectedFrom { get; private set; }

        public Tensor[] Output { get; private set; }

        public Tensor[] Input { get; private set; }

        public float Rate {  get; private set; }
        public int TensorsNum { get; private set; }

        private SaveRandom random;
        private bool[][] dropout;
        private Tensor[] preLayer;

        private bool[][][] pastDropouts;

        public DCPU(int input, int tensorsNum, float dropoutRate, Random random)
        {
            this.random = new SaveRandom(random);
            TensorsNum = tensorsNum;
            Input = new Tensor[tensorsNum];
            Output = new Tensor[tensorsNum];
            dropout = new bool[tensorsNum][];

            for (int i = 0; i < tensorsNum; i++)
            {
                Input[i] = new Tensor(null, input);
                Output[i] = new Tensor(null, input);
                dropout[i] = new bool[input];
            }

            Rate = dropoutRate;
        }
        public void AddParameters(List<Tensor> parameters) { }

        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            for (int t = 0; t < TensorsNum; t++)
            {
                Tensor loss = lossDer[t];
                bool[] pastDrops = pastDropouts[t][pastTime];

                for (int i = 0; i < Input[t].TensorLength; i++)
                {
                    preLayer[t][i] = pastDrops[i] ? 0 : loss[i] / (1 - Rate);
                }
            }

            ConnectedFrom?.Backpropagate(preLayer, pastTime);
        }

        public void Connect(IGradientDescent model)
        {
            ConnectedFrom = model;
        }

        public void Dispose()
        {
            if (preLayer != null)
                EndGD();

            Output = null;
            Input = null;
        }

        public void EndGD()
        {
            preLayer = null;
            pastDropouts = null;
        }

        public void PrepareGD(int maxTruncatedLength)
        {
            preLayer = new Tensor[TensorsNum];
            pastDropouts = new bool[TensorsNum][][];

            for (int i = 0; i < TensorsNum; i++)
            {
                preLayer[i] = Input[i].EmptyClone();

                pastDropouts[i] = new bool[maxTruncatedLength][];
                for (int n = 0; n < maxTruncatedLength; n++)
                {
                    pastDropouts[i][n] = new bool[Input[i].TensorLength];
                }
            }
        }

        public void Reset()
        {
            for (int i = 0; i < TensorsNum; i++)
            {
                Output[i].Zero();
            }
        }

        public void ResetGradients() { }

        public Tensor[] Run(Tensor[] dInput)
        {
            for (int t = 0; t < TensorsNum; t++)
            {
                Tensor input = Input[t];
                Tensor output = Output[t];
                bool[] dropO = dropout[t];

                dInput[t].CopyTo(input);

                for (int i = 0; i < Input[t].TensorLength; i++)
                {
                    float r = (float)random.NextDouble();
                    dropO[i] = r < Rate;
                    output[i] = r < Rate ? 0 : input[i] / (1 - Rate);
                }

                if (pastDropouts != null)
                {
                    for (int i = pastDropouts[t].Length - 2; i >= 0; i--)
                    {
                        for (int j = 0; j < Input[t].TensorLength; j++)
                        {
                            pastDropouts[t][i + 1][j] = pastDropouts[t][i][j];
                        }
                    }

                    for (int j = 0; j < input.TensorLength; j++)
                    {
                        pastDropouts[t][0][j] = dropO[j];
                    }
                }
            }

            return Output;
        }

        public void SaveRandomState()
        {
            random.SaveState();
        }

        public void LoadRandomState()
        {
            random.LoadSave();
        }
    }
    internal class DGPU : IGradientDescent
    {
        public Tensor[] PreLayerDer { get => preLayer; }
        public IGradientDescent ConnectedFrom { get; private set; }

        public Tensor[] Output { get; private set; }

        public Tensor[] Input { get; private set; }
        public float Rate { get; private set; }
        public int TensorsNum { get => Input.Length; }

        private SaveRandom random;
        private Tensor[] dropout;
        private Tensor[] preLayer;

        private CSDevice device;
        private ComputeShader drop;
        private ComputeShader dropDer;
        private CSCBuffer<Int3Float1> info;

        private Tensor[][] pastDropouts;

        public DGPU(int input, int tensorsNum, float dropoutRate, CSDevice device, Random random)
        {
            this.random = new SaveRandom(random);
            this.device = device;

            Input = new Tensor[tensorsNum];
            Output = new Tensor[tensorsNum];
            dropout = new Tensor[tensorsNum];

            for (int i = 0; i < tensorsNum; i++)
            {
                Input[i] = new Tensor(device, input);
                Output[i] = new Tensor(device, input);
                dropout[i] = new Tensor(device, input);
            }

            Rate = dropoutRate;

            info = device.CreateBuffer(new Int3Float1() { int1 = input, float1 = Rate }, Int3Float1.Size);
            drop = device.CreateComputeShader(DropoutShaders.Dropout);
        }
        public void SaveRandomState()
        {
            random.SaveState();
        }

        public void LoadRandomState()
        {
            random.LoadSave();
        }
        public void AddParameters(List<Tensor> parameters) { }

        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            for (int t = 0; t < TensorsNum; t++)
            {
                device.SetComputeShader(dropDer);
                device.SetBuffer(info, 0);
                dropout[t].SetUAV(0);
                lossDer[t].SetUAV(1);
                preLayer[t].SetUAV(2);

                device.Dispatch((int)Math.Ceiling(Input[t].TensorLength / 16f), 1, 1);
            }

            ConnectedFrom?.Backpropagate(preLayer, pastTime);
        }

        public void Connect(IGradientDescent model)
        {
            ConnectedFrom = model;
        }

        public void Dispose()
        {
            drop.Dispose();
            info.Dispose();

            for (int i = 0; i < TensorsNum; i++)
            {
                dropout[i].Dispose();
                Input[i].Dispose();
                Output[i].Dispose();
            }

            if (pastDropouts != null)
                EndGD();
        }

        public void EndGD()
        {
            dropDer.Dispose();

            for (int t = 0; t < TensorsNum; t++)
            {
                preLayer[t].Dispose();
                for (int i = 0; i < pastDropouts.Length; i++)
                {
                    pastDropouts[t][i].Dispose();
                }
            }

            pastDropouts = null;
        }

        public void PrepareGD(int maxTruncatedLength)
        {
            dropDer = device.CreateComputeShader(DropoutShaders.DropoutDer);
            pastDropouts = new Tensor[TensorsNum][];
            for (int t = 0; t < TensorsNum; t++)
            {
                pastDropouts[t] = new Tensor[maxTruncatedLength];

                for (int i = 0; i < maxTruncatedLength; i++)
                {
                    pastDropouts[t][i] = new Tensor(device, Input.Length);
                }
            }

            preLayer = new Tensor[TensorsNum];
            for (int i = 0; i < TensorsNum; i++)
            {
                preLayer[i] = Input[i].EmptyClone();
            }
        }

        public void Reset() 
        {
            for (int i = 0; i < TensorsNum; i++)
            {
                Output[i].Zero();
            }
        }

        public void ResetGradients() { }

        public Tensor[] Run(Tensor[] dInput)
        {
            for (int t = 0; t < TensorsNum; t++)
            {
                Tensor input = Input[t];
                Tensor output = Output[t];
                Tensor dropO = dropout[t];

                dInput[t].CopyTo(input);

                Int3Float1 inf = new Int3Float1()
                {
                    int1 = Input[t].TensorLength,
                    int2 = random.Next(),
                    float1 = Rate,
                };

                info.UpdateBuffer(inf);

                device.SetBuffer(info, 0);
                dropO.SetUAV(0);
                input.SetUAV(1);
                output.SetUAV(2);
                device.SetComputeShader(drop);

                device.Dispatch((int)Math.Ceiling(Input[t].TensorLength / 16f), 1, 1);

                if (pastDropouts != null)
                {
                    for (int i = pastDropouts.Length - 2; i >= 0; i--)
                    {
                        pastDropouts[t][i].CopyTo(pastDropouts[t][i + 1]);
                    }

                    dropO.CopyTo(pastDropouts[t][0]);
                }
            }

            return Output;
        }
    }
    internal class SaveRandom
    {
        private Random random;
        private int saveSeed;

        public SaveRandom(Random random)
        {
            this.random = random;
            SaveState();
        }

        public double NextDouble() => random.NextDouble();
        public int Next() => random.Next();

        public void SaveState()
        {
            saveSeed = random.Next();
            LoadSave();
        }
        public void LoadSave()
        {
            random = new Random(saveSeed);
        }

        public static implicit operator Random(SaveRandom random) => random.random;
    }
}

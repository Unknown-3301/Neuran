using ComputeShaders;
using Neuran.GradientDescent;
using Neuran.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Models
{
    public class RecurrentLayer : IGradientDescent
    {
        /// <inheritdoc/>
        public Tensor[] PreLayerDer { get => Layers[0].PreLayerDer; }

        /// <inheritdoc/>
        public IGradientDescent ConnectedFrom { get; private set; }

        /// <inheritdoc/>
        public Tensor[] Output { get; private set; }

        /// <inheritdoc/>
        public Tensor[] Input { get; private set; }

        public IGradientDescent[] Layers { get; private set; }

        private Tensor fullInput;
        private int input, output;

        /// <summary>
        /// Creates a new instance.
        /// </summary>
        /// <param name="input">Input neurons.</param>
        /// <param name="output">Output neurons.</param>
        /// <param name="layers">Network's layer. Note: the first layer's input length must be <paramref name="input"/> and the last layer's input length must be <paramref name="output"/></param>
        /// <param name="device"></param>
        public RecurrentLayer(int input, int output, IGradientDescent[] layers, CSDevice device = null)
        {
            this.input = input;
            this.output = output;

            Input = new Tensor[] { new Tensor(device, input) };
            fullInput = new Tensor(device, input + output);
            Output = layers[layers.Length - 1].Output;

            Layers = new IGradientDescent[layers.Length + 1];

            Layers[0] = new RecurrentCaller(this, fullInput, () => (new Tensor(device, input), new Tensor(device, output)), (fullPre, recurrentPre, normalPre) =>
            {
                fullPre.CopyTo(normalPre, new TensorBox(0, input), 0);
                fullPre.CopyTo(recurrentPre, new TensorBox(input, output), 0);
            });

            for (int i = 0; i < layers.Length; i++)
            {
                Layers[i + 1] = layers[i];
                Layers[i + 1].Connect(Layers[i]);
            }

        }

        /// <inheritdoc/>
        public void AddParameters(List<Tensor> parameters)
        {
            for (int i = Layers.Length - 1; i >= 0; i--)
            {
                Layers[i].AddParameters(parameters);
            }
        }

        /// <inheritdoc/>
        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            Layers[Layers.Length - 1].Backpropagate(lossDer, pastTime);
        }

        /// <inheritdoc/>
        public void Connect(IGradientDescent model)
        {
            ConnectedFrom = model;
            Layers[0].Connect(model);
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].Dispose();
            }
        }

        /// <inheritdoc/>
        public void EndGD()
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].EndGD();
            }
        }

        /// <inheritdoc/>
        public void PrepareGD(int maxTruncatedLength)
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].PrepareGD(maxTruncatedLength);
            }
        }

        /// <inheritdoc/>
        public void Reset()
        {
            Input[0].Zero();
            fullInput.Zero();
            Output[0].Zero();

            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].Reset();
            }
        }

        /// <inheritdoc/>
        public void ResetGradients()
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].ResetGradients();
            }
        }

        /// <inheritdoc/>
        public Tensor[] Run(Tensor[] input)
        {
            input[0].CopyTo(Input[0]);
            input[0].CopyTo(fullInput, new TensorBox(0, input[0].TensorLength), 0);
            Output[0].CopyTo(fullInput, new TensorBox(0, Output[0].TensorLength), this.input);

            Tensor[] current = new Tensor[] { fullInput };
            for (int i = 0; i < Layers.Length; i++)
            {
                current = Layers[i].Run(current);
            }
            
            return current;
        }

        /// <inheritdoc/>
        public void SaveRandomState()
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].SaveRandomState();
            }
        }

        /// <inheritdoc/>
        public void LoadRandomState()
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].LoadRandomState();
            }
        }
    }
}

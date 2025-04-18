using ComputeShaders;
using ComputeShaders.Windows;
using Neuran.GradientDescent;
using Neuran.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Models
{
    public class VectorConcatenationLayer : IGradientDescent
    {
        /// <inheritdoc/>
        public Tensor[] PreLayerDer { get; private set; }

        /// <inheritdoc/>
        public IGradientDescent ConnectedFrom { get; private set; }

        /// <inheritdoc/>
        public Tensor[] Output { get; private set; }

        /// <inheritdoc/>
        public Tensor[] Input { get; private set; }

        public VectorConcatenationLayer(Tensor[] input, CSDevice outputDevice)
        {
            for (int i = 0; i < input.Length; i++)
            {
                if (input[i].Dimensions.Length != 1)
                    throw new ArgumentException("All input tensors must be 1D tensors!");
            }

            Input = new Tensor[input.Length];
            int totalLength = 0;
            for (int i = 0; i < input.Length; i++)
            {
                totalLength += input[i].TensorLength;
                Input[i] = input[i].EmptyClone();
            }

            Output = new Tensor[] { new Tensor(outputDevice, new int[] { totalLength }) };
        }

        /// <inheritdoc/>
        public void AddParameters(List<Tensor> parameters)
        {
            
        }

        /// <inheritdoc/>
        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            int startIndex = 0;
            for (int i = 0; i < PreLayerDer.Length; i++)
            {
                lossDer[0].CopyTo(PreLayerDer[i], new TensorBox(startIndex, PreLayerDer[i].TensorLength), 0);

                startIndex += PreLayerDer[i].TensorLength;
            }

            ConnectedFrom?.Backpropagate(PreLayerDer, pastTime);
        }

        /// <inheritdoc/>
        public void Connect(IGradientDescent model)
        {
            ConnectedFrom = model;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            for (int i = 0; i < Input.Length; i++)
            {
                Input[i].Dispose();
            }

            Output[0].Dispose();

            if (PreLayerDer != null)
                EndGD();
        }

        /// <inheritdoc/>
        public void EndGD()
        {
            for (int i = 0; i < PreLayerDer.Length; i++)
            {
                PreLayerDer[i].Dispose();
            }
        }

        /// <inheritdoc/>
        public void LoadRandomState()
        {
            
        }

        /// <inheritdoc/>
        public void PrepareGD(int maxTruncatedLength)
        {
            PreLayerDer = Input.EmptyCloneArray();
        }

        /// <inheritdoc/>
        public void Reset()
        {
            
        }

        /// <inheritdoc/>
        public void ResetGradients()
        {
            
        }

        /// <inheritdoc/>
        public Tensor[] Run(Tensor[] input)
        {
            int startIndex = 0;
            for (int i = 0; i < input.Length; i++)
            {
                input[i].CopyTo(Output[0], new TensorBox(0, input[i].TensorLength), startIndex);

                startIndex += input[i].TensorLength;
            }

            return Output;
        }

        /// <inheritdoc/>
        public void SaveRandomState()
        {
            
        }
    }
}

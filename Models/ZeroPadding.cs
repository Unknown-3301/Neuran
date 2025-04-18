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
    public class ZeroPadding : IGradientDescent
    {
        /// <inheritdoc/>
        public Tensor[] PreLayerDer { get; private set; }

        /// <inheritdoc/>
        public IGradientDescent ConnectedFrom { get; private set; }

        /// <inheritdoc/>
        public Tensor[] Output { get; private set; }

        /// <inheritdoc/>
        public Tensor[] Input { get; private set; }

        public ZeroPadding(Tensor[] input, List<int[]> outputDimensions)
        {
            Input = input.EmptyCloneArray();

            Output = new Tensor[input.Length];
            for (int i = 0; i < outputDimensions.Count; i++)
            {
                Output[i] = new Tensor(input[i].device, outputDimensions[i]);
            }
        }

        /// <inheritdoc/>
        public void AddParameters(List<Tensor> parameters)
        {
            
        }

        private void CopySingleBackpropagation(Tensor lossDer, Tensor preLayer)
        {
            int startX = (lossDer.Dimensions[0] - preLayer.Dimensions[0]) / 2;
            int startY = lossDer.Dimensions.Length >= 2 ? (lossDer.Dimensions[1] - preLayer.Dimensions[1]) / 2 : 0;
            int startZ = lossDer.Dimensions.Length >= 3 ? (lossDer.Dimensions[2] - preLayer.Dimensions[2]) / 2 : 0;

            int countX = preLayer.Dimensions[0];
            int countY = preLayer.Dimensions.Length >= 2 ? preLayer.Dimensions[1] : 1;
            int countZ = preLayer.Dimensions.Length >= 3 ? preLayer.Dimensions[2] : 1;

            lossDer.CopyTo(preLayer, new TensorBox(startX, startY, startZ, countX, countY, countZ), 0);
        }
        /// <inheritdoc/>
        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            for (int i = 0; i < PreLayerDer.Length; i++)
            {
                CopySingleBackpropagation(lossDer[i], PreLayerDer[i]);
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
            for (int i = 0; i < Output.Length; i++)
            {
                Output[i].Dispose();
            }

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

        private void CopySingle(Tensor input, Tensor output)
        {
            int startX = (output.Dimensions[0] - input.Dimensions[0]) / 2;
            int startY = output.Dimensions.Length >= 2 ? (output.Dimensions[1] - input.Dimensions[1]) / 2 : 0;
            int startZ = output.Dimensions.Length >= 3 ? (output.Dimensions[2] - input.Dimensions[2]) / 2 : 0;

            int countX = input.Dimensions[0];
            int countY = input.Dimensions.Length >= 2 ? input.Dimensions[1] : 1;
            int countZ = input.Dimensions.Length >= 3 ? input.Dimensions[2] : 1;

            input.CopyTo(output, new TensorBox(0, 0, 0, countX, countY, countZ), startX, startY, startZ);
        }
        /// <inheritdoc/>
        public Tensor[] Run(Tensor[] input)
        {
            for (int i = 0; i < Input.Length; i++)
            {
                input[i].CopyTo(Input[i]);
                CopySingle(Input[i], Output[i]);
            }

            return Output;
        }

        /// <inheritdoc/>
        public void SaveRandomState()
        {
            
        }
    }
}

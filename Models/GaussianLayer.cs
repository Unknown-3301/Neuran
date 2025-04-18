using ComputeShaders;
using ComputeShaders.Windows;
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
    public class GaussianLayer : IGradientDescent
    {
        /// <inheritdoc/>
        public Tensor[] PreLayerDer { get; private set; }

        /// <inheritdoc/>
        public IGradientDescent ConnectedFrom { get; private set; }

        /// <inheritdoc/>
        public Tensor[] Output { get; private set; }

        /// <inheritdoc/>
        public Tensor[] Input { get; private set; }

        private MultiHeadLayer layer;
        private SaveRandom saveRandom;
        private GaussianTensor gaussian;

        private Tensor[] gaussiansPast;
        private Tensor[] multiLayerDer;

        public GaussianLayer(int input, int output, Random random, CSDevice device = null)
        {
            layer = new MultiHeadLayer(new IGradientDescent[]
            {
                new FullyConnectedLayer(input, output, new Identity(), random, device),  //mean
                new FullyConnectedLayer(input, output, new Exponential(), random, device),  //std
            }, false);

            gaussian = new GaussianTensor(device, random, output);

            saveRandom = new SaveRandom(random);

            Output = new Tensor[] { new Tensor(device, output) };
            Input = new Tensor[] { new Tensor(device, input) };
        }

        /// <inheritdoc/>
        public void AddParameters(List<Tensor> parameters)
        {
            layer.AddParameters(parameters);    
        }

        /// <inheritdoc/>
        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            lossDer[0].CopyTo(multiLayerDer[0]);
            lossDer[0].CopyTo(multiLayerDer[1]);

            multiLayerDer[1].Multiply(gaussiansPast[pastTime]);

            layer.Backpropagate(multiLayerDer, pastTime);

            ConnectedFrom?.Backpropagate(layer.PreLayerDer, pastTime);
        }

        /// <inheritdoc/>
        public void Connect(IGradientDescent model)
        {
            ConnectedFrom = model;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            layer.Dispose();

            Output[0].Dispose();
            Input[0].Dispose();

            if (PreLayerDer != null)
                EndGD();
        }

        /// <inheritdoc/>
        public void EndGD()
        {
            for (int i = 0; i < gaussiansPast.Length; i++)
            {
                gaussiansPast[i].Dispose();
            }

            PreLayerDer[0].Dispose();

            for (int i = 0; i < multiLayerDer.Length; i++)
            {
                multiLayerDer[i].Dispose();
            }

            gaussiansPast = null;
            multiLayerDer = null;
            PreLayerDer = null;
        }

        /// <inheritdoc/>
        public void LoadRandomState()
        {
            saveRandom.LoadSave();
        }

        /// <inheritdoc/>
        public void PrepareGD(int maxTruncatedLength)
        {
            gaussiansPast = new Tensor[maxTruncatedLength];

            for (int i = 0; i < maxTruncatedLength; i++)
            {
                gaussiansPast[i] = Output[0].EmptyClone();
            }

            PreLayerDer = Input.EmptyCloneArray();
            multiLayerDer = layer.Output.EmptyCloneArray();
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
            Tensor[] output = layer.Run(input);
            Tensor mean = output[0];
            Tensor std = output[1];

            //reparameterization trick
            gaussian.Generate(0, 1).CopyTo(Output[0]);

            if (gaussiansPast  != null)
            {
                for (int i = gaussiansPast.Length - 2; i >= 0; i--)
                {
                    gaussiansPast[i].CopyTo(gaussiansPast[i + 1]);
                }

                Output[0].CopyTo(gaussiansPast[0]);
            }

            Output[0].Multiply(std);
            Output[0].Add(mean);

            return Output;
        }

        /// <inheritdoc/>
        public void SaveRandomState()
        {
            saveRandom.SaveState();
        }
    }
}

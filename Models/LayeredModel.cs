using Neuran.GradientDescent;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Models
{
    /// <summary>
    /// The class for creating layer-based models.
    /// </summary>
    public class LayeredModel : IGradientDescent
    {
        /// <inheritdoc/>
        public Tensor[] PreLayerDer { get => Models[0].PreLayerDer; }


        /// <inheritdoc/>
        public IGradientDescent ConnectedFrom { get; private set; }

        /// <inheritdoc/>
        public Tensor[] Input { get => Models[0].Input; }
        /// <inheritdoc/>
        public Tensor[] Output { get => Models[Models.Count - 1].Output; }

        /// <summary>
        /// The model's layer.
        /// </summary>
        public List<IGradientDescent> Models { get; private set; }

        /// <summary>
        /// Creates a new instance.
        /// </summary>
        /// <param name="firstLayer"></param>
        public LayeredModel(IGradientDescent firstLayer)
        {
            Models = new List<IGradientDescent>() { firstLayer };
        }

        /// <summary>
        /// Adds a new layer to the model. Note that the dimensions and the processor type of <paramref name="layer"/> input must match the layer before it.
        /// </summary>
        /// <param name="layer"></param>
        /// <exception cref="Exception"></exception>
        public void AddLayer(IGradientDescent layer)
        {
            if (Models[Models.Count - 1].Output.Length != layer.Input.Length || Models[Models.Count - 1].Output[0].ProcessorType != layer.Input[0].ProcessorType)
                throw new Exception("The new layer and the previous layer are not compatible!");

            layer.Connect(Models[Models.Count - 1]);
            Models.Add(layer);
        }

        /// <inheritdoc/>
        public Tensor[] Run(Tensor[] input)
        {
            Tensor[] current = input;

            for (int i = 0; i < Models.Count; i++)
            {
                current = Models[i].Run(current);
            }

            return current;
        }

        /// <inheritdoc/>
        public void AddParameters(List<Tensor> parameters)
        {
            for (int i = Models.Count - 1; i >= 0; i--)
            {
                Models[i].AddParameters(parameters);
            }
        }

        /// <inheritdoc/>
        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            Models[Models.Count - 1].Backpropagate(lossDer, pastTime);
        }

        /// <inheritdoc/>
        public void Connect(IGradientDescent model)
        {
            ConnectedFrom = model;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            for (int i = 0; i < Models.Count; i++)
            {
                Models[i].Dispose();
            }
        }

        /// <inheritdoc/>
        public void EndGD()
        {
            for (int i = 0; i < Models.Count; i++)
            {
                Models[i].EndGD();
            }
        }

        /// <inheritdoc/>
        public void PrepareGD(int maxTruncatedLength)
        {
            for (int i = 0; i < Models.Count; i++)
            {
                Models[i].PrepareGD(maxTruncatedLength);
            }
        }

        /// <inheritdoc/>
        public void Reset()
        {
            for (int i = 0; i < Models.Count; i++)
            {
                Models[i].Reset();
            }
        }

        /// <inheritdoc/>
        public void ResetGradients()
        {
            for (int i = 0; i < Models.Count; i++)
            {
                Models[i].ResetGradients();
            }
        }
        /// <inheritdoc/>
        public void SaveRandomState()
        {
            for (int i = 0; i < Models.Count; i++)
            {
                Models[i].SaveRandomState();
            }
        }

        /// <inheritdoc/>
        public void LoadRandomState()
        {
            for (int i = 0; i < Models.Count; i++)
            {
                Models[i].LoadRandomState();
            }
        }
    }
}

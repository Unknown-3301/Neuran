using Neuran.GradientDescent;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Models
{
    /// <summary>
    /// A model used to recurrently call a model when backpropagating.
    /// </summary>
    public class RecurrentCaller : IGradientDescent
    {
        /// <inheritdoc/>
        public Tensor[] PreLayerDer { get => preLayerDer; }

        /// <summary>
        /// The recurrent model model to call its backpropagation function.
        /// </summary>
        public IGradientDescent RecurrentModel { get; set; }

        /// <inheritdoc/>
        public IGradientDescent ConnectedFrom { get; private set; }
        /// <inheritdoc/>
        public Tensor[] Output { get; private set; }

        /// <inheritdoc/>
        public Tensor[] Input { get; private set; }

        private Tensor[] preLayerDer;
        private Tensor recurrentPreLayerDer;
        private Func<(Tensor, Tensor)> createPreDer;
        private Action<Tensor, Tensor, Tensor> updatePreDer;

        private int maxTime;
        private int iteratedTime = 0;

        /// <summary>
        /// Creates a new instance.
        /// </summary>
        /// <param name="recurrentModel">The recurrent model.</param>
        /// <param name="createPreDer">The function to create the normal and recurrent preLayer derivative tensors (respectively) when backpropagating.</param>
        /// <param name="updatePreDer">The function to copy the recurrent data to the recurrent preLayer derivative tensor (and everthing else to the 'normal' preLayer derivative) where the first input is the next layer preLayer derivative and the second is this preLayer derivative tensor and the third is the 'normal' preLayer derivative.</param>
        public RecurrentCaller(IGradientDescent recurrentModel, Tensor fullInput, Func<(Tensor, Tensor)> createPreDer, Action<Tensor, Tensor, Tensor> updatePreDer)
        {
            RecurrentModel = recurrentModel;
            ConnectedFrom = recurrentModel.ConnectedFrom;
            Input = new Tensor[] { fullInput };
            Output = Input;

            this.createPreDer = createPreDer;
            this.updatePreDer = updatePreDer;
        }

        /// <inheritdoc/>
        public void AddParameters(List<Tensor> parameters) { }

        /// <inheritdoc/>
        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            updatePreDer(lossDer[0], recurrentPreLayerDer, preLayerDer[0]);
            ConnectedFrom?.Backpropagate(preLayerDer, pastTime);

            if (Math.Min(iteratedTime, maxTime) - pastTime - 1 > 0) // && pastTime < maxTime - 1
            {
                RecurrentModel.Backpropagate(new Tensor[] { recurrentPreLayerDer }, pastTime + 1);
            }
        }

        /// <inheritdoc/>
        public void Connect(IGradientDescent model) { ConnectedFrom = model; }

        /// <inheritdoc/>
        public void EndGD() { preLayerDer[0].Dispose(); recurrentPreLayerDer.Dispose(); }

        /// <inheritdoc/>
        public void PrepareGD(int maxTruncatedLength) 
        { 
            preLayerDer = new Tensor[1];
            (preLayerDer[0], recurrentPreLayerDer) = createPreDer();
            maxTime = maxTruncatedLength;
        }

        /// <inheritdoc/>
        public void Reset() { iteratedTime = 0; }

        /// <inheritdoc/>
        public void ResetGradients() { }

        /// <inheritdoc/>
        public Tensor[] Run(Tensor[] input)
        {
            iteratedTime++;

            return input;
        }

        /// <inheritdoc/>
        public void Dispose() => preLayerDer?[0].Dispose();

        /// <inheritdoc/>
        public void SaveRandomState() { }

        /// <inheritdoc/>
        public void LoadRandomState() { }
    }
}

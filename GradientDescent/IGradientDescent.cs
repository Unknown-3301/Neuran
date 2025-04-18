using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.GradientDescent
{
    /// <summary>
    /// The inerface that allows models to learn using Gradient Descent algorithm.
    /// </summary>
    public interface IGradientDescent : IModel
    {
        /// <summary>
        /// Stores the derivative of the loss w.r.t the input of this model.
        /// </summary>
        Tensor[] PreLayerDer { get; }

        /// <summary>
        /// The model connected to this model.
        /// </summary>
        IGradientDescent ConnectedFrom { get; }

        /// <summary>
        /// This is called at the start of the training. The model can initiate anything necessary to the training.
        /// </summary>
        /// <param name="maxTruncatedLength">The maximum length to backpropagate through the sequence.</param>
        void PrepareGD(int maxTruncatedLength);

        /// <summary>
        /// This is called at the end of the training process. The model can delete any unnecessary data from the training.
        /// </summary>
        void EndGD();

        /// <summary>
        /// Returns all the learnable parameters and their derivatives in the network. There are two important notes:
        /// <br>1- The Tensors in the list must be a reference to the actual learnable parameters and not just a value copy.</br>
        /// <br>2- The order of which the tensors where added most be the same for boo.</br>
        /// </summary>
        /// <returns></returns>
        void AddParameters(List<Tensor> parameters);

        /// <summary>
        /// Calculates the gradients in the model in a specific past time (with recurrsion if needed).
        /// <br>Note the calculated gradients should be added to previous gradients.</br>
        /// </summary>
        /// <param name="lossDer">The derivative of the loss with respect to the output of the model.</param>
        /// <param name="pastTime">The time variable. 0 means the present (input, output, ...) and N means N-th previous.</param>
        /// <returns></returns>
        void Backpropagate(Tensor[] lossDer, int pastTime);

        /// <summary>
        /// Connects this with <paramref name="model"/> such that the output of <paramref name="model"/> is the input of this.
        /// </summary>
        /// <param name="model"></param>
        void Connect(IGradientDescent model);

        /// <summary>
        /// 
        /// </summary>
        void ResetGradients();

        //While the idea of models able to connect to/from multiple models is cools, implementing it appears to be a mess as models here are
        //coded to be indpendent from each other and do not need to have the knolege of what type model they are connected to/from.
        //So it is decided that every model can be connected to/from a single model, and that any recursion connection between two different models
        //should be managed by another model that encapsulate both models and all models in between (models that connect the first model with the second)
        //this is done to ensure that all recurrent connections all from/to the same model to avoid managing "unknown" models.
        //In this case, neither the first nor the second model "know" that they have a recurrent connection, but the encapsulating model does.
        //NEW: to help with the implementation of a recurrent layer inside another recurrent layer or with the encapsulting recurrent model
        //this model might inject a a model "Recurrent Caller" at its input (first layer) which in forward pass just passes the input as it is
        //but when backpropagating it backpropagates normaly AND calls the backpropagation of a model it stores (lets call it "RecurrentModel" for now)
        //which can be the encapsulated model (this call represents the recurrent backpropagation so it would pass time as t - 1 [one step back])
    }
}

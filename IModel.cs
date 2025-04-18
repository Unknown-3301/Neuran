using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran
{
    /// <summary>
    /// The interface for all models.
    /// </summary>
    public interface IModel : IDisposable
    {
        /// <summary>
        /// The model output tensor.
        /// </summary>
        Tensor[] Output { get; }
        /// <summary>
        /// The model input tensor.
        /// </summary>
        Tensor[] Input { get; }

        /// <summary>
        /// Runs the model with input.
        /// </summary>
        /// <param name="input">Model's input.</param>
        /// <returns></returns>
        Tensor[] Run(Tensor[] input);

        /// <summary>
        /// Resets the model (for recursion).
        /// </summary>
        void Reset();

        /// <summary>
        /// Saves all the random states in the model to be loaded later.
        /// </summary>
        void SaveRandomState();

        /// <summary>
        /// Loads the latest random state stored previously (the last time <see cref="SaveRandomState"/> was called).
        /// </summary>
        void LoadRandomState();
    }
}

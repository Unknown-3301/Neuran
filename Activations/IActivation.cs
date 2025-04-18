using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Activations
{
    /// <summary>
    /// The interface for all activation functions. 
    /// </summary>
    public interface IActivation
    {
        /// <summary>
        /// Determines whether the function operation is elementwise or not. If true, <see cref="ActivateElementWise"/> and <see cref="GetDerivativeElementWise"/> can be called.
        /// </summary>
        bool ElementWise { get; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input">Input.</param>
        /// <returns></returns>
        float ActivateElementWise(float input);

        /// <summary>
        /// Takes <paramref name="beforeActivation"/> and activate it, and store the results in <paramref name="afterActivation"/>.
        /// </summary>
        /// <param name="beforeActivation">Input.</param>
        /// <param name="afterActivation">Output.</param>
        /// <returns></returns>
        void Activate(Tensor beforeActivation, Tensor afterActivation);

        /// <summary>
        /// Using whether <paramref name="input"/> or <paramref name="output"/> (whatever the implementation is) to calculate the derivative of the function with respect to its inputs.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        float GetDerivativeElementWise(float input, float output);

        /// <summary>
        /// Using whether <paramref name="beforeActivation"/> or <paramref name="afterActivation"/> (whatever the implementation is) to calculate the derivative of the function with respect to its inputs, and multiply (elementwise multiplication) the results with the <paramref name="derivatives"/>.
        /// </summary>
        /// <param name="beforeActivation">The values before activation.</param>
        /// <param name="afterActivation">The values after activation</param>
        /// <param name="derivatives">The derivative chain.</param>
        void GetDerivative(Tensor beforeActivation, Tensor afterActivation, Tensor derivatives);
    }
}

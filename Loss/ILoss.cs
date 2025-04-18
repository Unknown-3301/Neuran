using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Loss
{
    /// <summary>
    /// The interface for all loss functions.
    /// </summary>
    public interface ILoss
    {
        /// <summary>
        /// Returns the loss for the output <paramref name="predictedOutput"/> compared to <paramref name="correctOutput"/>.
        /// </summary>
        /// <param name="predictedOutput">The predicted output by the network.</param>
        /// <param name="correctOutput">The correct output.</param>
        /// <returns></returns>
        float GetLoss(Tensor predictedOutput, Tensor correctOutput);
        /// <summary>
        /// Calculates derivative value for the loss function and adds the result to <paramref name="correctOutput"/>.
        /// </summary>
        /// <param name="predictedOutput">The predicted output by the network.</param>
        /// <param name="correctOutput">The correct output.</param>
        /// <param name="derivatives">The tensor to store the derivatives results in.</param>
        /// <param name="overrideValue">Whether to override the old values in <paramref name="derivatives"/>.</param>
        /// <returns></returns>
        void GetDerivative(Tensor predictedOutput, Tensor correctOutput, Tensor derivatives, bool overrideValue = false);
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Optimizers
{
    /// <summary>
    /// The interface for all optimizers.
    /// </summary>
    public interface IOptimizer
    {
        /// <summary>
        /// Adds a new parameter.
        /// </summary>
        /// <param name="parameter"></param>
        void AddParameter(Tensor parameter);

        /// <summary>
        /// Apply the algorithm to all tensors added. and resets their gradients.
        /// </summary>
        void ApplyAll();
    }
}

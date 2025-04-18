using Neuran.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Activations
{
    public class Exponential : IActivation
    {
        /// <inheritdoc/>
        public bool ElementWise => true;

        /// <inheritdoc/>
        public void Activate(Tensor beforeActivation, Tensor afterActivation)
        {
            if (afterActivation.ProcessorType == ProcessorType.GPU)
            {
                TensorOperations.Exp(beforeActivation, afterActivation);
                return;
            }

            for (int i = 0; i < beforeActivation.TensorLength; i++)
            {
                afterActivation[i] = (float)Math.Exp(beforeActivation[i]);
            }
        }

        /// <inheritdoc/>
        public float ActivateElementWise(float input)
        {
            return (float)Math.Exp(input);
        }

        /// <inheritdoc/>
        public void GetDerivative(Tensor beforeActivation, Tensor afterActivation, Tensor derivatives)
        {
            if (afterActivation.ProcessorType == ProcessorType.GPU)
            {
                derivatives.Multiply(afterActivation);
                return;
            }

            for (int i = 0; i < afterActivation.TensorLength; i++)
            {
                derivatives[i] *= afterActivation[i];
            }
        }

        /// <inheritdoc/>
        public float GetDerivativeElementWise(float input, float output)
        {
            return output;
        }
    }
}

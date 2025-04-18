using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Activations
{
    public class Identity : IActivation
    {
        public bool ElementWise => true;

        public void Activate(Tensor beforeActivation, Tensor afterActivation)
        {
            beforeActivation.CopyTo(afterActivation);
        }

        public float ActivateElementWise(float input) => input;

        public void GetDerivative(Tensor beforeActivation, Tensor afterActivation, Tensor derivatives)
        {

        }

        public float GetDerivativeElementWise(float input, float output)
        {
            return 1;
        }
    }
}

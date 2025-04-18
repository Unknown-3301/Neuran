using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Remoting.Messaging;
using System.Text;
using System.Threading.Tasks;
using ComputeShaders;
using ComputeShaders.Windows;
using Neuran.Utilities;

namespace Neuran.Activations
{
    /// <summary>
    /// The activation function sigmoid.
    /// </summary>
    public class LeakyReLU : IActivation
    {
        /// <inheritdoc/>
        public bool ElementWise { get => true; }

        private GPUTensorProcesserApplier<Int4> function;
        private GPUTensorProcesserApplier<Int4> derFunction;

        Tensor input; //also afterActivation (output) tensor for derFunction 
        Tensor output; //also derivative tensor for derFunction 

        /// <summary>
        /// Creates a new instance
        /// </summary>
        /// <param name="device"></param>
        public LeakyReLU(CSDevice device = null)
        {
            if (device != null)
            {
                function = new GPUTensorProcesserApplier<Int4>(device, LeakyReLUShaders.LeakyReLU1, LeakyReLUShaders.LeakyReLU2, LeakyReLUShaders.LeakyReLU3, Int4.Size, () =>
                {
                    input.SetUAV(0);
                    output.SetUAV(1);
                });
                derFunction = new GPUTensorProcesserApplier<Int4>(device, LeakyReLUShaders.DerLeakyReLU1, LeakyReLUShaders.DerLeakyReLU2, LeakyReLUShaders.DerLeakyReLU3, Int4.Size, () =>
                {
                    input.SetUAV(0);
                    output.SetUAV(1);
                });
            }
        }



        /// <inheritdoc/>
        public void Activate(Tensor beforeActivation, Tensor afterActivation)
        {
            input = beforeActivation;
            output = afterActivation;

            if (input.ProcessorType == ProcessorType.GPU)
            {
                Int4 info = new Int4()
                {
                    int1 = input.Dimensions[0],
                    int2 = input.Dimensions.Length >= 2 ? input.Dimensions[1] : 0,
                    int3 = input.Dimensions.Length >= 3 ? input.Dimensions[2] : 0,
                };

                function.Run(info, input.Dimensions);

                return;
            }

            // for CPU tensors
            for (int i = 0; i < input.TensorLength; i++)
            {
                float n = input[i];
                output[i] = Math.Max(n, 0.01f * n);
            }
        }

        /// <inheritdoc/>
        public float ActivateElementWise(float input) => Math.Max(input, 0.01f * input);

        /// <inheritdoc/>
        public void GetDerivative(Tensor beforeActivation, Tensor afterActivation, Tensor derivatives)
        {
            input = afterActivation;
            output = derivatives;

            if (input.ProcessorType == ProcessorType.GPU)
            {
                Int4 info = new Int4()
                {
                    int1 = input.Dimensions[0],
                    int2 = input.Dimensions.Length >= 2 ? input.Dimensions[1] : 0,
                    int3 = input.Dimensions.Length >= 3 ? input.Dimensions[2] : 0,
                };

                derFunction.Run(info, input.Dimensions);

                return;
            }

            // for CPU tensors
            for (int i = 0; i < input.TensorLength; i++)
            {
                output[i] *= input[i] <= 0 ? 0.01f : 1;
            }
        }

        /// <inheritdoc/>
        public float GetDerivativeElementWise(float input, float output) => output <= 0 ? 0.01f : 1;
    }
}

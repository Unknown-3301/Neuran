using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http.Headers;
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
    public class Softmax : IActivation
    {
        /// <inheritdoc/>
        public bool ElementWise { get => false; }

        private GPUTensorProcesserApplier<Int4> function1;
        private GPUTensorProcesserApplier<Int4> function2;
        private GPUTensorProcesserApplier<Int4> derFunction;

        Tensor input;
        Tensor intermediate;
        Tensor output;

        Tensor afterAct;
        Tensor lossDer;
        Tensor preDer;

        /// <summary>
        /// Creates a new instance
        /// </summary>
        /// <param name="device"></param>
        /// <param name="dimensions">The dimensions</param>
        public Softmax(int[] dimensions, CSDevice device = null)
        {
            intermediate = new Tensor(device, dimensions);
            lossDer = new Tensor(device, dimensions);

            if (device != null)
            {
                function1 = new GPUTensorProcesserApplier<Int4>(device, SoftmaxShaders.Softmax1_1, SoftmaxShaders.Softmax2_1, SoftmaxShaders.Softmax3_1, Int4.Size, () =>
                {
                    input.SetUAV(0);
                });
                function2 = new GPUTensorProcesserApplier<Int4>(device, SoftmaxShaders.Softmax1_2, SoftmaxShaders.Softmax2_2, SoftmaxShaders.Softmax3_2, Int4.Size, () =>
                {
                    input.SetUAV(0);
                    output.SetUAV(1);
                });
                derFunction = new GPUTensorProcesserApplier<Int4>(device, SoftmaxShaders.DerSoftmax1, SoftmaxShaders.DerSoftmax2, SoftmaxShaders.DerSoftmax3, Int4.Size, () =>
                {
                    afterAct.SetUAV(0);
                    lossDer.SetUAV(1);
                    preDer.SetUAV(2);
                });
            }
        }



        /// <inheritdoc/>
        public void Activate(Tensor beforeActivation, Tensor afterActivation)
        {
            beforeActivation.CopyTo(intermediate);
            input = intermediate;
            output = afterActivation;

            if (input.ProcessorType == ProcessorType.GPU)
            {
                Int4 info = new Int4()
                {
                    int1 = input.Dimensions[0],
                    int2 = input.Dimensions.Length >= 2 ? input.Dimensions[1] : 0,
                    int3 = input.Dimensions.Length >= 3 ? input.Dimensions[2] : 0,
                };

                function1.Run(info, input.Dimensions);
                function2.Run(info, input.Dimensions);

                return;
            }

            // for CPU tensors
            float sum = 0;
            for (int i = 0; i < input.TensorLength; i++)
            {
                intermediate[i] = (float)Math.Exp(input[i]);
                sum += intermediate[i];
            }

            for (int i = 0; i < input.TensorLength; i++)
            {
                output[i] = intermediate[i] / sum;
            }
        }

        /// <inheritdoc/>
        public float ActivateElementWise(float input)
        {
            throw new NotImplementedException();
        }

        /// <inheritdoc/>
        public void GetDerivative(Tensor beforeActivation, Tensor afterActivation, Tensor derivatives)
        {
            afterAct = afterActivation;
            derivatives.CopyTo(lossDer);
            preDer = derivatives;

            if (input.ProcessorType == ProcessorType.GPU)
            {
                Int4 info = new Int4()
                {
                    int1 = input.Dimensions[0],
                    int2 = input.Dimensions.Length >= 2 ? input.Dimensions[1] : 0,
                    int3 = input.Dimensions.Length >= 3 ? input.Dimensions[2] : 0,
                };

                derFunction.Run(info, input.Dimensions);

                //float[] d1 = afterAct.GetData(); //DEBUG
                //float[] d2 = lossDer.GetData(); //DEBUG
                //float[] d3 = preDer.GetData(); //DEBUG

                return;
            }

            // for CPU tensors
            for (int i = 0; i < input.TensorLength; i++)
            {
                //float v = input[i];
                //output[i] *= v * (1 - v);

                float sum = 0;
                for (int j = 0; j < input.TensorLength; j++)
                {
                    sum += lossDer[j] * (i == j ? afterActivation[i] * (1 - afterActivation[i]) : -afterActivation[i] * afterActivation[j]);
                }
                derivatives[i] = sum;
            }
        }

        /// <inheritdoc/>
        public float GetDerivativeElementWise(float input, float output)
        {
            throw new NotImplementedException();
        }
    }
}

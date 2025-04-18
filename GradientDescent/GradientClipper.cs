using ComputeShaders;
using Neuran.Utilities;
using System;
using System.Collections.Generic;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.GradientDescent
{
    /// <summary>
    /// A class for gradient clipping (to avoid exploding/vanishing gradients)
    /// </summary>
    public class GradientClipper : IDisposable
    {
        TensorGPUSummation summation;
        GradientClippingInfo info;
        List<Tensor> cpuParameters;
        List<Tensor> gpuParameters;
        Tensor[] gpuParametersArray;
        Tensor result;

        /// <summary>
        /// Creates a new instance.
        /// </summary>
        /// <param name="parameters"></param>
        /// <param name="info"></param>
        public GradientClipper(List<Tensor> parameters, GradientClippingInfo info)
        {
            this.info = info;
            cpuParameters = new List<Tensor>();
            gpuParameters = new List<Tensor>();

            int maxLength = 0;
            CSDevice device = null;

            for (int i = 0; i < parameters.Count; i++)
            {
                if (parameters[i].ProcessorType == ProcessorType.GPU)
                {
                    gpuParameters.Add(parameters[i]);
                    device = parameters[i].device;
                    maxLength = Math.Max(maxLength, parameters[i].TensorLength);
                }
                else
                    cpuParameters.Add(parameters[i]);
            }

            if (device != null)
            {
                summation = new TensorGPUSummation(device, maxLength, 10);
                result = new Tensor(null, gpuParameters.Count);
                gpuParametersArray = new Tensor[gpuParameters.Count];
                for (int i = 0; i < gpuParameters.Count; i++)
                {
                    gpuParametersArray[i] = gpuParameters[i].Gradient;
                }
            }
        }

        private static float CPUL2NormSqr(Tensor t)
        {
            float sum = 0;
            for (int i = 0; i < t.TensorLength; i++)
            {
                sum += t[i] * t[i];
            }

            return sum;
        }
        /// <summary>
        /// Clips the gradients of the current parameters
        /// </summary>
        public void Clip()
        {
            float sum = 0;

            for (int i = 0; i < cpuParameters.Count; i++)
            {
                sum += CPUL2NormSqr(cpuParameters[i].Gradient);
            }

            if (summation != null)
            {
                for (int i = 0; i < gpuParametersArray.Length; i++)
                {
                    summation.L2NormSqr(result: result, inputs: gpuParametersArray[i]);
                    sum += result[0];
                }
            }

            if (sum < info.Min * info.Min || sum > info.Max * info.Max)
            {
                float l2norm = (float)Math.Sqrt(sum);
                float scale = Math.Max(info.Min, Math.Min(l2norm, info.Max)) / l2norm;

                for (int i = 0; i < cpuParameters.Count; i++)
                {
                    cpuParameters[i].Gradient.Multiply(scale);
                }
                for (int i = 0; i < gpuParameters.Count; i++)
                {
                    gpuParameters[i].Gradient.Multiply(scale);
                }
            }
        }

        public void Dispose()
        {
            summation?.Dispose();
            result?.Dispose();
        }
    }
}

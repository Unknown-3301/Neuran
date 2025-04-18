using ComputeShaders;
using ComputeShaders.Windows;
using Neuran.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Web;

namespace Neuran.Loss
{
    /// <summary>
    /// The Mean Squared Error function.
    /// </summary>
    public class MSE : ILoss
    {
        GPUTensorProcesserApplier<Int4> applier;

        Tensor pred, corr, der;

        /// <summary>
        /// Creates a new instance.
        /// </summary>
        /// <param name="device"></param>
        public MSE(CSDevice device = null)
        {
            if (device != null)
            {
                applier = new GPUTensorProcesserApplier<Int4>(device, MSEShaders.MSE1, MSEShaders.MSE2, MSEShaders.MSE3, Int4.Size, () =>
                {
                    pred.SetUAV(0);
                    corr.SetUAV(1);
                    der.SetUAV(2);
                });
            } 
                
        }

        /// <inheritdoc/>
        public void GetDerivative(Tensor predictedOutput, Tensor correctOutput, Tensor derivatives, bool overrideValue = false)
        {
            pred = predictedOutput;
            corr = correctOutput;
            der = derivatives;

            if (predictedOutput.ProcessorType == ProcessorType.GPU)
            {
                int or = overrideValue ? 1 : 0;

                Int4 info = new Int4()
                {
                    int1 = pred.Dimensions[0],
                    int2 = pred.Dimensions.Length >= 2 ? pred.Dimensions[1] : or,
                    int3 = pred.Dimensions.Length >= 3 ? pred.Dimensions[2] : or,
                    int4 = or,
                };
                applier.Run(info, pred.Dimensions);

                return;
            }

            //CPU
            for (int i = 0; i < pred.TensorLength; i++)
            {
                derivatives[i] = -2f / pred.TensorLength * (corr[i] - pred[i]) + (overrideValue ? 0 : derivatives[i]);
            }
        }

        /// <inheritdoc/>
        public float GetLoss(Tensor predictedOutput, Tensor correctOutput)
        {
            float[] preData = predictedOutput.GetData();
            float[] coData = correctOutput.GetData();

            float sum = 0;

            for (int i = 0; i < preData.Length; i++)
            {
                float diff = coData[i] - preData[i];
                sum += diff * diff;
            }

            return sum / preData.Length;
        }
    }
}

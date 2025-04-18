using ComputeShaders;
using Neuran.Utilities;
using System;
using System.Collections.Generic;
using System.Configuration;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Loss
{
    /// <summary>
    /// The Cross Entropy function.
    /// </summary>
    public class CrossEntropy : ILoss
    {
        GPUTensorProcesserApplier<Int4> applier;
        GPUTensorProcesserApplier<Int4> indexApplier;

        Tensor pred, corr, der;

        /// <summary>
        /// Creates a new instance.
        /// </summary>
        /// <param name="device"></param>
        public CrossEntropy(CSDevice device = null)
        {
            if (device != null)
            {
                applier = new GPUTensorProcesserApplier<Int4>(device, CrossEntropyShaders.CrossEntropy1, CrossEntropyShaders.CrossEntropy2, CrossEntropyShaders.CrossEntropy3, Int4.Size, () =>
                {
                    pred.SetUAV(0);
                    corr.SetUAV(1);
                    der.SetUAV(2);
                });
                indexApplier = new GPUTensorProcesserApplier<Int4>(device, CrossEntropyShaders.IndexCrossEntropy1, CrossEntropyShaders.IndexCrossEntropy2, CrossEntropyShaders.IndexCrossEntropy3, Int4.Size, () =>
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

                if (correctOutput.TensorLength == 1)
                    indexApplier.Run(info, pred.Dimensions);
                else
                    applier.Run(info, pred.Dimensions);

                return;
            }

            //CPU
            for (int i = 0; i < pred.TensorLength; i++)
            {
                float corOut = correctOutput.TensorLength == 1 ? (correctOutput[0] == i ? 1 : 0) : correctOutput[i]; // to account for index correct output

                float co = Math.Max(0.00001f, Math.Min(corOut, 0.99999f));
                float o = Math.Max(0.00001f, Math.Min(predictedOutput[i], 0.99999f));

                derivatives[i] = -(co / o) + ((1 - co) / (1 - o)) + (overrideValue ? 0 : derivatives[i]);
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
                float co = coData.Length == 1 ? (coData[0] == i ? 1 : 0) : coData[i];
                float o = preData[i];
                sum += -(co * (float)Math.Log(o) + (1 - co) * (float)Math.Log(1 - o));
            }

            return sum / preData.Length;
        }
    }
}

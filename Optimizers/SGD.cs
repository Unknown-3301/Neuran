using Neuran.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Optimizers
{
    /// <summary>
    /// Stochastic Gradient Descent.
    /// </summary>
    public class SGD : IOptimizer
    {
        /// <summary>
        /// The step size.
        /// </summary>
        public float LearningRate { get; private set; }

        private List<Tensor> tensors;
        private GPUTensorProcesserApplier<Int3Float1> applier;

        private Tensor par, grad;

        /// <summary>
        /// Creates a new instance.
        /// </summary>
        public SGD(float learningRate)
        {
            tensors = new List<Tensor>();
            LearningRate = learningRate;
        }
        /// <inheritdoc/>
        public void AddParameter(Tensor parameter)
        {
            tensors.Add(parameter);

            if (parameter.Gradient == null)
                parameter.CreateGradient();

            if (parameter.ProcessorType == ProcessorType.GPU && applier == null)
            {
                applier = new GPUTensorProcesserApplier<Int3Float1>(parameter.device, SGDShaders.SGD1, SGDShaders.SGD2, SGDShaders.SGD3, Int3Float1.Size, () =>
                {
                    par.SetUAV(0);
                    grad.SetUAV(1);
                });
            }
        }

        private void ApplySingle(Tensor param)
        {
            if (param.ProcessorType == ProcessorType.CPU)
            {
                for (int i = 0; i < param.TensorLength; i++)
                {
                    param[i] -= LearningRate * param.Gradient[i];
                }

                param.Gradient.Zero();

                return;
            }

            Int3Float1 info = new Int3Float1()
            {
                int1 = param.Dimensions[0],
                int2 = param.Dimensions.Length >= 2 ? param.Dimensions[1] : 0,
                int3 = param.Dimensions.Length >= 3 ? param.Dimensions[2] : 0,
                float1 = LearningRate,
            };

            par = param;
            grad = param.Gradient;
            applier.Run(info, param.Dimensions);

            param.Gradient.Zero();
        }

        /// <inheritdoc/>
        public void ApplyAll()
        {
            for (int i = 0; i < tensors.Count; i++)
            {
                ApplySingle(tensors[i]);
            }
        }
    }
}

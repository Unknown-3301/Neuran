using Neuran.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Optimizers
{
    /// <summary>
    /// SGD with Momentum
    /// </summary>
    public class SGDM : IOptimizer
    {
        /// <summary>
        /// The step size.
        /// </summary>
        public float LearningRate { get; private set; }
        public float MomentumFactor { get; private set; }

        private List<Tensor> tensors, momentum;

        /// <summary>
        /// Creates a new instance.
        /// </summary>
        public SGDM(float learningRate, float momentum_factor)
        {
            tensors = new List<Tensor>();
            LearningRate = learningRate;
            MomentumFactor = momentum_factor;
            momentum = new List<Tensor>();
        }
        /// <inheritdoc/>
        public void AddParameter(Tensor parameter)
        {
            tensors.Add(parameter);
            momentum.Add(parameter.EmptyClone());

            if (parameter.Gradient == null)
                parameter.CreateGradient();
        }

        private void ApplySingle(Tensor param, Tensor m)
        {
            if (param.ProcessorType == ProcessorType.CPU)
            {
                for (int i = 0; i < param.TensorLength; i++)
                {
                    m[i] = MomentumFactor * m[i] + (1 - MomentumFactor) * param.Gradient[i];
                    param[i] -= LearningRate * m[i];
                }

                param.Gradient.Zero();

                return;
            }

            param.Gradient.Zero();
        }

        /// <inheritdoc/>
        public void ApplyAll()
        {
            for (int i = 0; i < tensors.Count; i++)
            {
                ApplySingle(tensors[i], momentum[i]);
            }
        }
    }
}

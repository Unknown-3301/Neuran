using Neuran.Utilities;
using SharpDX;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Optimizers
{
    /// <summary>
    /// SWATS: SWitching from Adam To Sgd.  source: https://arxiv.org/pdf/1712.07628
    /// <br>NO GPU SUPPORT YET!</br>
    /// </summary>
    public class SWATS : IOptimizer
    {
        /// <summary>
        /// The step size.
        /// </summary>
        public float LearningRate { get; private set; }
        /// <summary>
        /// The Exponential decay rate 1
        /// </summary>
        public float Beta1 { get; set; }
        /// <summary>
        /// The Exponential decay rate 2
        /// </summary>
        public float Beta2 { get; set; }

        /// <summary>
        /// Represents the current state of this optimizer (Adam or SGD)
        /// </summary>
        public bool PhaseSGD { get; private set; }

        public float Epsilon { get; set; } = 0.0000001f;

        private float beta1T = 1, beta2T = 1, sqrtB2, lambda, sgd_lr;
        private int iteration;

        private List<Tensor> tensors, m, a, p, v;


        public SWATS(float learningRate, float beta1, float beta2)
        {
            LearningRate = learningRate;
            Beta1 = beta1;
            Beta2 = beta2;

            tensors = new List<Tensor>();
            m = new List<Tensor>();
            a = new List<Tensor>();
            p = new List<Tensor>();
            v = new List<Tensor>();
        }

        /// <inheritdoc/>
        public void AddParameter(Tensor parameter)
        {
            tensors.Add(parameter);

            m.Add(parameter.EmptyClone());
            a.Add(parameter.EmptyClone());
            p.Add(parameter.EmptyClone());
            v.Add(parameter.EmptyClone());

            if (parameter.Gradient == null)
                parameter.CreateGradient();
        }

        /// <inheritdoc/>
        public void ApplyAll()
        {
            if (PhaseSGD)
            {
                ApplySGD();
                return;
            }

            iteration++;
            beta1T *= Beta1;
            beta2T *= Beta2;

            // We do this because when beta1T and beta2T reach very low values(like 5.605194E-45)
            // The operations above become so slow.
            if (beta1T <= 5.6052E-36)
            {
                beta1T = 0;
            }
            if (beta2T <= 5.6052E-36)
            {
                beta2T = 0;
            }

            sqrtB2 = beta2T == 0 ? 1 : (float)Math.Sqrt(1 - beta2T);
            float sum1 = 0;
            float sum2 = 0;

            for (int i = 0; i < tensors.Count; i++)
            {
                ApplySingle(i, ref sum1, ref sum2);
            }

            if (sum1 == 0)
                return;

            float gamma = -sum2 / sum1;

            lambda = Beta2 * lambda + (1 - Beta2) * gamma;

            if (iteration > 1 && Math.Abs(lambda / (1 - beta2T) - gamma) < Epsilon)
            {
                PhaseSGD = true;
                sgd_lr = lambda / (1 - beta2T);
            }
        }

        private void ApplySingle(int index, ref float sum1, ref float sum2)
        {
            Tensor t = tensors[index];
            Tensor g = t.Gradient;
            Tensor mk = m[index];
            Tensor ak = a[index];
            Tensor pk = p[index];

            int length = t.TensorLength;

            for (int i = 0; i < length; i++)
            {
                mk[i] = Beta1 * mk[i] + (1 - Beta1) * g[i];
                ak[i] = Beta2 * ak[i] + (1 - Beta2) * g[i] * g[i];
                pk[i] = -LearningRate * sqrtB2 / (1 - beta1T) * mk[i] / ((float)Math.Sqrt(ak[i]) + Epsilon);
                t[i] += pk[i];

                sum1 += pk[i] * g[i];
                sum2 += pk[i] * pk[i];
            }

            g.Zero();
        }
        private void ApplySGD()
        {
            for (int i = 0; i < tensors.Count; i++)
            {
                Tensor t = tensors[i];
                Tensor vk = v[i];
                Tensor gk = t.Gradient;

                for (int j = 0; j < t.TensorLength; j++)
                {
                    vk[j] = Beta1 * vk[j] + gk[j];
                    t[j] -= (1 - Beta1) * sgd_lr * vk[j];
                }

                gk.Zero();
            }

            return;
        }
    }
}

using Neuran.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Optimizers
{
    /// <summary>
    /// Stochastic Gradient Descent.
    /// </summary>
    public class Adam : IOptimizer
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

        private List<Tensor> tensors;
        private List<Tensor> momentum1;
        private List<Tensor> momentum2;
        private GPUTensorProcesserApplier<AdamInfo> applier;

        private float beta1T, beta2T;

        private Tensor par, grad, m1, m2;

        /// <summary>
        /// Creates a new instance.
        /// </summary>
        public Adam(float learningRate, float beta1, float beta2)
        {
            tensors = new List<Tensor>();
            momentum1 = new List<Tensor>();
            momentum2 = new List<Tensor>();

            LearningRate = learningRate;
            Beta1 = beta1;
            Beta2 = beta2;
            beta1T = beta1;
            beta2T = beta2;
        }
        /// <inheritdoc/>
        public void AddParameter(Tensor parameter)
        {
            tensors.Add(parameter);

            momentum1.Add(parameter.EmptyClone());
            momentum2.Add(parameter.EmptyClone());

            if (parameter.Gradient == null)
                parameter.CreateGradient();

            if (parameter.ProcessorType == ProcessorType.GPU && applier == null)
            {
                applier = new GPUTensorProcesserApplier<AdamInfo>(parameter.device, AdamShaders.Adam1, AdamShaders.Adam2, AdamShaders.Adam3, AdamInfo.Size, () =>
                {
                    par.SetUAV(0);
                    grad.SetUAV(1);
                    m1.SetUAV(2);
                    m2.SetUAV(3);
                });
            }
        }

        private void ApplySingle(Tensor param, Tensor mo1, Tensor mo2)
        {
            m1 = mo1;
            m2 = mo2;

            if (param.ProcessorType == ProcessorType.CPU)
            {
                for (int i = 0; i < param.TensorLength; i++)
                {
                    float g = param.Gradient[i];
                    m1[i] = Beta1 * m1[i] + (1 - Beta1) * g;
                    m2[i] = Beta2 * m2[i] + (1 - Beta2) * g * g;
                    float md = m1[i] / (1 - beta1T);
                    float ud = m2[i] / (1 - beta2T);
                    param[i] -= LearningRate * md / ((float)Math.Sqrt(ud) + 0.0000001f);
                }

                param.Gradient.Zero();

                return;
            }

            AdamInfo info = new AdamInfo()
            {
                Width = param.Dimensions[0],
                Height = param.Dimensions.Length >= 2 ? param.Dimensions[1] : 0,
                Depth = param.Dimensions.Length >= 3 ? param.Dimensions[2] : 0,
                Beta1 = Beta1,
                Beta2 = Beta2,
                Beta1T = beta1T,
                Beta2T = beta2T,
                LearningRate = LearningRate,
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
                ApplySingle(tensors[i], momentum1[i], momentum2[i]);
            }

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
        }
    }

    /// <summary>
    /// The struct used to store information for GPU Adam.
    /// </summary>
    internal struct AdamInfo
    {
        public float Beta1;
        public float Beta2;
        public float Beta1T;
        public float Beta2T;

        public int Width;
        public int Height;
        public int Depth;

        public float LearningRate;

        public static int Size { get => sizeof(float) * 5 + sizeof(int) * 3; }
    }
}

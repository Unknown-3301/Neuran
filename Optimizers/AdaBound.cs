using Neuran.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Optimizers
{
    // INCOMPLETE


    /// <summary>
    /// Original paper: https://arxiv.org/pdf/1902.09843.pdf
    /// Original Implementation: https://github.com/Luolc/AdaBound/blob/2e928c3007a2fc44af0e4c97e343e1fed6986e44/adabound/adabound.py#L100
    /// </summary>
    //public class AdaBound : IOptimizer
    //{
    //    //Here if I am not wrong the optimizer starts like Adam and converges slowly to SGD
    //    //It clamps the learning rate by a range that starts like [0,♾️) and converges to [FinalLearningRate, FinalLearningRate] (FinalLearningRate represents SGD learning rate)

    //    /// <summary>
    //    /// The init step size.
    //    /// </summary>
    //    public float LearningRate { get; private set; }
    //    /// <summary>
    //    /// The final step size.
    //    /// </summary>
    //    public float FinalLearningRate { get; private set; }
    //    /// <summary>
    //    /// The Exponential decay rate 1
    //    /// </summary>
    //    public float Beta1 { get; set; }
    //    /// <summary>
    //    /// The Exponential decay rate 2
    //    /// </summary>
    //    public float Beta2 { get; set; }
    //    public float Gamma {  get; set; }
    //    public float Epsilon { get; set; } = 0.0000001f;

    //    private List<Tensor> tensors;
    //    private List<Tensor> momentum1;
    //    private List<Tensor> momentum2;
    //    private GPUTensorProcesserApplier<AdamInfo> applier;

    //    private float beta1T, beta2T;
    //    private float correctedStepSize;
    //    private int time = 1;
    //    private float sqrtTime = 1;

    //    private Tensor par, grad, m1, m2;

    //    /// <summary>
    //    /// Creates a new instance.
    //    /// </summary>
    //    public AdaBound(float learningRate = 0.001f, float finalLearningRate = 0.1f, float beta1 = 0.9f, float beta2 = 0.999f, float gamma = 0.001f)
    //    {
    //        tensors = new List<Tensor>();
    //        momentum1 = new List<Tensor>();
    //        momentum2 = new List<Tensor>();

    //        LearningRate = learningRate;
    //        FinalLearningRate = finalLearningRate;
    //        Gamma = gamma;
    //        Beta1 = beta1;
    //        Beta2 = beta2;
    //        beta1T = beta1;
    //        beta2T = beta2;
    //        correctedStepSize = LearningRate * (float)Math.Sqrt(1 - beta2T) / (1 - beta1T);
    //    }
    //    /// <inheritdoc/>
    //    public void AddParameter(Tensor parameter)
    //    {
    //        tensors.Add(parameter);

    //        momentum1.Add(parameter.EmptyClone());
    //        momentum2.Add(parameter.EmptyClone());

    //        if (parameter.Gradient == null)
    //            parameter.CreateGradient();

    //        if (parameter.ProcessorType == ProcessorType.GPU && applier == null)
    //        {
    //            applier = new GPUTensorProcesserApplier<AdamInfo>(parameter.device, AdamShaders.Adam1, AdamShaders.Adam2, AdamShaders.Adam3, AdamInfo.Size, () =>
    //            {
    //                par.SetUAV(0);
    //                grad.SetUAV(1);
    //                m1.SetUAV(2);
    //                m2.SetUAV(3);
    //            });
    //        }
    //    }

    //    private void ApplySingle(Tensor param, Tensor mo1, Tensor mo2)
    //    {
    //        m1 = mo1;
    //        m2 = mo2;

    //        if (param.ProcessorType == ProcessorType.CPU)
    //        {
    //            for (int i = 0; i < param.Length; i++)
    //            {
    //                float g = param.Gradient[i];
    //                m1[i] = Beta1 * m1[i] + (1 - Beta1) * g;
    //                m2[i] = Beta2 * m2[i] + (1 - Beta2) * g * g;

    //                float denom = (float)Math.Sqrt(m2[i]) + Epsilon;
    //                float lower_bound = FinalLearningRate * (1 - 1 / (Gamma * time + 1));
    //                float higher_bound = FinalLearningRate * (1 + 1 / (Gamma * time));

    //                float clamped = Math.Max(lower_bound, Math.Min(correctedStepSize / denom, higher_bound)) * m1[i] / sqrtTime;

    //                param[i] -= clamped;
    //            }

    //            param.Gradient.Zero();

    //            return;
    //        }

    //        throw new NotImplementedException("Not implemented for GPU yet!");

    //        AdamInfo info = new AdamInfo()
    //        {
    //            Width = param.Dimensions[0],
    //            Height = param.Dimensions.Length >= 2 ? param.Dimensions[1] : 0,
    //            Depth = param.Dimensions.Length >= 3 ? param.Dimensions[2] : 0,
    //            Beta1 = Beta1,
    //            Beta2 = Beta2,
    //            Beta1T = beta1T,
    //            Beta2T = beta2T,
    //            LearningRate = LearningRate,
    //        };

    //        par = param;
    //        grad = param.Gradient;
    //        applier.Run(info, param.Dimensions);

    //        param.Gradient.Zero();
    //    }

    //    /// <inheritdoc/>
    //    public void ApplyAll()
    //    {
    //        for (int i = 0; i < tensors.Count; i++)
    //        {
    //            ApplySingle(tensors[i], momentum1[i], momentum2[i]);
    //        }

    //        beta1T *= Beta1;
    //        beta2T *= Beta2;
    //        correctedStepSize = LearningRate * (float)Math.Sqrt(1 - beta2T) / (1 - beta1T);
    //        time++;
    //        sqrtTime = (float)Math.Sqrt(time);

    //        // We do this because when beta1T and beta2T reach very low values(like 5.605194E-45)
    //        // The operations above become so slow.
    //        if (beta1T <= 5.6052E-36)
    //        {
    //            beta1T = 0;
    //        }
    //        if (beta2T <= 5.6052E-36)
    //        {
    //            beta2T = 0;
    //        }
    //    }
    //}
}

using ComputeShaders;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace Neuran.Utilities
{
    public class GaussianTensor
    {
        public Tensor Tensor { get; private set; }

        private GPUTensorProcesserApplier<GInfo> applier;
        private GPUTensorProcesserApplier<Int4> m_applier;
        private Random random;

        private Tensor mean;
        private Tensor std;

        public GaussianTensor(CSDevice device, Random random, params int[] dimensions)
        {
            Tensor = new Tensor(device, dimensions);
            this.random = random;

            if (device != null)
            {
                applier = new GPUTensorProcesserApplier<GInfo>(device, GaussianTensorShaders.G1M, GaussianTensorShaders.G2M, GaussianTensorShaders.G3M, GInfo.Size, () =>
                {
                    Tensor.SetUAV(0);
                });
                m_applier = new GPUTensorProcesserApplier<Int4>(device, GaussianTensorShaders.G1, GaussianTensorShaders.G2, GaussianTensorShaders.G3, Int4.Size, () =>
                {
                    Tensor.SetUAV(0);
                    mean.SetUAV(1);
                    std.SetUAV(2);
                });

            }
        }

        public Tensor Generate(float mean, float std)
        {
            if (Tensor.ProcessorType == ProcessorType.GPU)
            {
                applier.Run(new GInfo()
                {
                    int1 = Tensor.Dimensions[0],
                    int2 = Tensor.Dimensions.Length >= 2 ? Tensor.Dimensions[1] : 0,
                    int3 = Tensor.Dimensions.Length >= 3 ? Tensor.Dimensions[2] : 0,
                    int4 = random.Next(),

                    mean = mean,
                    std = std
                }, Tensor.Dimensions);
                return Tensor;
            }

            for (int i = 0; i < Tensor.TensorLength; i++)
            {
                Tensor[i] = mean + std * (float)UtilitiesFuncs.RandomGaussain(0, 1, random);
            }

            return Tensor;
        }

        public Tensor Generate(Tensor mean, Tensor std)
        {
            if (Tensor.ProcessorType == ProcessorType.GPU)
            {
                this.mean = mean;
                this.std = std;

                m_applier.Run(new Int4()
                {
                    int1 = Tensor.Dimensions[0],
                    int2 = Tensor.Dimensions.Length >= 2 ? Tensor.Dimensions[1] : 0,
                    int3 = Tensor.Dimensions.Length >= 3 ? Tensor.Dimensions[2] : 0,
                    int4 = random.Next()
                }, Tensor.Dimensions);
                return Tensor;
            }

            for (int i = 0; i < Tensor.TensorLength; i++)
            {
                Tensor[i] = mean[i] + std[i] * (float)UtilitiesFuncs.RandomGaussain(0, 1, random);
            }

            return Tensor;
        }
    }

    internal struct GInfo
    {
        public int int1;
        public int int2;
        public int int3;
        public int int4;

        public float mean;
        public float std;

        float dummy_;
        float dummy__;

        public static int Size { get => sizeof(int) * 4 + sizeof(float) * 4; }
    }
}

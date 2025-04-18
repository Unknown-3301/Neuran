using ComputeShaders;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran
{
    internal class GPUTensorProcesserApplier<TInfo> : IDisposable where TInfo : struct
    {
        CSDevice device;
        ComputeShader s1, s2, s3;
        byte[] s1_bytes, s2_bytes, s3_bytes;
        CSCBuffer<TInfo> info;
        int info_size;

        public Action SetTensors { get; set; }

        public GPUTensorProcesserApplier(CSDevice device, byte[] s1_bytes, byte[] s2_bytes, byte[] s3_bytes, int tInfo_size, Action setTensors)
        {
            this.device = device;
            info_size = tInfo_size;
            this.s1_bytes = s1_bytes;
            this.s2_bytes = s2_bytes;
            this.s3_bytes = s3_bytes;

            SetTensors = setTensors;
        }
        public GPUTensorProcesserApplier(CSDevice device, byte[] s1_bytes, byte[] s2_bytes, byte[] s3_bytes, int tInfo_size ,CSCBuffer<TInfo> info, Action setTensors)
        {
            this.device = device;
            info_size = tInfo_size;
            this.s1_bytes = s1_bytes;
            this.s2_bytes = s2_bytes;
            this.s3_bytes = s3_bytes;
            this.info = info;

            SetTensors = setTensors;
        }

        private void GPUInit(int order)
        {
            if (info == null)
                info = device.CreateBuffer(new TInfo(), info_size);

            switch (order)
            {
                case 1:
                    if (s1 == null)
                        s1 = device.CreateComputeShader(s1_bytes);
                    break;
                case 2:
                    if (s2 == null)
                        s2 = device.CreateComputeShader(s2_bytes);
                    break;
                case 3:
                    if (s3 == null)
                        s3 = device.CreateComputeShader(s3_bytes);
                    break;
            }
        }

        private void Run1(int length)
        {
            device.SetComputeShader(s1);

            SetTensors();

            device.SetBuffer(info, 0);

            device.Dispatch((int)Math.Ceiling(length / 16f), 1, 1);
        }
        private void Run2(int width, int height)
        {
            device.SetComputeShader(s2);

            SetTensors();

            device.SetBuffer(info, 0);

            device.Dispatch((int)Math.Ceiling(width / 8f), (int)Math.Ceiling(height / 8f), 1);
        }
        private void Run3(int width, int height, int depth)
        {
            device.SetComputeShader(s3);

            SetTensors();

            device.SetBuffer(info, 0);

            device.Dispatch((int)Math.Ceiling(width / 8f), (int)Math.Ceiling(height / 8f), depth);
        }

        public void Run(TInfo info, int[] dimensions)
        {
            GPUInit(dimensions.Length);

            this.info.UpdateBuffer(info);

            switch (dimensions.Length)
            {
                case 1: Run1(dimensions[0]); break;
                case 2: Run2(dimensions[0], dimensions[1]); break;
                case 3: Run3(dimensions[0], dimensions[1], dimensions[2]); break;
            }
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            info?.Dispose();
            s1?.Dispose();
            s2?.Dispose();
            s3?.Dispose();
        }
    }
}

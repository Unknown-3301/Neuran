using ComputeShaders;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Utilities
{
    /// <summary>
    /// A class that contains some useful tensor (cpu/gpu) operations 
    /// </summary>
    public static class TensorOperations
    {
        private static ComputeShader add1, add2, add3, div1, div2, div3, zero1, zero2 , zero3;
        private static CSCBuffer<Int4> info;

        private static GPUTensorProcesserApplier<Int4> addApplier, divApplier, zeroApplier;

        private static void InitAdd(Tensor t)
        {
            if (info == null)
                info = t.device.CreateBuffer(new Int4(), Int4.Size);

            switch (t.Dimensions.Length)
            {
                case 1:
                    if (add1 == null)
                        add1 = t.device.CreateComputeShader(TensorsOperationShaders.Add1);
                    break;
                case 2:
                    if (add2 == null)
                        add2 = t.device.CreateComputeShader(TensorsOperationShaders.Add2);
                    break;
                case 3:
                    if (add3 == null)
                        add3 = t.device.CreateComputeShader(TensorsOperationShaders.Add3);
                    break;
            }
        }

        private static void AddGPU1(Tensor t1, Tensor t2)
        {
            info.UpdateBuffer(new Int4() { int1 = t1.Length });

            t1.device.SetComputeShader(add1);

            t1.SetUAV(0);
            t2.SetUAV(1);
            t1.device.SetBuffer(info, 0);

            t1.device.Dispatch((int)Math.Ceiling(t1.Length / 16f), 1, 1);
        }
        private static void AddGPU2(Tensor t1, Tensor t2)
        {
            info.UpdateBuffer(new Int4() { int1 = t1.Dimensions[0], int2 = t1.Dimensions[1] });

            t1.device.SetComputeShader(add2);

            t1.SetUAV(0);
            t2.SetUAV(1);
            t1.device.SetBuffer(info, 0);

            t1.device.Dispatch((int)Math.Ceiling(t1.Dimensions[0] / 8f), (int)Math.Ceiling(t1.Dimensions[1] / 8f), 1);
        }
        private static void AddGPU3(Tensor t1, Tensor t2)
        {
            info.UpdateBuffer(new Int4() { int1 = t1.Dimensions[0], int2 = t1.Dimensions[1], int3 = t1.Dimensions[2] });

            t1.device.SetComputeShader(add3);

            t1.SetUAV(0);
            t2.SetUAV(1);
            t1.device.SetBuffer(info, 0);

            t1.device.Dispatch((int)Math.Ceiling(t1.Dimensions[0] / 8f), (int)Math.Ceiling(t1.Dimensions[1] / 8f), t1.Dimensions[2]);
        }

        /// <summary>
        /// Applies the addition operation for both tensors and save results in <paramref name="tensor"/> (this).
        /// <br>Note that both tensors must have the same <see cref="ProcessorType"/> and Dimensions.</br>
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="other"></param>
        public static void Add(this Tensor tensor, Tensor other)
        {
            if (tensor.ProcessorType == ProcessorType.CPU)
            {
                for (int i = 0; i < tensor.Length; i++)
                {
                    tensor[i] += other[i];
                }

                return;
            }

            InitAdd(tensor);

            switch (tensor.Dimensions.Length)
            {
                case 1: AddGPU1(tensor, other); break;
                case 2: AddGPU2(tensor, other); break;
                case 3: AddGPU3(tensor, other); break;
            }
        }


        private static void InitDiv(Tensor t)
        {
            if (info == null)
                info = t.device.CreateBuffer(new Int4(), Int4.Size);

            switch (t.Dimensions.Length)
            {
                case 1:
                    if (div1 == null)
                        div1 = t.device.CreateComputeShader(TensorsOperationShaders.Div1);
                    break;
                case 2:
                    if (div2 == null)
                        div2 = t.device.CreateComputeShader(TensorsOperationShaders.Div2);
                    break;
                case 3:
                    if (div3 == null)
                        div3 = t.device.CreateComputeShader(TensorsOperationShaders.Div3);
                    break;
            }
        }

        private static void DivGPU1(Tensor t1, int den)
        {
            info.UpdateBuffer(new Int4() { int1 = t1.Length, int2 = den });

            t1.device.SetComputeShader(div1);

            t1.SetUAV(0);
            t1.device.SetBuffer(info, 0);

            t1.device.Dispatch((int)Math.Ceiling(t1.Length / 16f), 1, 1);
        }
        private static void DivGPU2(Tensor t1, int den)
        {
            info.UpdateBuffer(new Int4() { int1 = t1.Dimensions[0], int2 = t1.Dimensions[1], int3 = den });

            t1.device.SetComputeShader(div2);

            t1.SetUAV(0);
            t1.device.SetBuffer(info, 0);

            t1.device.Dispatch((int)Math.Ceiling(t1.Dimensions[0] / 8f), (int)Math.Ceiling(t1.Dimensions[1] / 8f), 1);
        }
        private static void DivGPU3(Tensor t1, int den)
        {
            info.UpdateBuffer(new Int4() { int1 = t1.Dimensions[0], int2 = t1.Dimensions[1], int3 = t1.Dimensions[2], int4 = den });

            t1.device.SetComputeShader(div3);

            t1.SetUAV(0);
            t1.device.SetBuffer(info, 0);

            t1.device.Dispatch((int)Math.Ceiling(t1.Dimensions[0] / 8f), (int)Math.Ceiling(t1.Dimensions[1] / 8f), t1.Dimensions[2]);
        }

        /// <summary>
        /// Applies the addition operation for both tensors and save results in <paramref name="tensor"/>.
        /// <br>Note that both tensors must have the same <see cref="ProcessorType"/> and Dimensions.</br>
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="denominator"></param>
        public static void Divide(this Tensor tensor, int denominator)
        {
            if (tensor.ProcessorType == ProcessorType.CPU)
            {
                for (int i = 0; i < tensor.Length; i++)
                {
                    tensor[i] /= denominator;
                }

                return;
            }

            InitDiv(tensor);

            switch (tensor.Dimensions.Length)
            {
                case 1: DivGPU1(tensor, denominator); break;
                case 2: DivGPU2(tensor, denominator); break;
                case 3: DivGPU3(tensor, denominator); break;
            }
        }

        private static void InitZero(Tensor t)
        {
            if (info == null)
                info = t.device.CreateBuffer(new Int4(), Int4.Size);

            switch (t.Dimensions.Length)
            {
                case 1:
                    if (zero1 == null)
                        zero1 = t.device.CreateComputeShader(TensorsOperationShaders.Zero1);
                    break;
                case 2:
                    if (zero2 == null)
                        zero2 = t.device.CreateComputeShader(TensorsOperationShaders.Zero2);
                    break;
                case 3:
                    if (zero3 == null)
                        zero3 = t.device.CreateComputeShader(TensorsOperationShaders.Zero3);
                    break;
            }
        }

        private static void ZeroGPU1(Tensor t1)
        {
            info.UpdateBuffer(new Int4() { int1 = t1.Length });

            t1.device.SetComputeShader(div1);

            t1.SetUAV(0);
            t1.device.SetBuffer(info, 0);

            t1.device.Dispatch((int)Math.Ceiling(t1.Length / 16f), 1, 1);
        }
        private static void ZeroGPU2(Tensor t1)
        {
            info.UpdateBuffer(new Int4() { int1 = t1.Dimensions[0], int2 = t1.Dimensions[1]});

            t1.device.SetComputeShader(div2);

            t1.SetUAV(0);
            t1.device.SetBuffer(info, 0);

            t1.device.Dispatch((int)Math.Ceiling(t1.Dimensions[0] / 8f), (int)Math.Ceiling(t1.Dimensions[1] / 8f), 1);
        }
        private static void ZeroGPU3(Tensor t1)
        {
            info.UpdateBuffer(new Int4() { int1 = t1.Dimensions[0], int2 = t1.Dimensions[1], int3 = t1.Dimensions[2] });

            t1.device.SetComputeShader(div3);

            t1.SetUAV(0);
            t1.device.SetBuffer(info, 0);

            t1.device.Dispatch((int)Math.Ceiling(t1.Dimensions[0] / 8f), (int)Math.Ceiling(t1.Dimensions[1] / 8f), t1.Dimensions[2]);
        }

        /// <summary>
        /// Sets the value of every element in the tensor to 0.
        /// </summary>
        /// <param name="tensor"></param>
        public unsafe static void Zero(this Tensor tensor)
        {
            if (tensor.ProcessorType == ProcessorType.CPU)
            {
                for (int i = 0; i < tensor.Length; i++)
                {
                    tensor[i] = 0;
                }

                return;
            }

            InitZero(tensor);

            switch (tensor.Dimensions.Length)
            {
                case 1: ZeroGPU1(tensor); break;
                case 2: ZeroGPU2(tensor); break;
                case 3: ZeroGPU3(tensor); break;
            }
        }
    }
}

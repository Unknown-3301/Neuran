using ComputeShaders;
using SharpDX;
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
        private static GPUTensorProcesserApplier<Int4> addApplier, zeroApplier, mulApplier, expApplier;
        private static GPUTensorProcesserApplier<Int3Float1> valMulApplier;
        private static Tensor operationTensor1, operationTensor2;

        public static void Dispose()
        {
            addApplier?.Dispose();
            valMulApplier?.Dispose();
            zeroApplier?.Dispose();
            mulApplier?.Dispose();
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
                for (int i = 0; i < tensor.TensorLength; i++)
                {
                    tensor[i] += other[i];
                }

                return;
            }

            operationTensor1 = tensor;
            operationTensor2 = other;

            if (addApplier == null)
                addApplier = new GPUTensorProcesserApplier<Int4>(tensor.device, TensorsOperationShaders.Add1, TensorsOperationShaders.Add2, TensorsOperationShaders.Add3, Int4.Size, () =>
                {
                    float[] d1 = operationTensor1.GetData();
                    float[] d2 = operationTensor2.GetData();

                    operationTensor1.SetUAV(0);
                    operationTensor2.SetUAV(1);
                });

            Int4 info = new Int4()
            {
                int1 = tensor.Dimensions[0],
                int2 = tensor.Dimensions.Length >= 2 ? tensor.Dimensions[1] : 0,
                int3 = tensor.Dimensions.Length >= 3 ? tensor.Dimensions[2] : 0,
            };

            addApplier.Run(info, tensor.Dimensions);
        }
        /// <summary>
        /// Applies the addition operation for both tensors and save results in <paramref name="tensor"/>.
        /// <br>Note that both tensors must have the same <see cref="ProcessorType"/> and Dimensions.</br>
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="denominator"></param>
        public static void Divide(this Tensor tensor, float denominator)
        {
            Multiply(tensor, 1 / denominator);
        }
        /// <summary>
        /// Applies the addition operation for both tensors and save results in <paramref name="tensor"/>.
        /// <br>Note that both tensors must have the same <see cref="ProcessorType"/> and Dimensions.</br>
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="value"></param>
        public static void Multiply(this Tensor tensor, float value)
        {
            if (tensor.ProcessorType == ProcessorType.CPU)
            {
                for (int i = 0; i < tensor.TensorLength; i++)
                {
                    tensor[i] *= value;
                }

                return;
            }

            operationTensor1 = tensor;

            if (valMulApplier == null)
                valMulApplier = new GPUTensorProcesserApplier<Int3Float1>(tensor.device, TensorsOperationShaders.ValMul1, TensorsOperationShaders.ValMul2, TensorsOperationShaders.ValMul3, Int3Float1.Size, () =>
                {
                    operationTensor1.SetUAV(0);
                });

            Int3Float1 info = new Int3Float1()
            {
                int1 = tensor.Dimensions[0],
                int2 = tensor.Dimensions.Length >= 2 ? tensor.Dimensions[1] : 0,
                int3 = tensor.Dimensions.Length >= 3 ? tensor.Dimensions[2] : 0,
                float1 = value,
            };

            valMulApplier.Run(info, tensor.Dimensions);
        }

        /// <summary>
        /// Sets the value of every element in the tensor to 0.
        /// </summary>
        /// <param name="tensor"></param>
        public unsafe static void Zero(this Tensor tensor)
        {
            if (tensor.ProcessorType == ProcessorType.CPU)
            {
                for (int i = 0; i < tensor.TensorLength; i++)
                {
                    tensor[i] = 0;
                }

                return;
            }

            operationTensor1 = tensor;

            if (zeroApplier == null)
                zeroApplier = new GPUTensorProcesserApplier<Int4>(tensor.device, TensorsOperationShaders.Zero1, TensorsOperationShaders.Zero2, TensorsOperationShaders.Zero3, Int4.Size, () =>
                {
                    operationTensor1.SetUAV(0);
                });

            Int4 info = new Int4()
            {
                int1 = tensor.Dimensions[0],
                int2 = tensor.Dimensions.Length >= 2 ? tensor.Dimensions[1] : 0,
                int3 = tensor.Dimensions.Length >= 3 ? tensor.Dimensions[2] : 0,
            };

            zeroApplier.Run(info, tensor.Dimensions);
        }

        /// <summary>
        /// Applies the multiplication operation for both tensors and save results in <paramref name="tensor"/> (this).
        /// <br>Note that both tensors must have the same <see cref="ProcessorType"/> and Dimensions.</br>
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="other"></param>
        public static void Multiply(this Tensor tensor, Tensor other)
        {
            if (tensor.ProcessorType == ProcessorType.CPU)
            {
                for (int i = 0; i < tensor.TensorLength; i++)
                {
                    tensor[i] *= other[i];
                }

                return;
            }

            operationTensor1 = tensor;
            operationTensor2 = other;

            if (mulApplier == null)
                mulApplier = new GPUTensorProcesserApplier<Int4>(tensor.device, TensorsOperationShaders.Mul1, TensorsOperationShaders.Mul2, TensorsOperationShaders.Mul3, Int4.Size, () =>
                {
                    operationTensor1.SetUAV(0);
                    operationTensor2.SetUAV(1);
                });

            Int4 info = new Int4()
            {
                int1 = tensor.Dimensions[0],
                int2 = tensor.Dimensions.Length >= 2 ? tensor.Dimensions[1] : 0,
                int3 = tensor.Dimensions.Length >= 3 ? tensor.Dimensions[2] : 0,
            };

            mulApplier.Run(info, tensor.Dimensions);
        }

        /// <summary>
        /// Calculates the exp() of every element in <paramref name="other"/> and save the result in <paramref name="tensor"/> (this).
        /// <br>Note that both tensors must have the same <see cref="ProcessorType"/> and Dimensions.</br>
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        public static void Exp(Tensor input, Tensor output)
        {
            if (input.ProcessorType == ProcessorType.CPU)
            {
                for (int i = 0; i < input.TensorLength; i++)
                {
                    output[i] = (float)Math.Exp(input[i]);
                }

                return;
            }

            operationTensor1 = input;
            operationTensor2 = output;

            if (expApplier == null)
                expApplier = new GPUTensorProcesserApplier<Int4>(input.device, TensorsOperationShaders.Exp1, TensorsOperationShaders.Exp2, TensorsOperationShaders.Exp3, Int4.Size, () =>
                {
                    operationTensor1.SetUAV(0);
                    operationTensor2.SetUAV(1);
                });

            Int4 info = new Int4()
            {
                int1 = input.Dimensions[0],
                int2 = input.Dimensions.Length >= 2 ? input.Dimensions[1] : 0,
                int3 = input.Dimensions.Length >= 3 ? input.Dimensions[2] : 0,
            };

            expApplier.Run(info, input.Dimensions);
        }
    }
}

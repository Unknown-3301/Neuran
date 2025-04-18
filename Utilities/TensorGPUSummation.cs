using ComputeShaders;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Utilities
{
    /// <summary>
    /// A class tha helps in getting the summation of a tensor (or a number of tensors as batches seperately).
    /// </summary>
    public class TensorGPUSummation : IDisposable
    {
        /// <summary>
        /// The input tensor to sum data from.
        /// </summary>
        public Tensor Input { get => shaderInput; }

        GPUTensorProcesserApplier<Int4> applier;
        GPUTensorProcesserApplier<Int4> l2Applier;

        private Tensor copyInput;
        private Tensor copyOutput;

        private Tensor shaderInput;
        private Tensor shaderOutput;

        private CSCBuffer<Int8> info;
        private ComputeShader shader;
        private CSDevice device;

        public TensorGPUSummation(CSDevice device, int tensorMaxLength, int tensorsMaxNum)
        {
            applier = new GPUTensorProcesserApplier<Int4>(device, null, TensorSummationShaders.TensorCopy1, TensorSummationShaders.TensorCopy2, Int4.Size, () =>
            {
                copyInput.SetUAV(0);
                copyOutput.SetUAV(1);
            });

            copyOutput = new Tensor(device, tensorMaxLength);

            shaderInput = new Tensor(device, tensorMaxLength * tensorsMaxNum);
            shaderOutput = new Tensor(device, (int)Math.Ceiling(tensorMaxLength * tensorsMaxNum / 2f));

            this.device = device;
            info = device.CreateBuffer(new Int8(), Int8.Size);
            shader = device.CreateComputeShader(TensorSummationShaders.BatchSummation);
        }

        private void ProcessInputs(int start, int count, Tensor[] inputs)
        {
            for (int i = 0; i < count; i++)
            {
                Tensor t = inputs[i + start];

                if (t.Dimensions.Length == 1)
                {
                    t.CopyTo(shaderInput, new TensorBox(0, t.TensorLength), i * t.TensorLength);
                }
                else
                {
                    copyInput = t;

                    applier.Run(new Int4()
                    {
                        int1 = t.Dimensions[0],
                        int2 = t.Dimensions.Length >= 2 ? t.Dimensions[1] : 0,
                        int3 = t.Dimensions.Length >= 3 ? t.Dimensions[2] : 0,
                    }, t.Dimensions);

                    copyOutput.CopyTo(shaderInput, new TensorBox(0, t.TensorLength), i * t.TensorLength);
                }
            }
        }
        private void RunSum(int tensorSize, int groupSize, int sumLayers, int currentBatchSize)
        {
            int inputSize = tensorSize;
            int outputSize = (int)Math.Ceiling(inputSize / (float)groupSize);

            device.SetComputeShader(shader);
            shaderInput.SetUAV(0);
            shaderOutput.SetUAV(1);
            device.SetBuffer(info, 0);

            for (int j = 0; j < sumLayers; j++)
            {
                Int8 info = new Int8()
                {
                    int1 = inputSize,
                    int2 = outputSize,
                    int3 = groupSize,
                    int4 = currentBatchSize,
                    int5 = inputSize % groupSize == 0 ? 0 : groupSize - (inputSize % groupSize),
                };

                this.info.UpdateBuffer(info);

                device.Dispatch((int)Math.Ceiling(outputSize * currentBatchSize / 16f), 1, 1);

                if (j != sumLayers - 1)
                {
                    shaderOutput.CopyTo(shaderInput, new TensorBox(0, shaderOutput.TensorLength), 0);
                    inputSize = outputSize;
                    outputSize = (int)Math.Ceiling(inputSize / (float)groupSize);
                }
            }
        }
        /// <summary>
        /// Returns the sum of data in <paramref name="inputs"/>.
        /// </summary>
        /// <param name="groupSize">The value to sum every iteration in the gpu. if the length of a single tensor in <paramref name="inputs"/> is less than this, it will be adjusted automatically.</param>
        /// <param name="batchSize">The number of tensors to process at once in the gpu. if the number of tensors in <paramref name="inputs"/> is less than this, it will be adjusted automatically.</param>
        /// <param name="seperate">Whether to return every tensor's sum seperately in the sum array (returned tensor) or to sum all the tensors to one value. This is always true when <paramref name="result"/> != null</param>
        /// <param name="result">This is a gpu tensor used to save the results to it (to avoid creating a new gpu tensor). If null, this function will create a cpu result tensor.</param>
        /// <param name="resultOffset">The index to start copy results to (in <paramref name="result"/>).</param>
        /// <param name="inputs">The input data. Note that the tensor's length must be equal.</param>
        /// <returns></returns>
        public Tensor Sum(int groupSize = 8, int batchSize = 10, bool seperate = true, Tensor result = null, int resultOffset = 0, params Tensor[] inputs) 
        {
            batchSize = Math.Min(inputs.Length, batchSize);

            int size = inputs[0].TensorLength;

            if (result == null)
            {
                result = new Tensor(null, inputs.Length);
                resultOffset = 0;
            }

            if (size == 1)
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    inputs[i].CopyTo(result, new TensorBox(0, 1), i + resultOffset);
                }

                return result;
            }

            groupSize = Math.Min(size, groupSize);

            for (int i = 1; i < inputs.Length; i++)
            {
                if (inputs[i].TensorLength != size)
                    throw new ArgumentException("The input tensors must have the same length!");
            }

            int batches = (int)Math.Ceiling(inputs.Length / (float)batchSize);
            int sumLayers = (int)Math.Ceiling(Math.Log(size, groupSize));

            for (int i = 0; i < batches; i++)
            {
                int currentBatchSize = Math.Min(inputs.Length - i * batchSize, batchSize); //this is for in case the total number of inputs is not divisible by the batch size, so the last batch will have a smaller size.
                ProcessInputs(i * batchSize, currentBatchSize, inputs);

                RunSum(size, groupSize, sumLayers, currentBatchSize);

                shaderOutput.CopyTo(result, new TensorBox(0, currentBatchSize), resultOffset + i * batchSize);
            }

            if (seperate || result != null)
                return result;

            Tensor total = new Tensor(null, 1);
            float sum = 0;

            for (int i = 0; i < result.TensorLength; i++)
            {
                sum += result[i];
            }

            total[0] = sum;
            return total;
        }

        private void ProcessInputsL2(int start, int count, Tensor[] inputs)
        {
            for (int i = 0; i < count; i++)
            {
                Tensor t = inputs[i + start];

                copyInput = t;

                l2Applier.Run(new Int4()
                {
                    int1 = t.Dimensions[0],
                    int2 = t.Dimensions.Length >= 2 ? t.Dimensions[1] : 0,
                    int3 = t.Dimensions.Length >= 3 ? t.Dimensions[2] : 0,
                }, t.Dimensions);

                copyOutput.CopyTo(shaderInput, new TensorBox(0, t.TensorLength), i * t.TensorLength);
            }
        }
        /// <summary>
        /// Returns the squared l2 norm of data in <paramref name="inputs"/>.
        /// </summary>
        /// <param name="groupSize">The value to sum every iteration in the gpu. if the length of a single tensor in <paramref name="inputs"/> is less than this, it will be adjusted automatically.</param>
        /// <param name="batchSize">The number of tensors to process at once in the gpu. if the number of tensors in <paramref name="inputs"/> is less than this, it will be adjusted automatically.</param>
        /// <param name="result">This is a gpu tensor used to save the results to it (to avoid creating a new gpu tensor). If null, this function will create a cpu result tensor.</param>
        /// <param name="resultOffset">The index to start copy results to (in <paramref name="result"/>).</param>
        /// <param name="inputs">The input data. Note that the tensor's length must be equal.</param>
        /// <returns></returns>
        public Tensor L2NormSqr(int groupSize = 8, int batchSize = 10, Tensor result = null, int resultOffset = 0, params Tensor[] inputs)
        {
            if (l2Applier == null)
            {
                l2Applier = new GPUTensorProcesserApplier<Int4>(device, TensorSummationShaders.L2Norm1, TensorSummationShaders.L2Norm2, TensorSummationShaders.L2Norm3, Int4.Size, () =>
                {
                    copyInput.SetUAV(0);
                    copyOutput.SetUAV(1);
                });
            }

            batchSize = Math.Min(inputs.Length, batchSize);

            int size = inputs[0].TensorLength;

            if (result == null)
            {
                result = new Tensor(null, inputs.Length);
                resultOffset = 0;
            }

            if (size == 1)
            {
                int b = (int)Math.Ceiling(inputs.Length / (float)batchSize);

                for (int i = 0; i < b; i++)
                {
                    int currentBatchSize = Math.Min(inputs.Length - i * batchSize, batchSize); //this is for in case the total number of inputs is not divisible by the batch size, so the last batch will have a smaller size.
                    ProcessInputsL2(i * batchSize, currentBatchSize, inputs);

                    shaderInput.CopyTo(result, new TensorBox(0, currentBatchSize), i * batchSize);
                }

                return result;
            }

            groupSize = Math.Min(size, groupSize);

            for (int i = 1; i < inputs.Length; i++)
            {
                if (inputs[i].TensorLength != size)
                    throw new ArgumentException("The input tensors must have the same length!");
            }

            int batches = (int)Math.Ceiling(inputs.Length / (float)batchSize);
            double l = Math.Log(size, groupSize);
            int sumLayers = (int)Math.Ceiling(double.IsNaN(l) ? 1 : l);

            for (int i = 0; i < batches; i++)
            {
                int currentBatchSize = Math.Min(inputs.Length - i * batchSize, batchSize); //this is for in case the total number of inputs is not divisible by the batch size, so the last batch will have a smaller size.
                ProcessInputsL2(i * batchSize, currentBatchSize, inputs);

                RunSum(size, groupSize, sumLayers, currentBatchSize);

                shaderOutput.CopyTo(result, new TensorBox(0, currentBatchSize), i * batchSize);
            }

            float sum = 0;
            for (int i = 0; i < result.TensorLength; i++)
            {
                sum += result[i];
            }

            Tensor t = new Tensor(null, 1);
            t[0] = sum;

            return t;
        }
        /// <summary>
        /// Get the sum of <see cref="Input"/>
        /// </summary>
        /// <param name="TensorLength">The length of a single tensor.</param>
        /// <param name="TensorsNum">The number of tensors.</param>
        /// <param name="result">The result tensor to save results to. Its length must be >= <paramref name="TensorsNum"/>.</param>
        /// <param name="resultOffset">The index to start copy results to (in <paramref name="result"/>).</param>
        /// <param name="groupSize">The value to sum every iteration in the gpu. if <paramref name="TensorLength"/> is less than this, it will be adjusted automatically.</param>
        /// <returns></returns>
        public Tensor RunFromInput(int TensorLength, int TensorsNum, Tensor result, int resultOffset, int groupSize = 8)
        {
            if (TensorLength == 1)
            {
                shaderInput.CopyTo(result, new TensorBox(0, TensorsNum), resultOffset);
                return result;
            }

            groupSize = Math.Min(TensorLength, groupSize);

            int sumLayers = (int)Math.Ceiling(Math.Log(TensorLength, groupSize));
            RunSum(TensorLength, groupSize, sumLayers, TensorsNum);

            shaderOutput.CopyTo(result, new TensorBox(0, TensorsNum), resultOffset);

            return result;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            applier.Dispose();
            copyOutput.Dispose();
            shaderInput.Dispose();
            shaderOutput.Dispose();
        }
    }
}

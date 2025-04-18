using ComputeShaders;
using Neuran.IDX;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran
{
    /// <summary>
    /// An IDataIterator class that reads data from an IDX file.
    /// Note: the length of the input and output arrays output at GetNext() are 1.
    /// </summary>
    public class IDXDataIterator : IDataIterator
    {
        /// <inheritdoc/>
        public int Length { get => inputReader.DimensionsSize[0]; }

        /// <inheritdoc/>
        public int SequenceLength { get => 1; }
        /// <inheritdoc/>
        public bool LoopSequence { get; set; }
        /// <inheritdoc/>
        public SequenceType DataType { get; private set; }

        /// <inheritdoc/>
        public int OutputLength { get => 1; }

        int[] permutations;
        int currentIndex;

        Tensor input, output;
        IDXReader inputReader, outputReader;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputPath"></param>
        /// <param name="outputPath"></param>
        /// <param name="device"></param>
        public IDXDataIterator(string inputPath, string outputPath, SequenceType dataType, CSDevice device = null)
        {
            DataType = dataType;

            inputReader = new IDXReader(inputPath);
            outputReader = new IDXReader(outputPath);

            int[] inputDims = new int[inputReader.DimensionsNumber - 1];
            int[] outputDims = new int[outputReader.DimensionsNumber - 1];

            for (int i = 0; i < inputDims.Length; i++)
                inputDims[i] = inputReader.DimensionsSize[inputReader.DimensionsNumber - 1 - i];

            for (int i = 0; i < outputDims.Length; i++)
                outputDims[i] = outputReader.DimensionsSize[outputReader.DimensionsNumber - 1 - i];

            if (device != null)
            {
                input = new Tensor(device, inputDims);
                output = new Tensor(device, outputDims);
            }
            else
            {
                input = new Tensor(null, inputDims);
                output = new Tensor(null, outputDims);
            }

            permutations = Enumerable.Range(0, Length).ToArray();
        }

        /// <inheritdoc/>
        public (Tensor[], Tensor[]) GetNext()
        {
            int iStart = permutations[currentIndex] * input.TensorLength * sizeof(float);
            int iCount = input.TensorLength * sizeof(float);
            inputReader.ReadRawData(ptr => input.UpdateRawData(ptr), iStart, iCount);

            int oStart = permutations[currentIndex] * output.TensorLength * sizeof(float);
            int oCount = output.TensorLength * sizeof(float);
            outputReader.ReadRawData(ptr => output.UpdateRawData(ptr), oStart, oCount);

            if (!LoopSequence)
                currentIndex++;

            if (currentIndex >= Length)
                currentIndex = 0;

            return (new Tensor[] { input }, new Tensor[] { output });
        }

        /// <inheritdoc/>
        public void Shuffle(Random random)
        {
            permutations = permutations.OrderBy(x => random.NextDouble()).ToArray();
        }

        /// <inheritdoc/>
        public void ResetSequence()
        {
            currentIndex = 0;
        }

        /// <inheritdoc/>
        public void ResetData()
        {
            currentIndex = 0;
        }
    }
}

using ComputeShaders;
using Neuran.IDX;
using Neuran.Utilities;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.MNIST
{
    public class MNISTDataIterator : IDataIterator
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

        //gpu 
        CSDevice device;
        ComputeShader shader;
        CSTexture3D inputBuffer;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputPath"></param>
        /// <param name="outputPath"></param>
        /// <param name="device"></param>
        public MNISTDataIterator(string inputPath, string outputPath, SequenceType dataType, CSDevice device)
        {
            DataType = dataType;
            this.device = device;
            shader = device.CreateComputeShader(MNISTShaders.ByteConverter);
            inputBuffer = device.CreateTexture3D(28, 28, 1, TextureFormat.R8_UNorm);

            inputReader = new IDXReader(inputPath);
            outputReader = new IDXReader(outputPath);

            int[] inputDims = new int[]
            {
                28,
                28,
                1,
            };
            int[] outputDims = new int[]
            {
                10,
            };

            input = new Tensor(device, inputDims);
            output = new Tensor(null, outputDims);

            permutations = Enumerable.Range(0, Length).ToArray();
        }
        public MNISTDataIterator(Stream input_stream, Stream label_stream, SequenceType dataType, CSDevice device)
        {
            DataType = dataType;
            this.device = device;
            shader = device.CreateComputeShader(MNISTShaders.ByteConverter);
            inputBuffer = device.CreateTexture3D(28, 28, 1, TextureFormat.R8_UNorm);

            inputReader = new IDXReader(input_stream);
            outputReader = new IDXReader(label_stream);

            int[] inputDims = new int[]
            {
                28,
                28,
                1,
            };
            int[] outputDims = new int[]
            {
                10,
            };

            input = new Tensor(device, inputDims);
            output = new Tensor(null, outputDims);

            permutations = Enumerable.Range(0, Length).ToArray();
        }

        /// <inheritdoc/>
        public unsafe (Tensor[], Tensor[]) GetNext()
        {
            int iStart = permutations[currentIndex] * input.TensorLength * sizeof(byte);
            int iCount = input.TensorLength * sizeof(byte);

            int oStart = permutations[currentIndex] * sizeof(byte);
            int oCount = sizeof(byte);

            inputBuffer.Dispose();
            inputReader.ReadRawData(data => inputBuffer = device.CreateTexture3D(input.Dimensions[0], input.Dimensions[1], input.Dimensions[2], TextureFormat.R8_UNorm, data), iStart, iCount);

            device.SetComputeShader(shader);

            device.SetUnorderedAccessView(inputBuffer, 0);
            input.SetUAV(1);
            device.Dispatch((int)Math.Ceiling(inputBuffer.Width / 8f), (int)Math.Ceiling(inputBuffer.Height / 8f), inputBuffer.Depth);

            outputReader.ReadRawData(data =>
            {
                byte index = *(byte*)data;
                output.Zero();
                output[index] = 1;
            }, oStart, oCount);

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

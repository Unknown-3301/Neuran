using ComputeShaders;
using Microsoft.SqlServer.Server;
using Neuran.DSS;
using System;
using System.Collections.Generic;
using System.Data.Common;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran
{
    /// <summary>
    /// An IDataIterator class that reads data from an DSS file.
    /// Note: the length of the input and output arrays output at GetNext() are 1.
    /// </summary>
    public class DSSDataIterator : IDataIterator
    {
        /// <inheritdoc/>
        public int Length { get => inputReader.NumberOfSequences; }

        /// <inheritdoc/>
        public int SequenceLength 
        { 
            get 
            { 
                if (DataType == SequenceType.ManyToMany)
                    return inputReader.SequenceLength;
                else
                    return inputReader.SequenceLength + outputReader.SequenceLength - 1;
            } 
        }
        /// <inheritdoc/>
        public bool LoopSequence { get; set; }

        /// <inheritdoc/>
        public SequenceType DataType { get; private set; }

        /// <inheritdoc/>
        public int OutputLength { get => outputReader.SequenceLength; }

        private int[] permutations;
        private int currentIndex;

        private DSSReader inputReader;
        private DSSReader outputReader;

        private Tensor input;
        private Tensor output;

        //really stupid way ¯\_(ツ)_/¯
        private bool gotoSInput; 
        private bool gotoSOutput;
        private bool finishedInput;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputPath"></param>
        /// <param name="outputPath"></param>
        /// <param name="device"></param>
        public DSSDataIterator(string inputPath, string outputPath, SequenceType dataType, CSDevice input_device = null, CSDevice output_device = null)
        {
            DataType = dataType;

            inputReader = new DSSReader(inputPath, true);
            outputReader = new DSSReader(outputPath, true);

            inputReader.OnSequenceEnd = () => { gotoSInput = true; finishedInput = true; };
            outputReader.OnSequenceEnd = () => gotoSOutput = true;

            input = new Tensor(input_device, inputReader.Dimensions);
            output = new Tensor(output_device, outputReader.Dimensions);

            permutations = Enumerable.Range(0, Length).ToArray();
        }

        private unsafe Tensor GetNextInput()
        {
            byte[] ib = inputReader.NextElement();

            fixed (byte* p = ib)
            {
                IntPtr ptr = (IntPtr)p;
                input.UpdateRawData(ptr);
            }

            return input;
        }
        private unsafe Tensor GetNextOutput()
        {
            byte[] ob = outputReader.NextElement();

            fixed (byte* p = ob)
            {
                IntPtr ptr = (IntPtr)p;
                output.UpdateRawData(ptr);
            }

            return output;
        }

        /// <inheritdoc/>
        public unsafe (Tensor[], Tensor[]) GetNext()
        {
            Tensor input = null;
            Tensor output = null;


            if (DataType == SequenceType.ManyToMany || DataType == SequenceType.ManyToOne || (DataType == SequenceType.OneToMany && outputReader.ElementIndex == 0) || (DataType == SequenceType.DelayedManyToMany && !finishedInput))
            {
                input = GetNextInput();
            }
            //checking for output before input is important because of 'finishedInput' variable
            if (DataType == SequenceType.ManyToMany || DataType == SequenceType.OneToMany || (DataType == SequenceType.ManyToOne && finishedInput) || (DataType == SequenceType.DelayedManyToMany && finishedInput))
            {
                output = GetNextOutput();
            }

            if (gotoSInput && gotoSOutput)
            {
                if (!LoopSequence)
                    currentIndex++;

                if (currentIndex >= Length)
                    currentIndex = 0;

                inputReader.GoToSequence(permutations[currentIndex]);
                outputReader.GoToSequence(permutations[currentIndex]);

                gotoSInput = false;
                gotoSOutput = false;
                finishedInput = false;
            }

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
            inputReader.GoToSequence(permutations[currentIndex]);
            outputReader.GoToSequence(permutations[currentIndex]);
        }

        /// <inheritdoc/>
        public void ResetData()
        {
            inputReader.GoToSequence(permutations[0]);
            outputReader.GoToSequence(permutations[0]);
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            inputReader.Dispose();
            outputReader.Dispose();

            input.Dispose();
            output.Dispose();
        }
    }
}

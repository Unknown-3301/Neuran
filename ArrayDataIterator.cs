using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran
{
    /// <summary>
    /// A IDataIterator that stores the data in an array.
    /// </summary>
    public class ArrayDataIterator : IDataIterator
    {
        /// <inheritdoc/>
        public int Length { get => data.Count; }

        /// <inheritdoc/>
        public int SequenceLength { get => data[sequenceIndex].Count; }

        /// <inheritdoc/>
        public bool LoopSequence { get; set; }

        /// <inheritdoc/>
        public SequenceType DataType { get; private set; }

        /// <inheritdoc/>
        public int OutputLength { get; private set; }

        private List<List<(Tensor[], Tensor[])>> data;

        private int dataIndex, sequenceIndex;

        /// <summary>
        /// Creates a new empty instance.
        /// </summary>
        public ArrayDataIterator(SequenceType dataType)
        {
            DataType = dataType;
            data = new List<List<(Tensor[], Tensor[])>>() { new List<(Tensor[], Tensor[])>() };
        }

        /// <summary>
        /// Add new data element.
        /// </summary>
        /// <param name="input">The input data. This can be null depending on <see cref="SequenceType"/>.</param>
        /// <param name="output">The output data. This can be null depending on <see cref="SequenceType"/>.</param>
        /// <param name="newSequence">Whether to add the data to a new sequence or to the last added sequence previously. For models that do not use sequence data, this should be true.</param>
        public void AddData(Tensor[] input, Tensor[] output, bool newSequence)
        {
            if (newSequence && data[data.Count - 1].Count != 0)
            {
                if (data.Count == 1)
                    OutputLength = data[0].Count(x => x.Item2 != null);

                data.Add(new List<(Tensor[], Tensor[])>());
            }

            data[data.Count - 1].Add((input, output));
        }

        /// <inheritdoc/>
        public (Tensor[], Tensor[]) GetNext()
        {
            (Tensor[] input, Tensor[] output) = data[sequenceIndex][dataIndex];

            dataIndex++;
            if (dataIndex >= data[sequenceIndex].Count)
            {
                if (!LoopSequence)
                    sequenceIndex++;

                dataIndex = 0;

                if (sequenceIndex >= data.Count)
                    sequenceIndex = 0;

                OutputLength = data[sequenceIndex].Count(x => x.Item2 != null);
            }

            return (input, output);
        }

        /// <inheritdoc/>
        public void Shuffle(Random random) => data = data.OrderBy(x => random.NextDouble()).ToList();

        /// <inheritdoc/>
        public void ResetSequence()
        {
            dataIndex = 0;
        }

        /// <inheritdoc/>
        public void ResetData()
        {
            sequenceIndex = 0;
            dataIndex = 0;
        }
    }
}

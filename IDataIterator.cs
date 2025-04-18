using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran
{
    /// <summary>
    /// The interface for structures that store data to train/test models.
    /// </summary>
    public interface IDataIterator
    {
        /* DATA STRUCTURE
        
            The data in the iterator is stored as sequences, this is for sequence-based models.
            The data structure is as follows:
            0 - data 0 (input and output), sequence 0
            1 - data 1, sequence 0
            2 - data 2, sequence 0
                    .
                    .
                    .
            n - data n-1, sequence 0
            n+1 - data 0, sequence 1
            n+2 - data 1, sequence 1
                    .
                    .
                    .
            When calling Shuffle() The order of sequences is to be shuffled NOT the order of data inside each sequence.
            For models that dont use sequence data, every data element can be considered a single sequence, so SequenceLength = 1:
            0 - data 0, sequence 0 (so one data in every sequence)
            1 - data 0, sequence 1
            2 - data 0, sequence 2
                    .
                    .
                    .
         */

        /// <summary>
        /// Whether to loop through a sequence when reached the end of the sequence.
        /// </summary>
        bool LoopSequence { get; set; }
        /// <summary>
        /// The total number of sequence samples.
        /// </summary>
        int Length { get; }

        /// <summary>
        /// The length of the current sequence. This is used for RNNs or other similar models. For all other models where data order is not important, this should be set to 1.
        /// </summary>
        int SequenceLength { get; }
        /// <summary>
        /// The number of outputs in the current sequence. This is used for RNNs or other similar models. For all other models where data order is not important, this should be set to 1.
        /// </summary>
        int OutputLength { get; }

        ///// <summary>
        ///// The number of inputs in current sequence. This is only used for <see cref="SequenceType.DelayedManyToMany"/>.
        ///// </summary>
        //int InputSequenceLength { get; }
        ///// <summary>
        ///// The number of outputs in current sequence. This is only used for <see cref="SequenceType.DelayedManyToMany"/>.
        ///// </summary>
        //int OutputSequenceLength { get; }

        /// <summary>
        /// The type of all sequences stored.
        /// </summary>
        SequenceType DataType { get; }

        /// <summary>
        /// Returns the next data sample. Depending on <see cref="SequenceType"/> either tensors could be null.
        /// </summary>
        /// <returns></returns>
        (Tensor[], Tensor[]) GetNext();

        /// <summary>
        /// Shuffles the data samples' order.
        /// </summary>
        void Shuffle(Random random);

        /// <summary>
        /// Resets the element counter to 0 to start reading from the beggining of the current sequence.
        /// </summary>
        void ResetSequence();
        /// <summary>
        /// Resets the sequence and element counter to 0 to start reading from the first sequence
        /// </summary>
        void ResetData();
    }
}

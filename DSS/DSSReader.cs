using Neuran.IDX;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.DSS
{
    public class DSSReader : IDisposable
    {
        /// <summary>
        /// The type of the data stored.
        /// </summary>
        public DSSDataTypes DataType { get; }
        /// <summary>
        /// The size of each dimension. The last dimension in the array is the fastest (fastest to change when iterating the whole data), for example
        /// if there were 3 dimensions (array of images) they will be sorted as: 0-numberOfImages 1-Heights 2-Widths (if row major), see https://agilescientific.com/blog/2018/12/28/what-is-the-fastest-axis-of-an-array for explanation.
        /// </summary>
        public int[] Dimensions { get; }

        /// <summary>
        /// The size of each element in each sequence in bytes.
        /// </summary>
        public int ElementSize {  get; }
        /// <summary>
        /// The number of sequences stored in the file.
        /// </summary>
        public int NumberOfSequences { get; }

        /// <summary>
        /// The length of the current sequence (in elements).
        /// </summary>
        public int SequenceLength { get; private set; }
        /// <summary>
        /// The index of the currect element (the element that yet to be read) in the sequence.
        /// </summary>
        public int ElementIndex { get; private set; }

        /// <summary>
        /// This is called after reaching the end of a sequence when calling <see cref="NextElement"/>
        /// </summary>
        public Action OnSequenceEnd { get; set; }

        BinaryReader reader;

        private long[] sequencesPositions;

        /// <summary>
        /// Creates a new .dss file reader
        /// </summary>
        /// <param name="dssPath">The path to the .dss file.</param>
        /// <param name="gotoSequence">Whether the ability to jump to a certain sequence is possible. If true then <see cref="GoToSequence"/> can be called.</param>
        public DSSReader(string dssPath, bool gotoSequence = false)
        {
            reader = new BinaryReader(new FileStream(dssPath, FileMode.Open));

            byte[] magic = reader.ReadBytes(sizeof(int));

            DataType = DSSExtensions.GetDSSTypeFromMagicByte(magic[2]);

            Dimensions = new int[magic[3]];
            int elemSize = DataType.Size();
            for (int i = 0; i < Dimensions.Length; i++)
            {
                Dimensions[i] = reader.ReadInt32();
                elemSize *= Dimensions[i];
            }
            ElementSize = elemSize;

            NumberOfSequences = reader.ReadInt32();

            if (gotoSequence)
            {
                long oldPos = reader.BaseStream.Position;

                long currectPos = reader.BaseStream.Position;
                sequencesPositions = new long[NumberOfSequences];

                for (int i = 0; i < NumberOfSequences; i++)
                {
                    //+sizeof(int) to skip the sequence size bytes (int32)
                    sequencesPositions[i] = currectPos + sizeof(int);

                    reader.BaseStream.Position = currectPos;
                    int sLength = reader.ReadInt32();
                    currectPos = currectPos + sizeof(int) + sLength * ElementSize; //go to the next sequence position
                }

                reader.BaseStream.Position = oldPos;
            }
            
            SequenceLength = reader.ReadInt32(); //reads the current sequence's length (number of elements)
        }

        /// <summary>
        /// Returns the bytes of the next element in the sequence. If it reached the end of the sequence, it will move to the next sequence.
        /// </summary>
        /// <returns></returns>
        public byte[] NextElement()
        {
            if (reader.BaseStream.Position == reader.BaseStream.Length)
                return null;

            byte[] data = reader.ReadBytes(ElementSize);

            ElementIndex++;
            if (ElementIndex >= SequenceLength)
            {
                OnSequenceEnd?.Invoke();

                if (reader.BaseStream.Position != reader.BaseStream.Length) //this condition was added to avoid reading out of range
                {
                    SequenceLength = reader.ReadInt32();
                    ElementIndex = 0;
                }
            }

            return data;
        }

        /// <summary>
        /// Moves the reader seek to the wanted sequence.
        /// </summary>
        /// <param name="sequenceIndex"></param>
        public void GoToSequence(int sequenceIndex) => reader.BaseStream.Position = sequencesPositions[sequenceIndex];

        /// <inheritdoc/>
        public void Dispose()
        {
            reader.Close();
            reader.Dispose();
        }
    }
}

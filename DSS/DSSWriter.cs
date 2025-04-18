using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Printing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.DSS
{
    /// <summary>
    /// DSS (Dynamic Sequence Structure) is used for storing sequence data.
    /// </summary>
    public class DSSWriter : IDisposable
    {
        /* ________________DATA_STRUCTURE_________________
         The bytes in an .dss file are structured for a n-dimensional data as follows:
            - magic number (1st and 2nd bytes are 0, 3rd byte for data type (float, int, ...), MSB first, 4th byte for number of dimensions (in this case n))
            - dimension 0 (int32)
            - dimension 1 (int32)
                    .
                    .
                    .
            - dimension n-1 (int32)
            - number of sequences
            - sequence 0 (first sequence) length (int32) (for example k)
            - data 0 (size = dimesion 0 * dimension 1 *...* dimension n * size of data type)
            - data 1
                .
                .
                .
            - data k-1 (this is the end of sequence 0)
            - sequence 1 length
            - data 0
                .
                .
                .
         * */

        /// <summary>
        /// The type of the data stored.
        /// </summary>
        public DSSDataTypes DataType { get; }
        /// <summary>
        /// The size of each dimension. The last dimension in the array is the fastest (fastest to change when iterating the whole data), for example
        /// if there were 3 dimensions (array of images) they will be sorted as: 0-numberOfImages 1-Heights 2-Widths (if row major), see https://agilescientific.com/blog/2018/12/28/what-is-the-fastest-axis-of-an-array for explanation.
        /// </summary>
        public int[] DimensionsSize { get; }
        /// <summary>
        /// The number of sequences.
        /// </summary>
        public int NumberOfSequences { get; private set; }

        FileStream stream;

        private long currectSequenceLengthIndex;
        private int writtenSequences;
        private int currectSequenceLength;

        /// <summary>
        /// Creates a new dss file and writes in dss information.
        /// </summary>
        /// <param name="path">The path to save the new dss file in it.</param>
        /// <param name="dataType">The type of data Stored.</param>
        /// <param name="sequencesNum">The number of sequences to store.</param>
        /// <param name="dimensionsSize">The size of each dimension. The length of the array represents the number of dimensions.</param>
        public DSSWriter(string path, DSSDataTypes dataType, int sequencesNum, params int[] dimensionsSize)
        {
            if (dimensionsSize == null)
            {
                throw new ArgumentException("dimensionsSize cannot be null nor it's size can be 0 or higher than 255");
            }
            if (dimensionsSize.Length == 0 || dimensionsSize.Length > 255)
            {
                throw new ArgumentException("dimensionsSize cannot be null nor it's size can be 0 or higher than 255");
            }

            stream = new FileStream(path, FileMode.OpenOrCreate, FileAccess.Write);

            DimensionsSize = new int[dimensionsSize.Length];

            byte[] dssInfo = new byte[4 + dimensionsSize.Length * 4];
            dssInfo[2] = dataType.DSSMagicByte();
            dssInfo[3] = (byte)dimensionsSize.Length;

            unsafe
            {
                fixed (int* d0 = &dimensionsSize[0])
                {
                    int* d = d0;

                    for (int i = 0; i < dimensionsSize.Length; i++)
                    {
                        DimensionsSize[i] = dimensionsSize[i];

                        byte* p = (byte*)d;

                        for (int i2 = 0; i2 < sizeof(int); i2++)
                        {
                            dssInfo[4 + i * sizeof(int) + i2] = *p;

                            p++;
                        }

                        d += 1;
                    }
                }

            }

            stream.Write(dssInfo, 0, dssInfo.Length);

            stream.Write(BitConverter.GetBytes(sequencesNum), 0, sizeof(int));
            NumberOfSequences = sequencesNum;

            currectSequenceLengthIndex = stream.Position;
            stream.Write(new byte[4], 0, 4); //to skip the 4 bytes (int32) that represents sequence 0 size.
        }

        /// <summary>
        /// Writes to the dss file without overwriting what is written already (adds to what was written before) the bytes from <paramref name="bytes"/>.
        /// </summary>
        /// <param name="bytes">The bytes array.</param>
        /// <param name="offset">the offset in <paramref name="bytes"/> to starts writing from. So if it's 1 i will copy the bytes starting from the byte 
        /// with index 1 in the bytes array to the dss file, so the first byte in the byte array is skipped.</param>
        /// <param name="count">The number of bytes to copy from <paramref name="bytes"/>.</param>
        /// <param name="newSequence">Whether to add this data to a new sequence.</param>
        public unsafe void AppendWrite(byte[] bytes, int offset, int count, bool newSequence)
        {
            if (newSequence)
            {
                currectSequenceLength = 1;
                currectSequenceLengthIndex = stream.Position;
                stream.Write(new byte[4], 0, 4); //just to move the position after the size bytes to start writing the data and not override the size bytes.

                writtenSequences++;

                if (writtenSequences >= NumberOfSequences)
                {
                    ChangesSequencesNum(NumberOfSequences + 1);
                }
            }
            else
            {
                currectSequenceLength++;
            }

            ChangeSequenceLength(currectSequenceLength);
            stream.Write(bytes, offset, count);
        }

        private unsafe void ChangeSequenceLength(int newLength)
        {
            long oldPos = stream.Position;

            stream.Position = currectSequenceLengthIndex;

            stream.Write(BitConverter.GetBytes(newLength), 0, 4);
        
            stream.Position = oldPos;
        }

        /// <summary>
        /// Changes the size of the dimension with the index <paramref name="dimensionIndex"/> in <see cref="DimensionsSize"/>.
        /// </summary>
        /// <param name="dimensionIndex">The index of the dimension.</param>
        /// <param name="newSize">The new Size of the dimension</param>
        public void ChangeDimensionSize(int dimensionIndex, int newSize)
        {
            long currentPos = stream.Position;

            stream.Position = 4 + dimensionIndex * sizeof(int);
            DimensionsSize[dimensionIndex] = newSize;

            stream.Write(BitConverter.GetBytes(newSize), 0, sizeof(int));

            stream.Position = currentPos;
        }
        /// <summary>
        /// Changes the number of sequences to be stored.
        /// </summary>
        /// <param name="sequenceNum">The new length.</param>
        public void ChangesSequencesNum(int sequenceNum)
        {
            long oldPos = stream.Position;
            stream.Position = 4 + DimensionsSize.Length * sizeof(int); //4 is for the magic number (4 bytes)

            stream.Write(BitConverter.GetBytes(sequenceNum), 0, sizeof(int));

            NumberOfSequences = sequenceNum;

            stream.Position = oldPos;
        }

        /// <summary>
        /// Closes the current writer and the underlying stream.
        /// </summary>
        public void Close()
        {
            stream.Close();
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            stream.Dispose();
        }
    }
}

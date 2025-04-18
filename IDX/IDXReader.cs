using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;

namespace Neuran.IDX
{
    /// <summary>
    /// A class that reads idx files. For more information about idx files see http://yann.lecun.com/exdb/mnist/
    /// </summary>
    public class IDXReader : IDisposable
    {
        /// <summary>
        /// The type of the data stored.
        /// </summary>
        public IDXDataTypes DataType { get; }
        /// <summary>
        /// The number of dimensions in the idx file
        /// </summary>
        public int DimensionsNumber { get; }
        /// <summary>
        /// The size of each dimension. The last dimension in the array is the fastest (fastest to change when iterating the whole data), for example
        /// if there were 3 dimensions (array of images) they will be sorted as: 0-numberOfImages 1-Heights 2-Widths (if row major), see https://agilescientific.com/blog/2018/12/28/what-is-the-fastest-axis-of-an-array for explanation.
        /// </summary>
        public int[] DimensionsSize { get; }

        long offset;
        BinaryReader reader;

        /// <summary>
        /// Opens an idx file.
        /// </summary>
        /// <param name="idxPath">The path to the idx file.</param>
        public IDXReader(string idxPath)
        {
            reader = new BinaryReader(new FileStream(idxPath, FileMode.Open));

            byte[] magicNumBytes = reader.ReadBytes(4);

            DataType = IDXExtensions.GetIDXTypeFromByte(magicNumBytes[2]);

            DimensionsNumber = magicNumBytes[3];
            DimensionsSize = new int[DimensionsNumber];
            for (int i = 0; i < DimensionsNumber; i++)
            {
                DimensionsSize[i] = reader.ReadIDXInt32();
            }

            offset = reader.BaseStream.Position;
        }
        /// <summary>
        /// Opens an idx file.
        /// </summary>
        /// <param name="idxStream">The stream reading the idx file.</param>
        public IDXReader(Stream idxStream)
        {
            reader = new BinaryReader(idxStream);

            byte[] magicNumBytes = reader.ReadBytes(4);

            DataType = IDXExtensions.GetIDXTypeFromByte(magicNumBytes[2]);

            DimensionsNumber = magicNumBytes[3];
            DimensionsSize = new int[DimensionsNumber];
            for (int i = 0; i < DimensionsNumber; i++)
            {
                DimensionsSize[i] = reader.ReadIDXInt32();
            }

            offset = reader.BaseStream.Position;
        }

        /// <summary>
        /// Reads the raw data in the idx file.
        /// Note: All the reading must be done inside <paramref name="readingAction"/>.
        /// </summary>
        /// <param name="readingAction">The actions that contains all the reading process. The IntPtr given is the pointer to the byte with the index <paramref name="start"/> in the idx file.</param>
        /// <param name="start">The index to the byte in the data wanted to read in the idx file. The first bytes that defines the idx file arent included because they do not contain the data themselfs.</param>
        /// <param name="count">The length of bytes wanted to read.</param>
        public unsafe void ReadRawData(Action<IntPtr> readingAction, int start, int count)
        {
            if (start < 0 || count <= 0)
            {
                throw new IndexOutOfRangeException("start or count are out of range");
            }

            reader.BaseStream.Position = offset + start;

            fixed (byte* b = reader.ReadBytes(count))
            {
                IntPtr pointer = (IntPtr)b;

                readingAction(pointer);
            }
        }


        /// <summary>
        /// Closes the current reader and the underlying stream.
        /// </summary>
        public void Close()
        {
            reader.Close();
        }
        /// <inheritdoc/>
        public void Dispose()
        {
            reader.Dispose();
        }
    }
}

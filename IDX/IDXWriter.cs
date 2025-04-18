using System.Collections;
using System.Collections.Generic;
using System;
using System.IO;

namespace Neuran.IDX
{
    /// <summary>
    /// A class that creates idx files. For more information about idx files see http://yann.lecun.com/exdb/mnist/
    /// </summary>
    public class IDXWriter : IDisposable
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

        FileStream stream;

        /// <summary>
        /// Creates a new idx file and writes in idx information.
        /// </summary>
        /// <param name="path">The path to save the new idx file in it.</param>
        /// <param name="dataType">The type of data Stored.</param>
        /// <param name="dimensionsSize">The size of each dimension. The length of the array represents the number of dimensions.</param>
        public IDXWriter(string path, IDXDataTypes dataType, params int[] dimensionsSize)
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

            byte[] idxInfo = new byte[4 + dimensionsSize.Length * 4];
            idxInfo[2] = dataType.IDXByte();
            idxInfo[3] = (byte)dimensionsSize.Length;

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
                            idxInfo[4 + i * sizeof(int) + (sizeof(int) - i2 - 1)] = *p;

                            p++;
                        }

                        d += 1;
                    }
                }

            }

            stream.Write(idxInfo, 0, idxInfo.Length);
        }

        /// <summary>
        /// Writes to the idx file without overwriting what is written already (adds to what was written before) the bytes from <paramref name="bytes"/>.
        /// </summary>
        /// <param name="bytes">The bytes array.</param>
        /// <param name="offset">the offset in <paramref name="bytes"/> to starts writing from. So if it's 1 i will copy the bytes starting from the byte 
        /// with index 1 in the bytes array to the idx file, so the first byte in the byte array is skipped.</param>
        /// <param name="count">The number of bytes to copy from <paramref name="bytes"/>.</param>
        public void AppendWrite(byte[] bytes, int offset, int count)
        {
            stream.Write(bytes, offset, count);
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

            byte[] bytes = new byte[sizeof(int)];
            unsafe
            {
                byte* p = (byte*)&newSize;

                bytes[3] = *p;
                p++;

                bytes[2] = *p;
                p++;

                bytes[1] = *p;
                p++;

                bytes[0] = *p;
            }

            stream.Write(bytes, 0, bytes.Length);

            stream.Position = currentPos;
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

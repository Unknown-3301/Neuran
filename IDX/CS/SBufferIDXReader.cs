using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ComputeShaders;
using Neuran.IDX;

namespace Neuran.IDX.CS
{
    /// <summary>
    /// A class to read idx data and convert it to <see cref="CSStructuredBuffer{T}"/>.
    /// </summary>
    public class SBufferIDXReader<T> : IDisposable where T : unmanaged
    {
        /// <summary>
        /// Number of buffers stored in the idx file.
        /// </summary>
        public int Buffers { get; }
        /// <summary>
        /// The Length of each buffer stored in the idx file.
        /// </summary>
        public int BufferLength { get; }

        private CSDevice device;
        private CSStructuredBuffer<T> buffer;
        private IDXReader reader;
        private int tSize;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="path">The path to the idx file.</param>
        /// <param name="device">Device used to create the buffer.</param>
        public unsafe SBufferIDXReader(string path, CSDevice device)
        {
            this.device = device;
            reader = new IDXReader(path);

            Buffers = reader.DimensionsSize[0];
            BufferLength = reader.DimensionsSize[1];
            tSize = sizeof(T);

            buffer = device.CreateStructuredBuffer(new T[BufferLength], sizeof(T));
        }

        /// <summary>
        /// The a texture from the idx file.
        /// </summary>
        /// <param name="index">The index of the texture.</param>
        /// <param name="readMethod">The method of reading The data from the idx file to the texture.</param>
        /// <returns></returns>
        public unsafe CSStructuredBuffer<T> ReadTexture(int index, BufferReadingMethod readMethod)
        {
            if (index >= Buffers)
                throw new ArgumentOutOfRangeException($"The buffer index ({index}) is out of range as there are only {Buffers} buffers in the idx file");

            reader.ReadRawData(p =>
            {
                switch (readMethod)
                {
                    case BufferReadingMethod.CreateNewBuffer:
                        buffer.Dispose();
                        buffer = device.CreateStructuredBuffer<T>(p, BufferLength, tSize);
                        break;

                    case BufferReadingMethod.SetData:
                        T[] ts = new T[BufferLength];

                        fixed(T* tp = &ts[0])
                        {
                            ComputeShaders.Utilities.CopyMemory((IntPtr)tp, p, (uint)(BufferLength * tSize));
                        }

                        buffer.SetData(ts);
                        break;

                    case BufferReadingMethod.WriteToRaw:
                        buffer.EnableCPU_Raw_ReadWrite();
                        buffer.AccessRawData(box =>
                        {
                            ComputeShaders.Utilities.CopyMemory(box.DataPointer, p, (uint)(BufferLength * tSize));
                        }, CPUAccessMode.Write);
                        break;
                }

            }, index * BufferLength * tSize, BufferLength * tSize);

            return buffer;
        }

        /// <summary>
        /// Closes the current reader and the underlying stream.
        /// </summary>
        public void Close() => reader.Close();

        /// <inheritdoc/>
        public void Dispose()
        {
            reader.Dispose();
            buffer.Dispose();
        }
    }
}

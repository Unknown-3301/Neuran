using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ComputeShaders;

namespace Neuran.IDX.CS
{

    /// <summary>
    /// A class to read idx data and convert it to <see cref="CSTexture2D"/>.
    /// </summary>
    public class CSTexture2DIDXReader : IDisposable
    {
        /// <summary>
        /// Number of textures stored in the idx file.
        /// </summary>
        public int Textures { get; }
        /// <summary>
        /// The height of the textures stored in the idx file.
        /// </summary>
        public int Height { get; }
        /// <summary>
        /// The width of the textures stored in the idx file.
        /// </summary>
        public int Width { get; }
        /// <summary>
        /// Number of channels (color channels) in each pixel in the textures stored in the idx file.
        /// </summary>
        public int Channels { get; }
        /// <summary>
        /// The type of each channel in the textures stored in the idx file.
        /// </summary>
        public IDXDataTypes ChannelType { get; }
        /// <summary>
        /// The format of the texture.
        /// </summary>
        public TextureFormat Format { get; }

        private CSDevice device;
        private CSTexture2D texture;
        private IDXReader reader;
        private int formatSize;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="path">The path to the idx file.</param>
        /// <param name="format">The format of the texture.</param>
        /// <param name="device">Device used to create the texture.</param>
        public CSTexture2DIDXReader(string path, TextureFormat format, CSDevice device)
        {
            this.device = device;
            reader = new IDXReader(path);

            Textures = reader.DimensionsSize[0];
            Height = reader.DimensionsSize[1];
            Width = reader.DimensionsSize[2];
            Channels = reader.DimensionsSize[3];
            ChannelType = reader.DataType;
            Format = format;
            formatSize = SharpDX.DXGI.FormatHelper.SizeOfInBytes((SharpDX.DXGI.Format)format);

            if (ChannelType.Size() * Channels != formatSize)
                throw new ArgumentException("The format size does not match the format size in the idx file!");

            texture = device.CreateTexture2D(Width, Height, format);
        }

        /// <summary>
        /// Reads a texture from the idx file.
        /// </summary>
        /// <param name="index">The index of the texture.</param>
        /// <param name="readMethod">The method of reading The data from the idx file to the texture.</param>
        /// <returns></returns>
        public CSTexture2D ReadTexture(int index, TextureReadingMethod readMethod)
        {
            if (index >= Textures)
                throw new ArgumentOutOfRangeException($"The texture index ({index}) is out of range as there are only {Textures} textures in the idx file");

            reader.ReadRawData(p =>
            {
                switch (readMethod)
                {
                    case TextureReadingMethod.CreateNewTexture:
                        texture.Dispose();
                        texture = device.CreateTexture2D(Width, Height, Format, p);
                        break;

                    case TextureReadingMethod.UpdateSubresource:
                        texture.UpdateSubresource(p);
                        break;

                    case TextureReadingMethod.WriteToRawData:
                        texture.EnableCPU_Raw_ReadWrite();
                        texture.AccessRawData(box =>
                        {
                            for (int y = 0; y < Height; y++)
                            {
                                IntPtr dst = IntPtr.Add(box.DataPointer, box.RowPitch * y);
                                IntPtr src = IntPtr.Add(p, Width * formatSize * y);

                                ComputeShaders.Utilities.CopyMemory(dst, src, (uint)(Width * formatSize));
                            }
                        }, CPUAccessMode.Write);
                        break;
                }

            }, index * Width * Height * formatSize, Width * Height * formatSize);

            return texture;
        }

        /// <summary>
        /// Closes the reader and its stream.
        /// </summary>
        public void Close() => reader.Close();

        /// <inheritdoc/>
        public void Dispose()
        {
            reader.Dispose();
            texture.Dispose();
        }
    }
}

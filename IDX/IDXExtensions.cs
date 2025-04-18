using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;


namespace Neuran.IDX
{
    /// <summary>
    /// A class that holds the functions used for idx.
    /// </summary>
    public static class IDXExtensions
    {
        /// <summary>
        /// Reads 4 bytes as an integer in idx file.
        /// source: //https://stackoverflow.com/questions/49407772/reading-mnist-database
        /// </summary>
        /// <param name="rd"></param>
        /// <returns></returns>
        public static int ReadIDXInt32(this BinaryReader rd)
        {
            return BitConverter.ToInt32(rd.ReadIDXBytes(sizeof(Int32)), 0);
        }
        /// <summary>
        /// Reads an array of bytes in idx file.
        /// </summary>
        /// <param name="rd"></param>
        /// <param name="count">The number of bytes to read.</param>
        /// <returns></returns>
        public static byte[] ReadIDXBytes(this BinaryReader rd, int count)
        {
            byte[] b = rd.ReadBytes(count);

            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(b);
            }

            return b;
        }

        /// <summary>
        /// The size of this data type.
        /// </summary>
        /// <param name="t"></param>
        /// <returns></returns>
        public static int Size(this IDXDataTypes t)
        {
            switch (t)
            {
                case IDXDataTypes.UnsignedByte:
                    return 1;
                case IDXDataTypes.SignedByte:
                    return 1;
                case IDXDataTypes.Short:
                    return 2;
                case IDXDataTypes.Int:
                    return 4;
                case IDXDataTypes.Float:
                    return 4;
                case IDXDataTypes.Double:
                    return 8;
            }

            return 0;
        }
        /// <summary>
        /// Converts the data type to the byte that represent it in idx file.
        /// </summary>
        /// <param name="dataType"></param>
        /// <returns></returns>
        public static byte IDXByte(this IDXDataTypes dataType)
        {
            switch (dataType)
            {
                case IDXDataTypes.UnsignedByte:
                    return 0x08;

                case IDXDataTypes.SignedByte:
                    return 0x09;

                case IDXDataTypes.Short:
                    return 0x0B;

                case IDXDataTypes.Int:
                    return 0x0C;

                case IDXDataTypes.Float:
                    return 0x0D;

                case IDXDataTypes.Double:
                    return 0x0E;
            }

            return 0x08;
        }
        /// <summary>
        /// Converts the data byte (the third byte in the magic number in idx file) to the data type it represent.
        /// </summary>
        /// <param name="magicNumByte3">the third byte in the magic number in idx file.</param>
        /// <returns></returns>
        public static IDXDataTypes GetIDXTypeFromByte(byte magicNumByte3)
        {
            switch (magicNumByte3)
            {
                case 0x08:
                    return IDXDataTypes.UnsignedByte;

                case 0x09:
                    return IDXDataTypes.SignedByte;

                case 0x0B:
                    return IDXDataTypes.Short;

                case 0x0C:
                    return IDXDataTypes.Int;

                case 0x0D:
                    return IDXDataTypes.Float;

                case 0x0E:
                    return IDXDataTypes.Double;
            }

            return IDXDataTypes.UnsignedByte;
        }
    }
}

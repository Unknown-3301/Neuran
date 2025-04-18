using Neuran.IDX;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.DSS
{
    public static class DSSExtensions
    {
        /// <summary>
        /// The size of this data type.
        /// </summary>
        /// <param name="t"></param>
        /// <returns></returns>
        public static int Size(this DSSDataTypes t)
        {
            switch (t)
            {
                case DSSDataTypes.UnsignedByte:
                    return 1;
                case DSSDataTypes.SignedByte:
                    return 1;
                case DSSDataTypes.Short:
                    return 2;
                case DSSDataTypes.Int:
                    return 4;
                case DSSDataTypes.Float:
                    return 4;
                case DSSDataTypes.Double:
                    return 8;
            }

            return 0;
        }
        /// <summary>
        /// Converts the data type to the byte that represent it in idx file.
        /// </summary>
        /// <param name="dataType"></param>
        /// <returns></returns>
        public static byte DSSMagicByte(this DSSDataTypes dataType)
        {
            switch (dataType)
            {
                case DSSDataTypes.UnsignedByte:
                    return 0x08;

                case DSSDataTypes.SignedByte:
                    return 0x09;

                case DSSDataTypes.Short:
                    return 0x0B;

                case DSSDataTypes.Int:
                    return 0x0C;

                case DSSDataTypes.Float:
                    return 0x0D;

                case DSSDataTypes.Double:
                    return 0x0E;
            }

            return 0x08;
        }
        /// <summary>
        /// Converts the data byte (the third byte in the magic number in idx file) to the data type it represent.
        /// </summary>
        /// <param name="magicNumByte3">the third byte in the magic number in idx file.</param>
        /// <returns></returns>
        public static DSSDataTypes GetDSSTypeFromMagicByte(byte magicNumByte3)
        {
            switch (magicNumByte3)
            {
                case 0x08:
                    return DSSDataTypes.UnsignedByte;

                case 0x09:
                    return DSSDataTypes.SignedByte;

                case 0x0B:
                    return DSSDataTypes.Short;

                case 0x0C:
                    return DSSDataTypes.Int;

                case 0x0D:
                    return DSSDataTypes.Float;

                case 0x0E:
                    return DSSDataTypes.Double;
            }

            return DSSDataTypes.UnsignedByte;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran
{
    /// <summary>
    /// A class to store tensor's data.
    /// </summary>
    public class TensorData
    {
        /// <summary>
        /// The dimensions of the data.
        /// </summary>
        public int[] Dimensions { get; set; }

        /// <summary>
        /// Tensor's data.
        /// </summary>
        public float[] Data { get; private set; }

        /// <summary>
        /// Creates a new instance
        /// </summary>
        /// <param name="dimensions">The dimensions of the data.</param>
        /// <param name="data">Tensor's data.</param>
        public TensorData(int[] dimensions, float[] data)
        {
            Dimensions = dimensions;
            Data = data;
        }
    }
}

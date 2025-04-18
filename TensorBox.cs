using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran
{
    /// <summary>
    /// A struct for enclosing data regions in a tensor up to 3 dimensions.
    /// </summary>
    public struct TensorBox
    {
        /// <summary>
        /// The x-coordinate of the starting position of the box.
        /// </summary>
        public int StartX;
        /// <summary>
        /// The y-coordinate of the starting position of the box. For 1D tesnors, this should be 0.
        /// </summary>
        public int StartY;

        /// <summary>
        /// The z-coordinate of the starting position of the box. For 1D and 2D tesnors, this should be 0.
        /// </summary>
        public int StartZ;

        /// <summary>
        /// The width of the box in elements.
        /// </summary>
        public int CountX;
        /// <summary>
        /// The height of the box in elements. For 1D tesnors, this should be 1.
        /// </summary>
        public int CountY;
        /// <summary>
        /// The depth of the box in elements. For 1D and 2D tesnors, this should be 1.
        /// </summary>
        public int CountZ;

        /// <summary>
        /// Creates a new box to enclose data in a 1D Tensor.
        /// </summary>
        /// <param name="startX">The x-coordinate of the starting position of the box.</param>
        /// <param name="countX">The width of the box in elements.</param>
        public TensorBox(int startX, int countX)
        {
            StartX = startX;
            CountX = countX;
            
            CountY = 1;
            CountZ = 1;

            StartY = 0;
            StartZ = 0;
        }
        /// <summary>
        /// Creates a new box to enclose data in a 2D Tensor.
        /// </summary>
        /// <param name="startX">The x-coordinate of the starting position of the box.</param>
        /// <param name="countX">The width of the box in elements.</param>
        /// <param name="startY">The y-coordinate of the starting position of the box.</param>
        /// <param name="countY">The height of the box in elements.</param>
        public TensorBox(int startX, int startY, int countX, int countY)
        {
            StartX = startX;
            CountX = countX;
            CountY = countY;
            StartY = startY;

            CountZ = 1;

            StartZ = 0;
        }
        /// <summary>
        /// Creates a new box to enclose data in a 2D Tensor.
        /// </summary>
        /// <param name="startX">The x-coordinate of the starting position of the box.</param>
        /// <param name="countX">The width of the box in elements.</param>
        /// <param name="startY">The y-coordinate of the starting position of the box.</param>
        /// <param name="countY">The height of the box in elements.</param>
        /// <param name="startZ">The z-coordinate of the starting position of the box.</param>
        /// <param name="countZ">The depth of the box in elements.</param>
        public TensorBox(int startX, int startY, int startZ, int countX, int countY, int countZ)
        {
            StartX = startX;
            CountX = countX;
            CountY = countY;
            StartY = startY;
            StartZ = startZ;
            CountZ = countZ;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran
{
    /// <summary>
    /// The types of sequence data.
    /// </summary>
    public enum SequenceType
    {
        /// <summary>
        /// Every input correspond to a output.
        /// </summary>
        ManyToMany,
        /// <summary>
        /// This is used when we run the model with all the inputs, after that we take output from the model.
        /// </summary>
        DelayedManyToMany,
        /// <summary>
        /// All the inputs correspond to a single output. This in a sequence means that after inputting running all input data in a sequence the last output from the model is considered.
        /// </summary>
        ManyToOne,
        /// <summary>
        /// The first input correspond to all outputs in the sequence. This in a sequence means that running the first input should give the first output. Afterwards, the model is run with no input (zero tensor) and the output is considered (in recurrent models).
        /// </summary>
        OneToMany,
    }
}

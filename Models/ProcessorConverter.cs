using ComputeShaders;
using Neuran.GradientDescent;
using Neuran.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Models
{
    public class ProcessorConverter : IGradientDescent
    {
        /// <inheritdoc/>
        public Tensor[] PreLayerDer { get; private set; }

        /// <inheritdoc/>
        public IGradientDescent ConnectedFrom { get; private set; }

        /// <inheritdoc/>
        public Tensor[] Output { get; private set; }

        /// <inheritdoc/>
        public Tensor[] Input { get; private set; }

        /// <summary>
        /// The type of conversion.
        /// </summary>
        public ProcessorConversionType ConversionType { get; private set; }

        /// <summary>
        /// Creates a new instance.
        /// </summary>
        /// <param name="dimensions"></param>
        /// <param name="device"></param>
        /// <param name="type"></param>
        public ProcessorConverter(int[] dimensions, CSDevice device, ProcessorConversionType type)
        {
            ConversionType = type;
            Input = new Tensor[] { new Tensor(type == ProcessorConversionType.GPUToCPU ? device : null, dimensions) };
            Output = new Tensor[] { new Tensor(type == ProcessorConversionType.GPUToCPU ? null : device, dimensions) };
        }

        /// <inheritdoc/>
        public void AddParameters(List<Tensor> parameters)
        {
            
        }

        /// <inheritdoc/>
        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            lossDer[0].CopyTo(PreLayerDer[0]);

            ConnectedFrom?.Backpropagate(PreLayerDer, pastTime);
        }

        /// <inheritdoc/>
        public void Connect(IGradientDescent model)
        {
            if (model.Output[0].ProcessorType != Input[0].ProcessorType)
                throw new ArgumentException("The model to connect from outputs a tensor with the wrong processor type!");

            ConnectedFrom = model;
            Input[0].Dispose();
            Input = model.Output;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (ConnectedFrom == null)
                Input[0].Dispose();
            Output[0].Dispose();
            PreLayerDer?[0].Dispose();
        }

        /// <inheritdoc/>
        public void EndGD()
        {
            PreLayerDer[0].Dispose();
            PreLayerDer = null;
        }

        /// <inheritdoc/>
        public void LoadRandomState()
        {
            
        }

        /// <inheritdoc/>
        public void PrepareGD(int maxTruncatedLength)
        {
            PreLayerDer = new Tensor[] { Input[0].EmptyClone() };
        }

        /// <inheritdoc/>
        public void Reset()
        {
            Output[0].Zero();
            if (ConnectedFrom == null)
                Input[0].Zero();
        }

        /// <inheritdoc/>
        public void ResetGradients()
        {
            
        }

        /// <inheritdoc/>
        public Tensor[] Run(Tensor[] input)
        {
            if (ConnectedFrom == null)
                input[0].CopyTo(Input[0]);
            input[0].CopyTo(Output[0]);

            return Output;
        }

        /// <inheritdoc/>
        public void SaveRandomState()
        {
            
        }

        
    }
    
    /// <summary>
    /// An enum for <see cref="ProcessorConverter"/>.
    /// </summary>
    public enum ProcessorConversionType
    {
        /// <summary>
        /// Convert from cpu tensor to gpu tensor.
        /// </summary>
        CPUToGPU,
        /// <summary>
        /// Convert from gpu tensor to cpu tensor.
        /// </summary>
        GPUToCPU,
    }
}

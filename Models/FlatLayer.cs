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
    public class FlatLayer : IGradientDescent
    {
        /// <inheritdoc/>
        public Tensor[] PreLayerDer { get; private set; }

        /// <inheritdoc/>
        public IGradientDescent ConnectedFrom { get; private set; }

        /// <inheritdoc/>
        public Tensor[] Output { get; private set; }

        /// <inheritdoc/>
        public Tensor[] Input { get; private set; }

        private GPUTensorProcesserApplier<Int4> applier;
        private GPUTensorProcesserApplier<Int4> derApplier;
        private Tensor lossDer;

        /// <summary>
        /// Creates a new instance.
        /// </summary>
        /// <param name="dimensions"></param>
        /// <param name="device"></param>
        /// <param name="type"></param>
        public FlatLayer(int[] dimensions, CSDevice device = null)
        {
            if (device != null)
                applier = new GPUTensorProcesserApplier<Int4>(device, null, TensorFlatShaders.Flat2, TensorFlatShaders.Flat3, Int4.Size, () =>
                {
                    Input[0].SetUAV(0);
                    Output[0].SetUAV(1);
                });

            Input = new Tensor[] { new Tensor(device, dimensions) };
            Output = new Tensor[] { new Tensor(device, Input[0].TensorLength) };
        }

        /// <inheritdoc/>
        public void AddParameters(List<Tensor> parameters)
        {

        }

        /// <inheritdoc/>
        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            this.lossDer = lossDer[0];

            if (derApplier == null)
            {
                for (int i = 0; i < lossDer[0].TensorLength; i++)
                {
                    PreLayerDer[0][i] = lossDer[0][i];
                }
            }
            else
            {
                derApplier.Run(new Int4()
                {
                    int1 = Input[0].Dimensions[0],
                    int2 = Input[0].Dimensions.Length >= 2 ? Input[0].Dimensions[1] : 0,
                    int3 = Input[0].Dimensions.Length >= 3 ? Input[0].Dimensions[2] : 0,
                }, Input[0].Dimensions);
            }

            //float[] d1 = this.lossDer.GetData(); //DEBUG
            //float[] d2 = PreLayerDer[0].GetData(); //DEBUG

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

            if (PreLayerDer != null)
                EndGD();
        }

        /// <inheritdoc/>
        public void EndGD()
        {
            PreLayerDer[0].Dispose();
            PreLayerDer = null;

            derApplier?.Dispose();
        }

        /// <inheritdoc/>
        public void LoadRandomState()
        {

        }

        /// <inheritdoc/>
        public void PrepareGD(int maxTruncatedLength)
        {
            PreLayerDer = new Tensor[] { Input[0].EmptyClone() };

            if (applier != null)
                derApplier = new GPUTensorProcesserApplier<Int4>(Input[0].device, null, TensorFlatShaders.Flat2Der, TensorFlatShaders.Flat3Der, Int4.Size, () =>
                {
                    lossDer.SetUAV(0);
                    PreLayerDer[0].SetUAV(1);
                });
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

            if (applier == null)
            {
                for (int i = 0; i < Input[0].TensorLength; i++)
                {
                    Output[0][i] = Input[0][i];
                }

                return Output;
            }

            applier.Run(new Int4()
            {
                int1 = Input[0].Dimensions[0],
                int2 = Input[0].Dimensions.Length >= 2 ? Input[0].Dimensions[1] : 0,
                int3 = Input[0].Dimensions.Length >= 3 ? Input[0].Dimensions[2] : 0,
            }, Input[0].Dimensions);

            return Output;
        }

        /// <inheritdoc/>
        public void SaveRandomState()
        {

        }


    }
}

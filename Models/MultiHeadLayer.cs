using Neuran.GradientDescent;
using Neuran.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Models
{
    /// <summary>
    /// A layer that splits input tensor(s) to multiple models in parallel.
    /// </summary>
    public class MultiHeadLayer : IGradientDescent
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
        /// The models connected in parallel.
        /// </summary>
        public IGradientDescent[] Models { get; private set; }
        /// <summary>
        /// Whether the input of every model in <see cref="Models"/> is a seperate tensor array in <see cref="Input"/> or all share the same input.
        /// </summary>
        public bool MultipleInput { get; private set; }

        private List<Tensor[]> modelsInput;
        private List<Tensor[]> modelsOutput;

        private List<Tensor[]> pastPrelayers;
        private MultiHeadLayerHelper[] helpers;
        private List<int> pastTimes;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="models">The models connected in parallel.</param>
        /// <param name="multipleInput">Whether the input of every model in <see cref="Models"/> is a seperate tensor array in <see cref="Input"/> or all share the same input. Note that if <paramref name="multipleInput"/> is false, then all models must share the same input structure and properties (cpu or gpu).</param>
        public MultiHeadLayer(IGradientDescent[] models, bool multipleInput)
        {
            MultipleInput = multipleInput;

            if (multipleInput)
                MultiInputInit(models);
            else
                SingleInputInit(models);
        }
        private void SingleInputInit(IGradientDescent[] models)
        {
            if (!SameStructure(models))
                throw new ArgumentException("The models do not share the same structure!");

            modelsOutput = new List<Tensor[]>();

            int outputSum = 0;
            for (int i = 0; i < models.Length; i++)
            {
                outputSum += models[i].Output.Length;
                modelsOutput.Add(models[i].Output.EmptyCloneArray());
            }

            Output = new Tensor[outputSum];

            Input = models[0].Input.EmptyCloneArray();
            modelsInput = new List<Tensor[]>()
            {
                models[0].Input.EmptyCloneArray(),
            };

            Models = new IGradientDescent[models.Length];

            int outputIndex = 0;

            for (int i = 0; i < models.Length; i++)
            {
                Models[i] = models[i];

                for (int j = 0; j < models[i].Output.Length; j++)
                {
                    Output[outputIndex] = models[i].Output[j].EmptyClone();
                    outputIndex++;
                }
            }
        }
        private void MultiInputInit(IGradientDescent[] models)
        {
            int outputSum = 0;
            int inputSum = 0;
            for (int i = 0; i < models.Length; i++)
            {
                inputSum += models[i].Input.Length;
                outputSum += models[i].Output.Length;
            }
            Input = new Tensor[inputSum];
            Output = new Tensor[outputSum];

            Models = new IGradientDescent[models.Length];

            modelsInput = new List<Tensor[]>(models.Length);
            modelsOutput = new List<Tensor[]>(models.Length);

            int outputIndex = 0;
            int inputIndex = 0;
            for (int i = 0; i < models.Length; i++)
            {
                Models[i] = models[i];
                modelsInput.Add(models[i].Input.EmptyCloneArray());
                modelsOutput.Add(models[i].Output.EmptyCloneArray());

                for (int j = 0; j < models[i].Input.Length; j++)
                {
                    Input[inputIndex] = models[i].Input[j].EmptyClone();
                    inputIndex++;
                }
                for (int j = 0; j < models[i].Output.Length; j++)
                {
                    modelsOutput.Add(models[i].Output.EmptyCloneArray());
                    Output[outputIndex] = models[i].Output[j].EmptyClone();
                    outputIndex++;
                }
            }

        }
        private bool SameStructure(IGradientDescent[] models)
        {
            Tensor[] input = models[0].Input;
            for (int i = 0; i < models.Length; i++)
            {
                if (input.Length != models[i].Input.Length)
                    return false;

                for (int j = 0; j < input.Length; j++)
                {
                    if (input[j].ProcessorType != models[i].Input[j].ProcessorType || input[j].Dimensions.Length != models[i].Input[j].Dimensions.Length)
                        return false;

                    for (int k = 0; k < input[j].Dimensions.Length; k++)
                    {
                        if (input[j].Dimensions[k] != models[i].Input[j].Dimensions[k])
                            return false;
                    }
                }
            }

            return true;
        }

        /// <inheritdoc/>
        public void AddParameters(List<Tensor> parameters)
        {
            for (int i = 0; i < Models.Length; i++)
            {
                Models[i].AddParameters(parameters);
            }
        }

        /// <inheritdoc/>
        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            int index = 0;
            for (int i = 0; i < Models.Length; i++)
            {
                for (int j = 0; j < Models[i].Output.Length; j++)
                {
                    lossDer[index + j].CopyTo(modelsOutput[i][j]);
                }
                index += Models[i].Output.Length;
            }

            for (int i = 0; i < Models.Length; i++)
            {
                Models[i].Backpropagate(modelsOutput[i], pastTime);
            }

            for (int i = 0; i < pastTimes.Count; i++)
            {
                int time = pastTimes[i];
                PreLayerDer = pastPrelayers[time]; //this is incase the ConnectedFrom model checks PreLayerDer
                ConnectedFrom?.Backpropagate(pastPrelayers[time], time);

                for (int j = 0; j < pastPrelayers[time].Length; j++)
                {
                    pastPrelayers[time][j].Zero();
                }
            }

            pastTimes.Clear();
            PreLayerDer = pastPrelayers[0];
        }

        /// <inheritdoc/>
        public void Connect(IGradientDescent model)
        {
            ConnectedFrom = model;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            for (int i = 0; i < Models.Length; i++)
            {
                Models[i].Dispose();
            }

            for (int i = 0; i < Input.Length; i++)
            {
                Input[i].Dispose();
            }
            for (int i = 0; i < Output.Length; i++)
            {
                Output[i].Dispose();
            }
        }

        /// <inheritdoc/>
        public void EndGD()
        {
            for (int i = 0; i < Models.Length; i++)
            {
                Models[i].EndGD();
                helpers[i].Dispose();
            }

            helpers = null;

            for (int i = 0; i < pastPrelayers.Count; i++)
            {
                for (int j = 0; j < pastPrelayers[i].Length; j++)
                {
                    pastPrelayers[i][j].Dispose();
                }
            }
            pastPrelayers = null;
            pastTimes = null;
        }

        /// <inheritdoc/>
        public void LoadRandomState()
        {
            for (int i = 0; i < Models.Length; i++)
            {
                Models[i].LoadRandomState();
            }
        }

        /// <inheritdoc/>
        public void PrepareGD(int maxTruncatedLength)
        {
            pastPrelayers = new List<Tensor[]>(maxTruncatedLength);
            for (int i = 0; i < maxTruncatedLength; i++)
            {
                pastPrelayers.Add(Input.EmptyCloneArray());
            }
            PreLayerDer = pastPrelayers[0];

            helpers = new MultiHeadLayerHelper[Models.Length];
            pastTimes = new List<int>();
            int index = 0;
            for (int i = 0; i < Models.Length; i++)
            {
                helpers[i] = new MultiHeadLayerHelper(Models[i], pastPrelayers, pastTimes, index);
                Models[i].PrepareGD(maxTruncatedLength);

                if (MultipleInput)
                    index += Models[i].Input.Length;
            }
        }

        /// <inheritdoc/>
        public void Reset()
        {
            for (int i = 0; i < Models.Length; i++)
            {
                Models[i].Reset();
            }
        }

        /// <inheritdoc/>
        public void ResetGradients()
        {
            for (int i = 0; i < Models.Length; i++)
            {
                Models[i].ResetGradients();
            }
        }

        /// <inheritdoc/>
        public Tensor[] Run(Tensor[] input)
        {
            int inputIndex = 0;
            for (int i = 0; i < modelsInput.Count; i++)
            {
                for (int j = 0; j < modelsInput[i].Length; j++)
                {
                    input[inputIndex + j].CopyTo(modelsInput[i][j]);
                }

                inputIndex += modelsInput[i].Length;
            }

            int outputIndex = 0;
            for (int i = 0; i < Models.Length; i++)
            {
                Tensor[] output = Models[i].Run(MultipleInput ? modelsInput[i] : modelsInput[0]);

                for (int j = 0; j < output.Length; j++)
                {
                    output[j].CopyTo(Output[outputIndex + j]);
                }

                outputIndex += output.Length;
            }

            return Output;
        }

        /// <inheritdoc/>
        public void SaveRandomState()
        {
            for (int i = 0; i < Models.Length; i++)
            {
                Models[i].SaveRandomState();
            }
        }
    }
}

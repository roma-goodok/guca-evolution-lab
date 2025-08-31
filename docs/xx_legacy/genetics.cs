/* (c) Roman Gudchenko 2010
This code provided as reference of Graph Unfolding Cellular Automata (GUCA) + genetic algorithm to reimplement it on Python
*/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using QuickGraph;
using Physical2DGraph;
using GraphUnfoldingMachine;
using AForge;
using AForge.Genetic;


namespace GraphUnfoldingMachine.Genetic
{
    public class GUMChromosome : IChromosome
    {

        /// <summary>
        /// "Age" of the chromosome.
        /// </summary>
        int age = 0;
        //public int Age { get { return age; } }

        int maxAge = 10;
        private int mutationAvgStepLengh = 10; // TODO: parameterize

        double singleActiveGenMutaionFactor = 0.1;
        MutationKind singleActiveGenMutaionKind = MutationKind.Byte;
        double singlePassiveGenMutaionFactor = 0.5;
        MutationKind singlePassiveGenMutaionKind = MutationKind.Byte;

        protected double fitness = 0;
        int maxLength;
        GumGen[] genes;

        protected bool isNeedToUpdateFitnessValue = true;
        public string activeGensScheme;
        public int activeGensCount;

        public void InvalidateFitnessValue()
        {
            isNeedToUpdateFitnessValue = true;
        }

        // random number generator for chromosomes generation
        //protected static ThreadSafeRandom rand = new ThreadSafeRandom((int)DateTime.Now.Ticks);
        //protected static Random rand = new Random((int)DateTime.Now.Ticks);

        double IChromosome.Fitness
        {
            get { return fitness; }
        }

        public GUMChromosome(
            int length,
            int maxLength,
            double singleActiveGenMutaionFactor,
            MutationKind singleActiveGenMutaionKind,
            double singlePassiveGenMutaionFactor,
            MutationKind singlePassiveGenMutaionKind
                            )
        {

            this.singleActiveGenMutaionFactor = singleActiveGenMutaionFactor;
            this.singleActiveGenMutaionKind = singleActiveGenMutaionKind;
            this.singlePassiveGenMutaionFactor = singlePassiveGenMutaionFactor;
            this.singlePassiveGenMutaionKind = singlePassiveGenMutaionKind;


            this.maxLength = maxLength;
            genes = new GumGen[length];
            for (int i = 0; i < length; i++)
            {
                genes[i] = new GumGen();
            }
            // save ancestor as a temporary head
            // generate the chromosome
            Generate();

        }

        /// <summary>
        /// Clone constructor
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public GUMChromosome(GUMChromosome source)
        {
            // allocate genes array
            genes = new GumGen[source.genes.Length];
            fitness = source.fitness;

            // copy genes
            for (int i = 0; i < genes.Length; i++)
            {
                genes[i] = new GumGen(source.genes[i].GetValue());
                genes[i].WasActive = source.genes[i].WasActive;
            }

            // copy fields
            maxLength = source.maxLength;
            age = source.age;
            singleActiveGenMutaionFactor = source.singleActiveGenMutaionFactor;
            singleActiveGenMutaionKind = source.singleActiveGenMutaionKind;
            singlePassiveGenMutaionFactor = source.singlePassiveGenMutaionFactor;
            singlePassiveGenMutaionKind = source.singlePassiveGenMutaionKind;
        }

        public void SetGAParametres(GAParametres parametres)
        {
            singleActiveGenMutaionFactor = parametres.SingleActiveGenMutaionFactor;
            singleActiveGenMutaionKind = parametres.SingleActiveGenMutaionKind;
            singlePassiveGenMutaionFactor = parametres.SinglePassiveGenMutaionFactor;
            singlePassiveGenMutaionKind = parametres.SinglePassiveGenMutaionKind;

        }

        public GUMChromosome(ChangeTable changeTable)
        {
            this.maxLength = changeTable.Count * 2;
            SetChangeTable(changeTable);
        }

        public int Age()
        {
            return age;
        }

        IChromosome IChromosome.Clone()
        {
            return new GUMChromosome(this);

        }

        IChromosome IChromosome.Born()
        {
            GUMChromosome result = new GUMChromosome(this);
            this.age++;
            result.age = 0;
            return result;

        }

        IChromosome IChromosome.CreateOffspring()
        {
            GUMChromosome result = new GUMChromosome(this);//.genes.Length, maxLength);
            if (RandomGen3.NextDouble() < 0.9)
            {
                (result as IChromosome).Mutate();
            }
            return result;
        }

        void IChromosome.Crossover(IChromosome pair)
        {
            //age++;
            //((GUMChromosome)pair).age++;
            // 1. During crossover the chromosome length changes. It can become shorter or longer.
            // 2. Parental chromosome lengths may differ.

            GumGen[] pairGens = ((GUMChromosome)pair).genes;

            int L1 = this.genes.Length;
            int L2 = pairGens.Length;
            int crossOverPoint1 = RandomGen3.Next(L1 - 1) + 1;
            int crossOverPoint2;

            bool Symmetric = false;
            if (Symmetric)
            {
                crossOverPoint2 = crossOverPoint1;
            }
            else
            {
                crossOverPoint2 = RandomGen3.Next(L2 - 1) + 1;
            }


            // New gene sequences after crossover:
            // genes1: 0..crossOverPoint1 from this.genes, then pairGens[crossOverPoint2..L2).
            int newL1 = Math.Min(maxLength, crossOverPoint1 + L2 - crossOverPoint2);
            int newL2 = Math.Min(maxLength, crossOverPoint2 + L1 - crossOverPoint1);

            GumGen[] genes1 = new GumGen[newL1];
            GumGen[] genes2 = new GumGen[newL2];



            Array.Copy(this.genes, 0, genes1, 0, crossOverPoint1);
            Array.Copy(pairGens, crossOverPoint2, genes1, crossOverPoint1, newL1 - crossOverPoint1);

            Array.Copy(pairGens, 0, genes2, 0, crossOverPoint2);
            Array.Copy(this.genes, crossOverPoint1, genes2, crossOverPoint2, newL2 - crossOverPoint2);

            this.genes = genes1;
            ((GUMChromosome)pair).genes = genes2;

            this.isNeedToUpdateFitnessValue = true;
            ((GUMChromosome)pair).isNeedToUpdateFitnessValue = true;

        }

        void IChromosome.Evaluate(IFitnessFunction function)
        {
            //if ((isNeedToUpdateFitnessValue) /*|| (age > maxAge)*/)
            //{
            //    /*if (age <= maxAge)
            //    {
            //     */

            fitness = function.Evaluate(this);
            isNeedToUpdateFitnessValue = false;
            //    /*}
            //    else
            //    {

            //        fitness = double.MinValue;
            //    }*/
            //}
        }

        public virtual void Generate()
        {
            foreach (GumGen gen in genes)
            {
                GenerateGen(gen);
            }

            isNeedToUpdateFitnessValue = true;
        }

        private static void GenerateGen(GumGen gen)
        {
            byte[] bytes = new byte[8];

            RandomGen3.NextBytes(bytes);
            ulong lng = BitConverter.ToUInt64(bytes, 0);

            gen.SetValue(lng);


        }



        private static void MutateGen(GumGen gen, MutationKind mutationKind)
        {
            //byte[] bytes = new byte[8];

            //rand.NextBytes(bytes);
            //ulong lng = BitConverter.ToUInt64(bytes, 0);

            switch (mutationKind)
            {
                case MutationKind.Bit:
                    // flip individual bits (part of the gene)
                    #region
                    {
                        ulong lng = gen.GetValue();


                        lng ^= ((ulong)1 << RandomGen3.Next(64));
                        lng ^= ((ulong)1 << RandomGen3.Next(64));
                        lng ^= ((ulong)1 << RandomGen3.Next(64));
                        lng ^= ((ulong)1 << RandomGen3.Next(64));

                        gen.SetValue(lng);
                    }
                    #endregion
                    break;
                case MutationKind.Byte:

                    // change one byte (part of the gene)
                    #region

                    {
                        byte[] bytes = new byte[8];

                        RandomGen3.NextBytes(bytes);
                        ulong lng = BitConverter.ToUInt64(bytes, 0);

                        ulong lngPrior = gen.GetValue();
                        ulong mask = 0x0000000000000000;
                        ulong maskNot = 0xFFFFFFFFFFFFFFFF;

                        switch (RandomGen3.Next(0, 7))
                        {
                            case 0:
                                mask = 0x00000000000000FF;
                                maskNot = 0xFFFFFFFFFFFFFF00;
                                break;

                            case 1:
                                mask = 0x000000000000FF00;
                                maskNot = 0xFFFFFFFFFFFF00FF;
                                break;

                            case 2:
                                mask = 0x0000000000FF0000;
                                maskNot = 0xFFFFFFFFFF00FFFF;
                                break;
                            case 3:
                                mask = 0x00000000FF000000;
                                maskNot = 0xFFFFFFFF00FFFFFF;
                                break;

                            case 4:
                                mask = 0x000000FF00000000;
                                maskNot = 0xFFFFFF00FFFFFFFF;
                                break;
                            case 5:
                                mask = 0x0000FF0000000000;
                                maskNot = 0xFFFF00FFFFFFFFFF;
                                break;
                            case 6:
                                mask = 0x00FF000000000000;
                                maskNot = 0xFF00FFFFFFFFFFFF;
                                break;
                            case 7:
                                mask = 0xFF00000000000000;
                                maskNot = 0x00FFFFFFFFFFFFFF;
                                break;

                        }


                        gen.SetValue((lng & maskNot) | (mask & lng));
                    }
                    #endregion
                    break;
                case MutationKind.AllBytes:
                    // change all bytes (replace with random value)
                    #region

                    {
                        byte[] bytes = new byte[8];

                        RandomGen3.NextBytes(bytes);
                        ulong lng = BitConverter.ToUInt64(bytes, 0);


                        gen.SetValue(lng);
                    }
                    #endregion
                    break;
                case MutationKind.Shift:
                    // rotate by 1 byte
                    #region
                    {
                        ulong lng = gen.GetValue();
                        lng = (lng << 8) + (lng >> 56);
                        gen.SetValue(lng);
                    }
                    break;
                    #endregion

                default:
                    break;
            }
        }

        void IChromosome.Mutate()
        {
            if ((singleActiveGenMutaionFactor == singlePassiveGenMutaionFactor) && (singleActiveGenMutaionKind == singlePassiveGenMutaionKind))
            {
                // If gene activity does not matter:
                for (int i = 0; i < genes.Length; i++)
                {
                    if (RandomGen3.NextDouble() < singleActiveGenMutaionFactor)
                    {
                        GUMChromosome.MutateGen(genes[i], singleActiveGenMutaionKind);
                    }
                }

            }
            else
            {
                // If mutation type and probability depend on whether the gene was active:

                int activeGensCount = 0;
                for (int i = 0; i < genes.Length; i++)
                {

                    if (genes[i].WasActive)
                    {
                        activeGensCount++;
                        if (RandomGen3.NextDouble() < singleActiveGenMutaionFactor)
                        {
                            GUMChromosome.MutateGen(genes[i], singleActiveGenMutaionKind);
                        }

                        if (RandomGen3.NextDouble() < singleActiveGenMutaionFactor * 0.2)
                        {
                            GUMChromosome.MutateGen(genes[i], MutationKind.Shift);
                        }
                    }
                    else
                    {
                        if (RandomGen3.NextDouble() < singlePassiveGenMutaionFactor)
                        {
                            GUMChromosome.MutateGen(genes[i], singlePassiveGenMutaionKind);
                        }

                        if (RandomGen3.NextDouble() < singlePassiveGenMutaionFactor * 0.2)
                        {
                            GUMChromosome.MutateGen(genes[i], MutationKind.Shift);
                        }
                    }

                }

                // Draft: random insertion of an active gene

                if (RandomGen3.NextDouble() < 0.1)
                {
                    if ((genes.Length < maxLength) && (activeGensCount > 1))
                    {
                        GumGen[] genes1 = new GumGen[genes.Length + 1];
                        // find a random index of an active gene:
                        int rndActiveNumberForInsertion = RandomGen3.Next(1, activeGensCount);
                        int activeGenIndex = -1;
                        int activeGenPassedCount = 0;
                        while (activeGenPassedCount < rndActiveNumberForInsertion)
                        {
                            activeGenIndex++;
                            if (genes[activeGenIndex].WasActive) { activeGenPassedCount++; };

                        }

                        Array.Copy(genes, 0, genes1, 0, activeGenIndex + 1);
                        Array.Copy(genes, activeGenIndex, genes1, activeGenIndex + 1, 1);
                        if (genes.Length - activeGenIndex - 1 > 0)
                        {
                            Array.Copy(genes, activeGenIndex + 1, genes1, activeGenIndex + 2, genes.Length - activeGenIndex - 1);
                        }

                        //this.genes = genes1;
                    }
                };

                if (RandomGen3.NextDouble() < 0.1)
                {

                    // Remove the first random inactive gene, if any

                    if ((activeGensCount < genes.Length) && (genes.Length >= 100))
                    {
                        int rndInActiveNumberForInsertion = RandomGen3.Next(1, genes.Length - activeGensCount);

                        int inactiveGenIndex = -1;
                        int inactiveGenPassedCount = 0;
                        while (inactiveGenPassedCount < rndInActiveNumberForInsertion)
                        {
                            inactiveGenIndex++;
                            if (!genes[inactiveGenIndex].WasActive) { inactiveGenPassedCount++; };

                        }

                        if ((inactiveGenIndex < genes.Length) && (inactiveGenIndex > 0))
                        {
                            GumGen[] genes1 = new GumGen[genes.Length - 1];
                            //Array.Copy(genes, genes1, genes.Length - 1);
                            Array.Copy(genes, 0, genes1, 0, inactiveGenIndex);
                            Array.Copy(genes, inactiveGenIndex + 1, genes1, inactiveGenIndex, genes.Length - 1 - inactiveGenIndex);

                            this.genes = genes1;
                        }

                    }


                };
            }

            // duplicate the first gene
            if (RandomGen3.NextDouble() < 0.2)
            {
                if ((genes.Length < maxLength))
                {
                    GumGen[] genes1 = new GumGen[genes.Length + 1];

                    Array.Copy(genes, 0, genes1, 0, 1);
                    Array.Copy(genes, 0, genes1, 1, genes.Length);

                    this.genes = genes1;
                }
            };

            isNeedToUpdateFitnessValue = true;
        }

        public void ArrangeActiveGens()
        {

            int activeGenCount = genes.Count(s => s.WasActive);
            GumGen[] arrangegGenes = new GumGen[genes.Length];
            int activeIdx = 0; // index iterating over active genes in the original array
            int nonactiveIdx = 0; // index iterating over inactive genes in the original array

            double ratio = (double)activeGenCount / genes.Length;

            for (int resultIdx = 0; resultIdx < genes.Length; resultIdx++)
            {
                int curValue = (int)Math.Floor((resultIdx) * ratio);
                int nextValue = (int)Math.Floor((resultIdx + 1) * ratio);
                bool isFound = false;

                if (nextValue > curValue)
                {
                    // resultIdx should hold an active gene
                    while (!isFound && activeIdx < genes.Length)
                    {
                        if (genes[activeIdx].WasActive)
                        {
                            isFound = true;
                            Array.Copy(genes, activeIdx, arrangegGenes, resultIdx, 1);
                        }
                        activeIdx++;
                    }


                }
                else
                {
                    // resultIdx should hold an inactive gene
                    while (!isFound && nonactiveIdx < genes.Length)
                    {
                        if (!genes[nonactiveIdx].WasActive)
                        {
                            isFound = true;
                            Array.Copy(genes, nonactiveIdx, arrangegGenes, resultIdx, 1);
                            // "turn off" the inactive gene
                            arrangegGenes[resultIdx].Connections_LE = 0;
                            arrangegGenes[resultIdx].Connections_GE = 1;
                            arrangegGenes[resultIdx].OperationType = 0; // turn to state
                            arrangegGenes[resultIdx].OperandStatus = arrangegGenes[resultIdx].Status;

                        }
                        nonactiveIdx++;
                    }
                }

            }

            if (arrangegGenes[arrangegGenes.Length - 1] == null)
            {
                Array.Copy(genes, arrangegGenes.Length - 1, arrangegGenes, arrangegGenes.Length - 1, 1);
            }

            this.genes = arrangegGenes;
        }

        public void SetActiveGensMask(System.Collections.BitArray activeGensBitArray)
        {
            //this.activeGensBitArray = activeGensBitArray;
            for (int i = 0; i < genes.Count(); i++)
            {
                genes[i].WasActive = activeGensBitArray[i];
            }
        }

        int IComparable.CompareTo(object obj)
        {
            double f = ((GUMChromosome)obj).fitness;
            int result;

            if (fitness == f)
            {
                GUMChromosome other = (GUMChromosome)obj;
                result = (this.age == other.age) ? 0 : (this.age > other.age) ? 1 : -1;
            }
            else
            {
                result = (fitness < f) ? 1 : -1; // sort by fitness descending
            }

            return result;

        }

        public ChangeTable GetChangeTable()
        {
            ChangeTable result = new ChangeTable();
            foreach (GumGen gen in genes)
            {
                result.Add(gen.ToChangeTableItem());
            }

            return result;
        }

        public void SetChangeTable(ChangeTable changeTable)
        {
            genes = new GumGen[changeTable.Count];
            for (int i = 0; i < changeTable.Count; i++)
            {
                genes[i] = new GumGen();
                genes[i].Status = (byte)changeTable[i].Condition.CurrentState;
                genes[i].PriorStatus = (byte)changeTable[i].Condition.PriorState;
                genes[i].Connections_GE = (byte)changeTable[i].Condition.AllConnectionsCount_GE;
                genes[i].Connections_LE = (byte)changeTable[i].Condition.AllConnectionsCount_LE;
                genes[i].Parents_GE = (byte)changeTable[i].Condition.ParentsCount_GE;
                genes[i].Parents_LE = (byte)changeTable[i].Condition.ParentsCount_LE;

                genes[i].OperationType = (byte)changeTable[i].Operation.Kind;
                genes[i].OperandStatus = (byte)changeTable[i].Operation.OperandNodeState;
                genes[i].WasActive = changeTable[i].WasActive;
            }
            // save ancestor as a temporary head
            // generate the chromosome
        }

        public int Length { get { return genes.Length; } }
    }


    public abstract class PlanarGraphFitnessFunction : IFitnessFunction
    {

        // fitness function parameters
        protected int maxVertexCount;
        protected int maxGUCAIterationCount;
        protected bool isGenomeLengthPenalty;
        protected bool isNotPlannedPenalty;
        protected NumericalMethodParametres unfoldingParametres;
        private TranscriptionWay transcriptionWay;
        //protected  int MaxGUCAIterationCount { get { return maxGUCAIterationCount; } }


        /// <summary>
        /// Fitness function for a planar graph: the closer the graph statistics are to the target, the better.
        /// </summary>
        /// <param name="fasetsDistributionByLength">Target distribution of faces by their length, as pairs (Key, Value).
        /// Key = face length, Value = count of faces with that length.</param>
        /// <param name="verticesDistributionByDegree">Target distribution of vertex degrees as pairs (Key, Value).
        /// Key = vertex degree, Value = count of vertices with that degree.</param>
        public PlanarGraphFitnessFunction(
                                            int maxGUCAIterationCount,
                                            int maxVertexCount,
                                            bool isGenomeLengthPenalty,
                                            bool isNotPlannedPenalty,
                                            NumericalMethodParametres unfoldingParametres,
                                            TranscriptionWay transcriptionWay
                                           )
        {
            this.maxGUCAIterationCount = maxGUCAIterationCount;
            this.maxVertexCount = maxVertexCount;
            this.isGenomeLengthPenalty = isGenomeLengthPenalty;
            this.isNotPlannedPenalty = isNotPlannedPenalty;
            this.unfoldingParametres = unfoldingParametres;
            this.transcriptionWay = transcriptionWay;
        }

        /// <summary>
        /// Evaluate the fitness of a chromosome.
        /// </summary>
        /// <param name="chromosome"></param>
        /// <returns></returns>
        double IFitnessFunction.Evaluate(IChromosome chromosome)
        {
            // Unfold a graph from the chromosome (grow the phenotype) and compute phenotype fitness.
            double result;
            int stepsPassed;
            Physical2DGraph.Physical2DGraph graph = GrowGraph(chromosome, out stepsPassed);

            // Basic filters: planarity, minimal size, etc.
            result = this.EvaluateCommonFilter(graph, stepsPassed);

            // Main fitness evaluation
            if (result == 1.0)
            {
                result = this.Evaluate(graph);// +1.0 / (((chromosome as GUMChromosome).Length));
            }

            // Bonus for shorter genome
            if (isGenomeLengthPenalty)
            {
                result = result + 1.0 / (((chromosome as GUMChromosome).Length));
            }
            return result;
        }

        /// <summary>
        /// Filter out minimally non-viable individuals (must have more than one node, be planar, etc.).
        /// </summary>
        /// <param name="graph"></param>
        /// <param name="stepsPassed"></param>
        /// <returns>Returns a value in [0, 1]. The rest of the fitness range is [1.0, +inf).</returns>
        private double EvaluateCommonFilter(Physical2DGraph.Physical2DGraph graph, int stepsPassed)
        {
            // Graph must not consist of a single node ("stillborn")
            if (graph.VertexCount == 1)
            {
                return 0;
            };

            // Graph must not be too large
            if ((graph.VertexCount >= maxVertexCount))
            {
                return 0.1;
            }

            // Unfolding process should not diverge (no endless growth)
            if (stepsPassed >= maxGUCAIterationCount - 2)
            {
                return 0.9;
            }

            bool isPlanar = graph.Planarize();

            if (!isPlanar)
            {
                return 0.3;
            }

            return 1.0;
        }

        /// <summary>
        /// Grow the phenotype from the genotype (build a graph from the chromosome).
        /// </summary>
        /// <param name="chromosome"></param>
        /// <returns></returns>
        public Physical2DGraph.Physical2DGraph GrowGraph(IChromosome chromosome, out int stepsPassed)
        {

            GUMGraph gumGraph = new GUMGraph();
            gumGraph.AddVertex(new GUMNode(NodeState.A));
            gumGraph.MaxVerticesCount = maxVertexCount;
            gumGraph.MaxConnectionCount = 6;

            GraphUnfoldingMachine graphUnfoldingMachine = new GraphUnfoldingMachine(gumGraph);

            graphUnfoldingMachine.MaxStepsCount = maxGUCAIterationCount;
            graphUnfoldingMachine.Support1Connected = true;
            graphUnfoldingMachine.TranscriptionWay = this.transcriptionWay;

            graphUnfoldingMachine.ChangeTable = TranslateNative(chromosome);
            graphUnfoldingMachine.Reset();
            graphUnfoldingMachine.Run();

            #region Fill active genes pattern
            int activeGensCount = 0;
            bool priorGenIsActive = false;

            int counter = 0;

            if (graphUnfoldingMachine.ChangeTable.Count > 0)
            {
                priorGenIsActive = graphUnfoldingMachine.ChangeTable[0].WasActive;
            }

            StringBuilder sb = new StringBuilder();

            foreach (var chi in graphUnfoldingMachine.ChangeTable)
            {
                if (chi.WasActive)
                {
                    activeGensCount++;

                }

                if (chi.WasActive)
                {
                    if (!priorGenIsActive)
                    {
                        sb.Append(counter);
                    }

                    sb.Append("x");

                    counter = 1;
                }
                else
                {

                    counter++;
                }

                priorGenIsActive = chi.WasActive;

                //// If the status continues, just increase the counter
                //if (chi.WasActive == priorGenIsActive)
                //{
                //    counter++;
                //}
                //else
                //{
                //    // If activity status changed, record the count of the previous run
                //    sb.AppendFormat(priorGenIsActive ? "x" : "o{0}", counter );
                //    priorGenIsActive = chi.WasActive;
                //    counter = 1;
                //}
            }

            if (!priorGenIsActive)
            {
                sb.Append(counter);
            }

            stepsPassed = graphUnfoldingMachine.PassedStepsCount;

            (chromosome as GUMChromosome).activeGensScheme = sb.ToString();
            (chromosome as GUMChromosome).activeGensCount = activeGensCount;

            #endregion

            #region Build and pass to the chromosome the bitmask of gene activity
            System.Collections.BitArray activeGensBitArray = new System.Collections.BitArray(graphUnfoldingMachine.ChangeTable.Count, false);

            for (int i = 0; i < graphUnfoldingMachine.ChangeTable.Count; i++)
            {
                if (graphUnfoldingMachine.ChangeTable[i].WasActive)
                {
                    activeGensBitArray[i] = true;
                }
            }

            chromosome.SetActiveGensMask(activeGensBitArray);
            #endregion

            return graphUnfoldingMachine.Graph;
        }

        /// <summary>
        /// Evaluate the fitness of the phenotype (grown graph).
        /// </summary>
        /// <param name="graph"></param>
        /// <returns>Must return a value in [1.0, +inf).</returns>
        public abstract double Evaluate(Physical2DGraph.Physical2DGraph graph);

        #region OLD (legacy notes and drafts; kept for reference)
        // Old, generic version (kept commented for historical reference).
        // The legacy exploratory code below compares target and current
        // distributions (vertex degrees and face lengths) and experimented
        // with various constraints and penalties for planarity and mesh quality.
        // If you need this legacy block fully translated, I can provide a
        // separate variant with all inline comments in English.
        #endregion

        object IFitnessFunction.Translate(IChromosome chromosome)
        {
            return TranslateNative(chromosome);

        }

        public ChangeTable TranslateNative(IChromosome chromosome)
        {
            if (chromosome == null)
            {
                return new ChangeTable();
            }
            else
            {
                return ((GUMChromosome)chromosome).GetChangeTable();
            }
        }
    }

    public class BySamplePlanarGraphFitnessFunction : PlanarGraphFitnessFunction
    {

        Dictionary<int, int> fasetsDistributionByLength;
        Dictionary<int, int> verticesDistributionByDegree;
        /// <summary>
        /// Fitness function for a planar graph: the closer the graph statistics are to the provided target, the better.
        /// </summary>
        /// <param name="fasetsDistributionByLength">Target distribution of faces by their length, as pairs (Key, Value).
        /// Key = face length, Value = count of faces with that length.</param>
        /// <param name="verticesDistributionByDegree">Target distribution of vertex degrees as pairs (Key, Value).
        /// Key = vertex degree, Value = count of vertices with that degree.</param>
        public BySamplePlanarGraphFitnessFunction(Dictionary<int, int> fasetsDistributionByLength,
                                            Dictionary<int, int> verticesDistributionByDegree,
                                            int maxGUCAIterationCount,
                                            int maxVertexCount,
                                            bool isGenomeLengthPenalty,
                                            bool isNotPlannedPenalty,
                                            NumericalMethodParametres unfoldingParametres,
                                            TranscriptionWay transcriptionWay)
            : base(maxGUCAIterationCount, maxVertexCount, isGenomeLengthPenalty, isNotPlannedPenalty, unfoldingParametres, transcriptionWay)
        {

            this.fasetsDistributionByLength = fasetsDistributionByLength;
            this.verticesDistributionByDegree = verticesDistributionByDegree;
        }

        /// <summary>
        /// Returns the distribution of vertices by degree.
        /// </summary>
        private Dictionary<int, int> GetVerticiesDistributionByDegree(Physical2DGraph.Physical2DGraph graph)
        {

            var vertParts = from x in graph.Vertices
                            group x by graph.AdjacentEdges(x).Count() into part
                            orderby part.Key
                            select new
                            {
                                Key = part.Key,
                                Count = part.Count()
                            };
            Dictionary<int, int> result = new Dictionary<int, int>();
            foreach (var part in vertParts)
            {
                result.Add(part.Key, part.Count);
            }

            return result;
        }

        /// <summary>
        /// Returns the distribution of faces by their length.
        /// </summary>
        private Dictionary<int, int> GetFasetsDistributionByLength(List<Cycle<Physical2DGraphVertex>> fasets)
        {

            var fasetParts = from x in fasets
                             group x by x.Count
                                 //group x by x.MinCycleLength
                                 into part
                                 orderby part.Key
                                 select new
                                 {
                                     Key = part.Key,
                                     Count = part.Count()
                                 };


            Dictionary<int, int> result = new Dictionary<int, int>();
            foreach (var part in fasetParts)
            {
                result.Add(part.Key, part.Count);
            }

            return result;
        }

        public override double Evaluate(Physical2DGraph.Physical2DGraph graph)
        {

            if (isNotPlannedPenalty)
            {
                // TODO: use knowledge of topology to speed up unfolding
                graph.ProcessUnfoldingModel(this.unfoldingParametres, new Point(0, 0)); // expensive operation!
                if (!graph.IsPlanned()) { return 1.2; }
            }


            if (!graph.IsPlanarized)
            {
                graph.Planarize();
            }

            // 1. Compute the current distributions of vertex degrees and face lengths.
            Dictionary<int, int> curFasetsDistributionByLength = graph.GetFasetsDistributionByLength();
            Dictionary<int, int> curVerticesDistributionByDegree = graph.GetVerticiesDistributionByDegree();

            // 2. Compute the distance between target and current vertex-degree distributions
            var vertUnion = (from x in curVerticesDistributionByDegree
                             select new
                             {
                                 Key = x.Key,
                                 Value = -x.Value
                             }).Union
                                (from y in this.verticesDistributionByDegree
                                 select new
                                 {
                                     Key = y.Key,
                                     Value = y.Value
                                 }
                                );

            var vertGroupedSum =
                    from x in vertUnion
                    group x by x.Key into g
                    select new { g.Key, KeySum = g.Sum(x => x.Value) };

            int vertDistance = vertGroupedSum.Sum(x => Math.Abs(x.KeySum));

            // 3. Compute the distance between target and current face-length distributions
            var fasetsUnion = (from x in curFasetsDistributionByLength
                               select new
                               {
                                   Key = x.Key,
                                   Value = -x.Value
                               }).Union
                               (from y in fasetsDistributionByLength
                                select new
                                {
                                    Key = y.Key,
                                    Value = y.Value
                                }
                               );

            int fasetDistance =
                    (from x in fasetsUnion
                     group x by x.Key into g
                     select new { g.Key, KeySum = g.Sum(x => x.Value) })
                    .Sum(x => Math.Abs(x.KeySum));

            // 4. The larger the distance, the less fit the graph.
            // First priority: number of faces.
            double result = Math.Abs(fasetsDistributionByLength.Sum(x => x.Value) - curFasetsDistributionByLength.Sum(x => x.Value));
            //result =  - result*100 - 10*fasetDistance - vertDistance;

            return result;
        }

    }

    public class TriangleMeshPlanarGraphFitnessFunction : PlanarGraphFitnessFunction
    {

        double shellVertexWeight;

        public TriangleMeshPlanarGraphFitnessFunction(
            int maxGUCAIterationCount,
            int maxVertexCount,
            bool isGenomeLengthPenalty,
            bool isNotPlannedPenalty,
            NumericalMethodParametres unfoldingParametres,
            TranscriptionWay transcriptionWay,
            double shellVertexWeight)
            : base(maxGUCAIterationCount, maxVertexCount, isGenomeLengthPenalty, isNotPlannedPenalty, unfoldingParametres, transcriptionWay)
        {
            this.shellVertexWeight = shellVertexWeight;
        }

        public override double Evaluate(Physical2DGraph.Physical2DGraph graph)
        {

            double result;

            if (!graph.IsPlanarized)
            {
                graph.Planarize();
            }

            List<Cycle<Physical2DGraphVertex>> fasets = graph.Fasets;

            // target facet length
            int AimMeshCycleLength = 3;
            // target vertex degree
            int AimVertexDegree = 6;

            if (graph.VertexCount <= 2) { return 1.0; }

            // To avoid expensive computations, pre-filter topology for minimal viability
            // before planar layout and vertex statistics.
            // Additional constraints: graph must have at least one face
            if (fasets.Count == 1) { return 1.01; }
            // Additional: graph must be biconnected
            if (!graph.IsBeconnected()) { return 1.02; }
            // Additional: after 2D unfolding the graph must be drawable (planar embedding)

            // Maximum vertex degree must not exceed 6
            int MaxVertexDegree = graph.Vertices.Max(x => graph.AdjacentDegree(x));
            if (MaxVertexDegree > 6) { return 1.03; }

            if (isNotPlannedPenalty)
            {
                // TODO: use topology knowledge to speed up unfolding
                graph.ProcessUnfoldingModel(this.unfoldingParametres, new Point(0, 0)); // expensive operation!
                if (!graph.IsPlanned()) { return 1.06; }
            }

            double wellConnectedVertexCount = graph.VertexCount;// (double)graph.Vertices.Where(x => x.ConnectionsCount > 1).Count();

            Cycle<Physical2DGraphVertex> shellFaset = (from f in fasets orderby f.MinCycleLength descending select f).First();

            int MaxCycleLength = shellFaset.MinCycleLength; // fasets.Max(x => x.MinCycleLength);

            int perimetr = shellFaset.Count;
            double innerVertexCount = graph.VertexCount - perimetr;
            result = 1.01 * (double)fasets.Count + 1.1 * innerVertexCount - perimetr + 20;

            // count vertices with target degree, excluding those on the boundary
            double outerAimVertexCount = (double)shellFaset.Vertices.Take(shellFaset.Count).Where(x => x.ConnectionsCount == AimVertexDegree).Count();
            double innerAimVertexCount = (double)graph.Vertices.Where(x => x.ConnectionsCount == AimVertexDegree).Count() - outerAimVertexCount;

            double Fasets3Count = (from f in fasets where f.MinCycleLength == 3 select f).Count();
            // exclude outer cycle if it's of target length
            if (MaxCycleLength == AimMeshCycleLength) { Fasets3Count = Fasets3Count - 1; };

            double Fasets4Count = (from f in fasets where f.MinCycleLength == 4 select f).Count();
            // exclude outer cycle if it's of target length
            if (MaxCycleLength == AimMeshCycleLength) { Fasets4Count = Fasets4Count - 1; };

            //result = 2 * Fasets3Count + innerAimVertexCount + shellVertexWeight*outerAimVertexCount - graph.VertexCount;
            result = 2 * Fasets3Count + innerAimVertexCount - shellVertexWeight * shellFaset.Count - graph.VertexCount + 20;

            //result = 2 * Fasets3Count + innerAimVertexCount + shellVertexWeight * outerAimVertexCount - MaxCycleLength - Fasets4Count;

            return result;
        }

    }

    public class QuadricMeshPlanarGraphFitnessFunction : PlanarGraphFitnessFunction
    {
        double shellVertexWeight;
        private double faset3penaltyProbability;

        public QuadricMeshPlanarGraphFitnessFunction(
                                            int maxGUCAIterationCount,
                                            int maxVertexCount,
                                            bool isGenomeLengthPenalty,
                                            bool isNotPlannedPenalty,
                                            NumericalMethodParametres unfoldingParametres,
                                            TranscriptionWay transcriptionWay,
                                            double shellVertexWeight,
                                            double faset3penaltyProbability)
            : base(maxGUCAIterationCount, maxVertexCount, isGenomeLengthPenalty, isNotPlannedPenalty, unfoldingParametres, transcriptionWay)
        {
            this.shellVertexWeight = shellVertexWeight;
            this.faset3penaltyProbability = faset3penaltyProbability;
        }


        public override double Evaluate(Physical2DGraph.Physical2DGraph graph)
        {

            double result;

            if (!graph.IsPlanarized)
            {
                graph.Planarize();
            }

            List<Cycle<Physical2DGraphVertex>> fasets = graph.Fasets;

            double penalty = 0;

            // target vertex degree
            int AimVertexDegree = 4;

            if (graph.VertexCount <= 2) { return 1.0; }

            // Pre-filter topology for minimal viability before expensive layout/stats
            // Additional: must have at least one face
            if (fasets.Count == 1) { return 1.01; }
            // Additional: must be biconnected
            if (!graph.IsBeconnected()) { return 1.02; }
            // Additional: must allow a planar embedding after 2D unfolding

            // Maximum vertex degree should not exceed 4 (with early returns for larger)
            int MaxVertexDegree = graph.Vertices.Max(x => graph.AdjacentDegree(x));
            if (MaxVertexDegree > 6) { return 1.03; }
            if (MaxVertexDegree > 5) { return 1.04; }
            if (MaxVertexDegree > 4) { return 1.06; }

            int MinCycleLength = fasets.Min(x => x.MinCycleLength);

            //if ((MinCycleLength < 4) && (fasets.Count == 2)) { return 1.07; }
            //if (MinCycleLength < 4)  { return 1.08; }

            if (faset3penaltyProbability > 0)
            {
                int Fasets3Count = (from f in fasets where f.MinCycleLength == 3 select f).Count();

                if ((Fasets3Count > 0) && (RandomGen3.NextDouble() < faset3penaltyProbability)) { return 1.08; };
            }

            // "Mesh shell" (outer cycle)
            Cycle<Physical2DGraphVertex> shellFaset = (from f in fasets orderby f.MinCycleLength descending select f).First();

            if (isNotPlannedPenalty)
            {
                // TODO: use topology knowledge to speed up unfolding
                graph.ProcessUnfoldingModel(this.unfoldingParametres, new Point(0, 0)); // expensive operation!
                if (!graph.IsPlanned()) { return 1.1; }
            }

            double wellConnectedVertexCount = graph.VertexCount;// (double)graph.Vertices.Where(x => x.ConnectionsCount > 1).Count();

            int MaxCycleLength = shellFaset.MinCycleLength; // fasets.Max(x => x.MinCycleLength);

            // Count vertices with target degree, excluding those on the boundary
            double outerAimVertexCount = (double)shellFaset.Vertices.Take(shellFaset.Count).Where(x => x.ConnectionsCount == AimVertexDegree).Count();
            double innerAimVertexCount = (double)graph.Vertices.Where(x => x.ConnectionsCount == AimVertexDegree).Count() - outerAimVertexCount;

            if (fasets.Count > 2)
            {
                // Vertices with degree less than target must be boundary only.
                // If any interior vertex has degree < 4, penalize.
                int innerNotAimVertexCount = graph.Vertices.Where(x => (x.ConnectionsCount != AimVertexDegree && !shellFaset.Vertices.Contains(x))).Count();

                //penalty = penalty + innerNotAimVertexCount * 10.0;
                //return 1.08;
                if (innerNotAimVertexCount > 0) return 1.11;
            }

            // For rings: exclude the outer cycle
            double Fasets4Count = (from f in fasets where f.MinCycleLength == 4 select f).Count();
            if (MaxCycleLength == 4) { Fasets4Count = Fasets4Count - 1; };

            //double Fasets3Count = (from f in fasets where f.MinCycleLength == 3 select f).Count();
            //if (MaxCycleLength == 3) { Fasets4Count = Fasets3Count - 1; };

            //result = Math.Max(3.99, 2 * Fasets4Count + 2*innerAimVertexCount + shellVertexWeight * outerAimVertexCount - wellConnectedVertexCount + 4.0 - penalty);
            double koef = Fasets4Count <= 4 ? 2.1 : 2;
            result = Math.Max(3.98, 2.1 * Fasets4Count + 2 * innerAimVertexCount - wellConnectedVertexCount + 10.0);

            return result;
        }

    }

    public class HexMeshPlanarGraphFitnessFunction : PlanarGraphFitnessFunction
    {

        double shellVertexWeight;

        public HexMeshPlanarGraphFitnessFunction(
                                            int maxGUCAIterationCount,
                                            int maxVertexCount,
                                            bool isGenomeLengthPenalty,
                                            bool isNotPlannedPenalty,
                                            NumericalMethodParametres unfoldingParametres,
                                            TranscriptionWay transcriptionWay,
                                            double shellVertexWeight)
            : base(maxGUCAIterationCount, maxVertexCount, isGenomeLengthPenalty, isNotPlannedPenalty, unfoldingParametres, transcriptionWay)
        {
            this.shellVertexWeight = shellVertexWeight;
        }

        public override double Evaluate(Physical2DGraph.Physical2DGraph graph)
        {

            double result;

            if (!graph.IsPlanarized)
            {
                graph.Planarize();
            }

            List<Cycle<Physical2DGraphVertex>> fasets = graph.Fasets;

            // target facet length
            int AimMeshCycleLength = 6;
            // target vertex degree
            int AimVertexDegree = 3;

            if (graph.VertexCount <= 2) { return 1.0; }

            // Pre-filter topology for minimal viability before expensive layout/stats:
            // Additional: must have at least one face
            if (fasets.Count == 1) { return 1.01; }
            // Additional: must be biconnected
            if (!graph.IsBeconnected())
            {
                if (RandomGen3.NextDouble() < shellVertexWeight)
                {
                    return 1.02;
                }
            }
            // Additional: must allow a planar embedding after 2D unfolding

            // Maximum vertex degree must not exceed 6 (allowing small hex-only cases)
            int MaxVertexDegree = graph.Vertices.Max(x => graph.AdjacentDegree(x));
            if (MaxVertexDegree > 6) { return 1.03; }
            if ((MaxVertexDegree > 5) && (graph.VertexCount > 7)) { return 1.04; }
            if ((MaxVertexDegree > 4) && (graph.VertexCount > 7)) { return 1.05; }
            //if ((MaxVertexDegree > 3) && (graph.VertexCount > 7)) 
            //{
            //    if (RandomGen3.NextDouble() < shellVertexWeight)
            //    {
            //        return 1.05;
            //    }
            //}

            double V = graph.VertexCount;// (double)graph.Vertices.Where(x => x.ConnectionsCount > 1).Count();

            // "shell" = the largest face
            Cycle<Physical2DGraphVertex> shell = (from f in fasets orderby f.MinCycleLength descending select f).First();

            int MaxCycleLength = shell.MinCycleLength; // fasets.Max(x => x.MinCycleLength);

            // Count interior vertices whose degree != 3
            int VinnerNot3 = graph.Vertices.Where(x => (x.ConnectionsCount != 3) && !shell.Vertices.Contains(x)).Count();
            if (VinnerNot3 > 0)
            { // There are interior vertices with degree != 3
                List<Physical2DGraphVertex> Vinnrt4List = graph.Vertices.Where(x => (x.ConnectionsCount == 4) && !shell.Vertices.Contains(x)).ToList();
                //int Vinner4 = graph.Vertices.Where(x => (x.ConnectionsCount == 4) && !shell.Vertices.Contains(x)).Count();
                // However, allow degree-4 vertices provided that
                // they belong to at most two hexagonal faces
                if (!(VinnerNot3 == Vinnrt4List.Count))
                {
                    // There exist interior vertices with degree not equal to 4 (2,5,6...)
                    return 1.08;
                }
                else
                {
                    if (!((Vinnrt4List.Count < 4) && (!Vinnrt4List.Exists(x => x.Fasets.Where(f => f.MinCycleLength == 6).Count() > 2))))
                    {
                        return 1.09;
                    }
                }
            };

            // Also forbid boundary degree-4 vertices that belong to > 2 hexagons
            var VOuter4List = graph.Vertices.Where(x => (x.ConnectionsCount == 4) && shell.Vertices.Contains(x));

            if (VOuter4List.Any(x => x.Fasets.Where(f => f.MinCycleLength == 6).Count() > 2))
            {
                return 1.1;
            }

            if (isNotPlannedPenalty)
            {
                // TODO: use topology knowledge to speed up unfolding
                if (RandomGen3.NextDouble() < shellVertexWeight)
                {
                    graph.ProcessUnfoldingModel(this.unfoldingParametres, new Point(0, 0)); // expensive operation!
                    if (!graph.IsPlanned())
                    {
                        return 1.06;
                    }
                }
            }

            // Count interior degree-3 vertices that touch only hexagonal faces
            // (i.e., if shell length != 6, the "interior" check can be skipped)
            double V3inner = graph.Vertices.Where(x => x.ConnectionsCount == 3 && x.Fasets.Where(f => f.MinCycleLength == 6).Count() == 3).Count();

            double F6 = (from f in fasets where f.MinCycleLength == 6 select f).Count();
            if (MaxCycleLength == 6) { F6 = F6 - 1; };

            double F3 = (from f in fasets where f.MinCycleLength == 3 select f).Count();
            if (MaxCycleLength == 3) { F3 = F3 - 1; }

            if (F3 > 0)
            {
                if (RandomGen3.NextDouble() < shellVertexWeight)
                {
                    return 1.05;
                }
            }

            double F4 = (from f in fasets where f.MinCycleLength == 4 select f).Count();
            if (MaxCycleLength == 4) { F4 = F4 - 1; };
            //double Fasets5Count = (from f in fasets where f.MinCycleLength == 5 select f).Count();
            //if (MaxCycleLength == 5) { Fasets5Count = Fasets5Count - 1; };

            result = 0;

            if (F4 > 10)
                if (RandomGen3.NextDouble() < shellVertexWeight)
                {
                    result = result - 10;
                }

            //result = 12 * aimFasetsCount + innerAimVertexCount + shellVertexWeight * outerAimVertexCount - wellConnectedVertexCount + 0.0 * Fasets5Count + 4.0 * Math.Min(32.0, Fasets4Count) + 1 * Math.Min(32.0, Fasets3Count) + 4.0;
            result = result + 5.1 * F6 + 2.01 * F4 + 4.0 * V3inner - V + 10.0;

            return Math.Max(result, 1.1);
        }

    }

    public class GumGen
    {

        public bool WasActive = false;
        public byte Status;
        public byte PriorStatus;
        public byte Connections_GE;
        public byte Connections_LE;
        public byte Parents_GE;
        public byte Parents_LE;
        public byte OperationType;
        public byte OperandStatus;

        public GumGen()
        {
            SetValue(0);
        }

        public GumGen(ulong value)
        {
            SetValue(value);
        }

        public ulong GetValue()
        {
            ulong result = 0;

            result = result | this.Status;
            result = result << 8;
            result = result | this.PriorStatus;
            result = result << 8;
            result = result | this.Connections_GE;
            result = result << 8;
            result = result | this.Connections_LE;
            result = result << 8;
            result = result | this.Parents_GE;
            result = result << 8;
            result = result | this.Parents_LE;
            result = result << 8;
            result = result | this.OperationType;
            result = result << 8;
            result = result | this.OperandStatus;

            return result;
        }

        public void SetValue(ulong value)
        {

            this.OperandStatus = Math.Max((byte)1, (byte)(value & 0x1F));
            value = value >> 8;

            this.OperationType = (byte)(value & 0x0F);
            value = value >> 8;

            this.Parents_LE = (byte)(value);
            value = value >> 8;

            this.Parents_GE = (byte)(value);
            value = value >> 8;

            this.Connections_LE = (byte)(value & 0x0F);
            value = value >> 8;

            this.Connections_GE = (byte)(value & 0x0F);
            value = value >> 8;

            this.PriorStatus = (byte)(value & 0x1F);
            value = value >> 8;

            this.Status = Math.Max((byte)1, (byte)(value & 0x1F));
        }

        public ChangeTableItem ToChangeTableItem()
        {
            OperationCondition condition = new OperationCondition();
            Operation operation = new Operation();

            condition.CurrentState = (NodeState)Status;
            //condition.PriorState = (NodeState)Status;
            condition.PriorState = (NodeState)PriorStatus;
            //condition.PriorState = NodeState.Ignored;

            condition.AllConnectionsCount_GE = Connections_GE > 8 ? -1 : Connections_GE;
            condition.AllConnectionsCount_LE = Connections_LE > 8 ? -1 : Connections_LE;
            //condition.AllConnectionsCount_GE = Connections_GE % 8;
            //condition.AllConnectionsCount_LE = Connections_LE % 8;

            //condition.AllConnectionsCount_LE = condition.AllConnectionsCount_GE;

            condition.ParentsCount_GE = Parents_GE > 64 ? -1 : Parents_GE;
            condition.ParentsCount_LE = Parents_LE > 64 ? -1 : Parents_LE;

            //condition.ParentsCount_LE = condition.ParentsCount_GE;

            /* Allowed operations:
                TurnToState  - 0x0,
                TryToConnectWithNearest 0x1,
                GiveBirthConnected 0x2,
                DisconnectFrom 0x3
                Die 0x4,
            */
            operation.Kind = (OperationKindEnum)(OperationType % 4);
            operation.OperandNodeState = (NodeState)OperandStatus;

            return new ChangeTableItem(condition, operation);
        }
    }
}

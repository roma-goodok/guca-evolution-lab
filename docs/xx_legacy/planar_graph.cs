/* (c) roma@goodok.ru  */
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

// сторонние компоненты
using QuickGraph;
using QuickGraph.Algorithms.Search;



// !! Silverlight compatible version


namespace Physical2DGraph
{


    public class NewVertexSeedModel
    {
        private Random _rnd;
        public Random Random { get { return _rnd; } set { _rnd = value; } }

        //GN_SeedAreaSize : double = 20;
        /// <summary>
        /// Радиус окружности рядом центром плоскости, в пределах которой создаётся новый узел со случайной позицией
        /// </summary>
        public double SeedAreaSize = 20D;

        //GN_SeedAreaCenter : double = 2500;
        /// <summary>
        /// Центр 2D плоскости - рядом с которым создаётся новый узел со случайной позицией
        /// </summary>
        public Point SeedAreaCenter = new Point(2500, 2500);


        //GN_ToBornNewNodeNearByDistance : double = 5.0;
        /// <summary>
        /// Радиус окружности рядом с уже существующим узлом, в пределах которой создаётся новый узел со случайной позицией
        /// </summary>
        public double ToBornNewNodeNearByDistance = 5.0D;

        public NewVertexSeedModel()
        {
            _rnd = new Random(0);
        }

        internal Point GetNextPosition()
        {
            return new Point(_rnd.NextDouble() * SeedAreaSize + SeedAreaCenter.X, _rnd.NextDouble() * SeedAreaSize + SeedAreaCenter.Y);
        }

        internal Point GetNextPositionNear(Point point)
        {
            //    tGeometry2DNode(xNewNode).fPosition := PointF2(tGeometry2DNode(xOldNode).Position.x + random*GN_ToBornNewNodeNearByDistance,
            //                                           tGeometry2DNode(xOldNode).Position.y + random*GN_ToBornNewNodeNearByDistance);

            return new Point(point.X + _rnd.NextDouble() * ToBornNewNodeNearByDistance, point.Y + _rnd.NextDouble() * ToBornNewNodeNearByDistance);
        }
    }

    public class PhysicalModelParametres
    {
        ////GN_UnLadenConnectionLength : float      = 5.0;  
        ///// <summary>
        ///// размер связи не нагруженный
        ///// </summary>  
        //public double UnLadenConnectionLength = 40.0D;

        ////GN_NodeGravitationKoeff : float         = 10.0;    
        ///// <summary>
        ///// коэффициент силы отталкивания между узлами
        ///// </summary>
        //public double NodeGravitationKoeff = 100.0D;


        ////GN_ConectionStiffnessFactor : float     = 0.5;
        ///// <summary>
        ///// коэффициент упругости соеденителя (для закона Гука)
        ///// </summary>
        //public double ConectionStiffnessFactor = 0.5D;


        ////GN_TimeStep : float                     = 0.1;
        ///// <summary>
        ///// временной шаг расчёта модели
        ///// </summary>
        //public double TimeStep = 0.1D;

        ////GN_NodeInertialMass : float             = 0.1;    
        ///// <summary>
        ///// инерционаня масса = коэффициент между ускорением и силой
        ///// </summary>
        //public double NodeInertialMass = 0.1D;


        ////GN_ResistingForceFactor_0 : float       = 0.1;
        ///// <summary>
        ///// постоянный коэффициент для расчёта силы сопротивления Force = GN_ResistingForceFactor_0 + GN_ResistingForceFactor_1*Velocity)
        ///// </summary>
        //public double ResistingForceFactor_0 = 1.0D;

        ////GN_ResistingForceFactor_1 : float       = 0.4;
        ///// <summary>
        ///// коэффициент для расчёта силы сопротивления Force = GN_ResistingForceFactor_0 + GN_ResistingForceFactor_1*Velocity)
        ///// </summary>
        //public double ResistingForceFactor_1 = 4D;


        ////GN_IsWithConnectedNodesGravitationInteractionOnly : boolean = false; 
        ///// <summary>
        ///// гравитационное отталкивание действует только на связанные узлы. Иначе - между всеми узлами сети (большие объёмы расчётов)
        ///// </summary>
        //public bool IsWithConnectedNodesGravitationInteractionOnly = false;


        //GN_UnLadenConnectionLength : float      = 5.0;  
        /// <summary>
        /// размер связи не нагруженный
        /// </summary>  
        public double FreeConnectionLength = 2.0D;

        //GN_NodeGravitationKoeff : float         = 10.0;    
        /// <summary>
        /// коэффициент гравитационной силы отталкивания между узлами
        /// </summary>
        public double NodeGravitationCoefficient = 20.0D; // 100


        //GN_ConectionStiffnessFactor : float     = 0.5;
        /// <summary>
        /// коэффициент упругости соеденителя (для закона Гука)
        /// </summary>
        public double ConectionGripFactor = 0.5D;


        //GN_NodeInertialMass : float             = 0.1;    
        /// <summary>
        /// инерционаня масса = коэффициент между ускорением и силой
        /// </summary>
        public double NodeInertialMass = 0.1D;

        //GN_ResistingForceFactor_0 : float       = 0.1;
        /// <summary>
        /// постоянный коэффициент для расчёта силы сопротивления Force = GN_ResistingForceFactor_0 + GN_ResistingForceFactor_1*Velocity)
        /// </summary>
        public double ResistingForceFactor_0 = 1D;


        //GN_ResistingForceFactor_1 : float       = 0.4;
        /// <summary>
        /// коэффициент для расчёта силы сопротивления Force = GN_ResistingForceFactor_0 + GN_ResistingForceFactor_1*Velocity)
        /// </summary>
        public double ResistingForceFactor_1 = 0.5D;


        //GN_IsWithConnectedNodesGravitationInteractionOnly : boolean = false; 
        /// <summary>
        /// гравитационное отталкивание действует только на связанные узлы. Иначе - между всеми узлами сети (большие объёмы расчётов)
        /// </summary>
        public bool IsWithConnectedNodesGravitationInteractionOnly = false;

        public double MaxVelocity = 10;

    }

    /// <summary>
    /// Метод численной реализации модели
    /// </summary>
    public enum NumericalMethod
    {
        /// <summary>
        /// 
        /// </summary>
        Euler,
        /// <summary>
        /// Метод релаксации Ньютона
        /// </summary>
        NewtonRelax,
        /// <summary>
        /// Метод Рунге Кутта 4-го порядка
        /// </summary>
        RungeKutta4

    }

    /// <summary> Параметры численного метода расчёта физической модели.
    /// После очередной итерации модификации графа несколько раз вызывается расчёт физической моделиотрисовки результата.
    /// </summary>
    public class NumericalMethodParametres
    {

        public NumericalMethod Method;
        ///// <summary>Количество вызовов обсчёта физической модели и отрисовки результата за одну итерацию модификации графа
        ///// </summary>
        //public int DusplayAndCalculationStepsCount = 10;



        /// <summary>
        /// Количество внешних итераций, в ходе каждой из которых запускается один обсчёт физической модели.
        /// Перед каждым обсчётом физической модели - пересчитывается положение центра масс узлов класстеров
        /// 
        /// </summary>
        public int OuterIterationCount = 1;
        
        /// <summary> Количество итерации для одного обсчёта физической модели
        /// </summary>
        public int OneProcessIterationCount = 100;

        /// <summary> шаг времени одной итерации 
        /// </summary>
        public double OneIterationTimeStep = 0.1D;

        /// <summary> Значение ошибки для условия останова        
        /// </summary>
        public double StopConditionErrorValue = 0.01D;

        /// <summary> Автоматическое уменьшение шага времени, для обеспечения устойчивости метода Эйлера. Критерий : ошибка difference превышает 1.0
        /// </summary>
        public bool AutoDecreaseTimeStep = true;

        /// <summary> Не рассчитывать силу гравитации от далёких узлов на каждой итерации, а использовать центры масс "далёких" кластеров
        /// </summary>
        public bool DontCalculateRemoteNodesGravitation = true;

        /// <summary> количество ближайших узлов, которые считаются близкими        
        /// </summary>
        public int NearestNodesCountThreshold = 10;

    }

    public class NewtonNumericalMethodParametres : NumericalMethodParametres
    {


        /// <summary>
        /// приращение dx (dy) Для численного вычисления производных.
        /// </summary>
        public double Delta = 0.00001D; //0.00001D;


        /// <summary>
        /// тормоз метода Ньютона (чтоб модель не бесилась) (в принципе можно оставить 1, только хорошо подобрать параметр MaxShift)
        /// </summary>
        public double BrakeFactor = 1D;

        /// <summary>
        /// Максимальеное перемещение узла (чтоб модель не бесилась)
        /// </summary>
        public double MaxShift = 1D;


        /// <summary>
        /// перемещать точку сразу (не откладывать), после расчета ее новых координат
        /// (так лучше, а то будет резонанс - для можели из двух вершин, когда точно расчитывается будущее положение для одной вершины при закрепленной второй, и для другой - они притянуться на вбое большее расстояние)
        /// </summary>
        public bool IsMoveNow = true;

        /// <summary> Коструктор контейнера параметров численного метода со значениями по умолчанию
        /// </summary>        

        public NewtonNumericalMethodParametres()
        {
            Method = NumericalMethod.NewtonRelax;
        }

        /// <summary> Коструктор контейнера параметров численного метода со указанием значений        
        /// <param name="iterationCount">Количество итераций</param>
        /// <param name="delta">приращение dx (dy) Для численного вычисления производных.</param>
        /// <param name="tormoz">тормоз метода Ньютона (чтоб модель не бесилась)</param>
        /// <param name="maxShift">максимальеное перемещение узла (чтоб модель не бесилась)</param>
        /// <param name="isMoveNow">перемещать точку сразу (не откладывать), после расчета ее новых координат</param>
        public NewtonNumericalMethodParametres(int iterationCount, double delta, double brakeFactor, double maxShift, bool isMoveNow)
        {
            Method = NumericalMethod.NewtonRelax;
            Delta = delta;
            BrakeFactor = brakeFactor;
            MaxShift = maxShift;
            IsMoveNow = isMoveNow;
            OneProcessIterationCount = iterationCount;
        }

        /// <summary>
        /// Решать систему двух уравнений относительно координат точки (а не отдельные уравнения)
        /// </summary>
        public bool isTwoCoords = true;
    }


    /// <summary>
    /// Вершина "физического" графа на плоскости
    /// </summary>
    public class Physical2DGraphVertex : IComparable<Physical2DGraphVertex>
    {

        int IComparable<Physical2DGraphVertex>.CompareTo(Physical2DGraphVertex other)
        {
            return this.GetHashCode() - other.GetHashCode();
        }

        /// <summary>
        /// ссылка на фигуру отображения
        /// Применяется для отображения графа на плоскости 2D
        /// (пока полагаем, что отображалка MVC бужет в одном экземпляре)        
        /// Объект отображение создаётся при первом рисовании (т.е. пока граф не отрисовывается, объекты не создаются)
        /// При удалении узла - фигура отображения удаляется при очередной перерисовке
        /// Вид фигуры, способ её отображения определяет объект View в MVC (класс Physical2DGraphCanvas)
        /// </summary>
        public Object visualForCanvasViewer = null; // равна null если объект отображения ещё не создан


        public object Tag;

        internal int isMarkedForAlgorithms = 0;



        protected internal Physical2DGraph _ownerGraph;

        /// <summary>
        /// Если граф планаризован - список граней, к которым принадлежит вершина.
        /// Заполняется при планеризации графа Physical2DGraph.Planarize()
        /// </summary>
        protected internal List<Cycle<Physical2DGraphVertex>> fasets = new List<Cycle<Physical2DGraphVertex>>();
        public List<Cycle<Physical2DGraphVertex>> Fasets { get { return fasets; }  }

        public Physical2DGraph ownerGraph { get { return _ownerGraph; } }
        /// <summary>
        /// Количество соединений с вершиной (для графа, к которому пренадлежит вершина)
        /// </summary>
        public int ConnectionsCount
        {
            get
            {
                return _ownerGraph.AdjacentEdges(this).Count();
            }
        }

        /// <summary> множество ближайших узлов
        /// Фиксируется раз в несколько итераций численного метода для оптимизации расчёта гравитационной силы
        /// </summary> 
        protected internal List<Physical2DGraphVertex> nearestNodes = new List<Physical2DGraphVertex>();
        public List<Physical2DGraphVertex> NearestNodes { get { return nearestNodes; } set { nearestNodes = value; } }

        protected internal Vector _remoteNodesGravitationForce;

        /// <summary> Заполнение множества ближайших узлов. Вычисление силы гравитации для остальных узлов
        /// </summary>
        /// <param name="count"></param>
        public void FillNearesNodes(int count)
        {


            // 1. определяем список ближайших узлов:
            // пока просто берём список узлов того же кластера, к которому принадлежит текущий узел.
            // TODO: на самом деле можно взять список узлов всех кластеров, к которым принадлежит этот и узлы соседние к этому узлу 
            if (ClusterID != -1)
            {
                nearestNodes = _ownerGraph.clusters[this.ClusterID].Vertices.ToList<Physical2DGraphVertex>();
                nearestNodes.Remove(this);

            }
            else
            {
                nearestNodes.Clear();
            }

            // 2. вычисляем (оценку) гравитационную силу остальных узлов

            _remoteNodesGravitationForce.X = 0; _remoteNodesGravitationForce.Y = 0;
            foreach (var cluster in _ownerGraph.clusters)
            {
                if (cluster.ID != this.ClusterID)
                {
                    double distance = _ownerGraph.VectorMath_Distance(this.Position, cluster.Center);
                    Vector direction = Vector.Subtract((Vector)this.Position, (Vector)cluster.Center);
                    double GFToSingleNode_Value = cluster.SumGravitationMass * _ownerGraph.PhysicalModelParametres.NodeGravitationCoefficient / (distance * distance);
                    Vector vGFToSingleNode = Vector.Multiply(direction, GFToSingleNode_Value);
                    _remoteNodesGravitationForce = Vector.Add(_remoteNodesGravitationForce, vGFToSingleNode);
                }
            }

            #region OLD
            //nearestNodes = _ownerGraph.Vertices.ToList();
            //// удаляем себя из списка
            //nearestNodes.Remove(this);

            ////nearestNodes.Sort(DistanceComparison); // самая затратная часть

            //_remoteNodesGravitationForce.X = 0; _remoteNodesGravitationForce.Y = 0;

            //if (nearestNodes.Count > count)
            //{


            //    for (int i = count; i < nearestNodes.Count - count; i++)
            //    {
            //        Physical2DGraphVertex vertexTo = nearestNodes[i];
            //        double distance = _ownerGraph.VectorMath_Distance(this.Position, vertexTo.Position);
            //        Vector direction = Vector.Subtract((Vector)this.Position, (Vector)vertexTo.Position);
            //        double GFToSingleNode_Value = _ownerGraph.PhysicalModelParametres.NodeGravitationCoefficient / (distance * distance);
            //        Vector vGFToSingleNode = Vector.Multiply(direction, GFToSingleNode_Value);
            //        _remoteNodesGravitationForce = Vector.Add(_remoteNodesGravitationForce, vGFToSingleNode);
            //    }

            //    nearestNodes.RemoveRange(count, nearestNodes.Count - count);                
            //}
            #endregion

        }

        public int DistanceComparison(Physical2DGraphVertex x, Physical2DGraphVertex y)
        {

            Vector sub = Vector.Subtract((Vector)x.Position, (Vector)this.Position);
            double Lx = Math.Sqrt(Vector.Multiply(sub, sub));

            sub = Vector.Subtract((Vector)y.Position, (Vector)this.Position);
            double Ly = Math.Sqrt(Vector.Multiply(sub, sub));

            return Math.Sign(Lx - Ly);

        }

        // -- для расчётов
        protected internal Vector velocity;    // текущая скорость узла в 2D пространстве
        protected internal Vector force;       // вектор суммарной силы, прилагаемой к узлу
        protected internal Vector acselerationPrior; // вектор ускорения

        // временные значения для расчётов по явной итерационной схеме
        protected internal Point _newPosition;
        protected internal Vector _newVelocity;


        #region временные значения для метода Рунге-Кутта
        // координата  x
        protected internal double _xk1; //  для уравнения x' = Vx
        protected internal double _xk2;
        protected internal double _xk3;
        protected internal double _xk4;

        protected internal double _xm1; //  для уравнения Vx' = ... '
        protected internal double _xm2;
        protected internal double _xm3;
        protected internal double _xm4;

        // координата  y
        protected internal double _yk1; //  для уравнения y' = Vy
        protected internal double _yk2;
        protected internal double _yk3;
        protected internal double _yk4;

        protected internal double _ym1; //  для уравнения Vy' = ... '
        protected internal double _ym2;
        protected internal double _ym3;
        protected internal double _ym4;
        #endregion

        protected internal int clusterID = -1;
        public int ClusterID { get { return clusterID; } set { clusterID = value; } }

        protected internal Point _oldPosition;
        protected internal Vector _oldVelocity;

        /// <summary>текущее положение узла на плоскости
        /// </summary> 
        protected internal Point position;
        public Point Position { get { return position; } set { position = value; } }

        public Physical2DGraphVertex()
        {
            velocity = new Vector(0, 0);
            force = new Vector(0, 0);

        }

        public Physical2DGraphVertex(Point position)
        {
            velocity = new Vector(0, 0);
            force = new Vector(0, 0);
            this.position = position;

        }

        public override string ToString()
        {
            if (Tag == null)
            {
                return Position.ToString();
            }
            else
            {
                return String.Format("{0}; Tag='{1}'", Position.ToString(), Tag.ToString());
            }
        }
    }

    /// <summary>
    /// Ребро "физического" графа на плоскости
    /// </summary>
    /// <typeparam name="TVertex"></typeparam>
    public class Physical2DGraphEdge<TVertex> : Edge<TVertex>
    {


        /// <summary>
        /// Фигура, созданная для рисования.
        /// </summary>
        public Object visualForCanvasViewer;

        public Physical2DGraphEdge(TVertex source, TVertex target)
            : base(source, target)
        {

        }



        #region Для алгоритма поиска циклов
        public bool IsEmbededFromSourceToTarget = false;
        public bool IsEmbededFromTargetToSource = false;
        public bool IsPassedFromSourceToTarget = false;
        public bool IsPassedFromTargetToSource = false;

        public bool IsEmbededFrom(TVertex from)
        {
            bool result = false;
            if (from.Equals(this.Source))
            {
                result = IsEmbededFromSourceToTarget;
            };
            if (from.Equals(this.Target))
            {
                result = IsEmbededFromTargetToSource;
            };
            return result;

        }
        public bool IsEmbededTo(TVertex from)
        {
            bool result = false;
            if (from.Equals(this.Target))
            {
                result = IsEmbededFromSourceToTarget;
            };
            if (from.Equals(this.Source))
            {
                result = IsEmbededFromTargetToSource;
            };
            return result;

        }

        internal bool IsEmbeded()
        {
            return (IsEmbededFromSourceToTarget || IsEmbededFromTargetToSource);
        }

        public bool IsPassedFrom(TVertex from)
        {
            bool result = false;
            if (from.Equals(this.Source))
            {
                result = IsPassedFromSourceToTarget;
            };
            if (from.Equals(this.Target))
            {
                result = IsPassedFromTargetToSource;
            };
            return result;

        }
        public bool IsPassedTo(TVertex from)
        {
            bool result = false;
            if (from.Equals(this.Target))
            {
                result = IsPassedFromSourceToTarget;
            };
            if (from.Equals(this.Source))
            {
                result = IsPassedFromTargetToSource;
            };
            return result;

        }

        /// <summary>
        /// Ребро пройдено хотя бы в одном направлении
        /// </summary>
        /// <returns></returns>
        public bool IsPassed()
        {
            return (IsPassedFromSourceToTarget || IsPassedFromTargetToSource);
        }

        public bool IsPassedBothDirection()
        {
            return (IsPassedFromSourceToTarget && IsPassedFromTargetToSource);
        }

        #endregion

        #region для отображения принадлежности к  граням (Временно)
        public bool IsBelongToFacet = false;

        #endregion





    }


    /// <summary>
    /// "Кластер" узлов - для оптимизации расчёта гравитационного взаимодействия
    /// </summary>
    public class Physical2DGraphVertexCluster
    {
        public int ID;
        public Point Center;
        public double SumGravitationMass;
        public List<Physical2DGraphVertex> Vertices = new List<Physical2DGraphVertex>();

    }




    /// <summary>
    /// Направленный цикл: начальная вершина + список рёбер.
    /// Любое ребро может быть пройдено в двух направлениях (но не обязатаельно)
    /// При этом рёбра могут повторяться, только если у этой пары рёбре разное направление обхода.    
    /// Одно и тоже ребро не может быть в цикле рядом.
    /// </summary>
    /// <typeparam name="TVertex"></typeparam>
    public class Cycle<TVertex> : System.Collections.Generic.List<Physical2DGraphEdge<TVertex>>, /*ICloneable,*/ IEquatable<Cycle<TVertex>>
    {


        public TVertex RootVertex = default(TVertex);

        //VerticesEnumerable<TVertex> verticesEnumerable;


        public Cycle(TVertex rootVertext)
        {
            RootVertex = rootVertext;
            //verticesEnumerable = new VerticesEnumerable<TVertex>(this);
        }



        public IEnumerable<TVertex> GetVerticesIterator()
        {
            TVertex current = this.RootVertex;
            yield return current;

            foreach (Physical2DGraphEdge<TVertex> edge in this)
            {
                if (current.Equals(edge.Source))
                {
                    current = edge.Target;
                }
                else
                {
                    current = edge.Source;
                }

                yield return current;
            }
        }


        /// <summary>
        /// Модификация цикла - получение обратного (инвертированного) цикла путём изменения направления. Начальная вершина остаётся прежней.
        /// При этом грани пройденные в обоих направлениях - не попадают в инвертированный цикл.
        /// Вообще, говоря, инвертированных циклов может получиться несколько(например, если исходный цикл -  "восмёрка" ввиде двух колец, сединённыйх ребром)
        /// Если исходный цикл - дерево (и все грани - пройдены в обоих направления), то инвертированный цикл - пустой
        /// </summary>
        public void Invert()
        {

            TVertex newRoot = this.Vertices.Last();

            Physical2DGraphEdge<TVertex>[] array = this.ToArray();
            this.Clear();
            // добавляем в обратном порядке:
            for (int i = 0; i < array.Length; i++)
            {
                Physical2DGraphEdge<TVertex> edge = array[array.Length - 1 - i];

                // добавляем только те рёбра, которые в исходном цикле присутствуют только в одном экземпляре
                if (array.Count(delegate(Physical2DGraphEdge<TVertex> e) { return e.Equals(edge); }) == 1)
                {
                    this.Add(edge);
                }
            }

            RootVertex = newRoot;

        }

        /// <summary>
        /// Возвращаем список узлов
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder(this.Count + 1);

            sb.AppendFormat("Edges: {0}; Vertices:", this.Count);
            foreach (TVertex v in this.Vertices)
            {
                sb.Append(v.ToString() + " -> ");
            }

            return sb.ToString();

        }

        /// <summary>
        /// Считаем маршруты равными, если равны их списки (ссылок!) на рёбра, даже если начальные вершины различны
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public bool Equals(Cycle<TVertex> other)
        {

            bool result = true;


            // 1. Проверяем равны ли длины маршрутов
            if (this.Count == other.Count)
            {
                if (this.Count != 0)
                {
                    // 2. Находим первое ребро из нашего списка в другом списке
                    int p = other.IndexOf(this[0]);

                    // списки равны, если this[i] = other[i+p], other[i+p+Count]
                    for (int i = 1; i < this.Count; i++)
                    {

                        if (i + p < this.Count)
                        {
                            if (this[i] != other[i + p])
                            {
                                result = false;
                                break;
                            }
                        }
                        else
                        {
                            if (this[i] != other[i + p - this.Count])
                            {
                                result = false;
                                break;
                            }
                        }
                    }
                }
                // если this.Count == 0 и othe.Count == 0, то считаем циклы равными
            }
            else
            {
                result = false;
            }


            return result;
        }

        /// <summary>
        /// Клонируем корневой узел и и список рёбер, но не рёбра
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            Cycle<TVertex> result = new Cycle<TVertex>(this.RootVertex);
            result.AddRange(this);
            return result;
        }

        /// <summary>
        /// Все рёбра помечаются как размещённые на плоскости.
        /// При этом учитывается направления. Считаем что направление обхода рёбер любой грани кроме внешней - по часовой стрелке 
        /// Т.е. каждое уложенное на плоскость ребро принадлежит двум граням (включая внешнюю грань), 
        /// то ребро помечается в двух направлениях
        /// </summary>
        protected internal void MarkAllEdgesAsEmbedded()
        {
            TVertex current = RootVertex;

            foreach (Physical2DGraphEdge<TVertex> edge in this)
            {
                if (current.Equals(edge.Source))
                {
                    // 1. помечаем, что ребро пройдено путешественником в направлении от текущей вершины
                    edge.IsEmbededFromSourceToTarget = true;
                    current = edge.Target;
                }
                else
                {
                    edge.IsEmbededFromTargetToSource = true;
                    current = edge.Source;
                }

            }
        }


        public IEnumerable<TVertex> Vertices { get { return GetVerticesIterator(); } }

        int minCycleChached = -1;
        public void MinCycleLengthCache()
        {
            minCycleChached = GetMinCycle().Count;
        }

        public int MinCycleLength { get { return minCycleChached; } }
        /// <summary>
        /// Минимальный цикл (без деревянных ответвлений)
        /// </summary>
        public Cycle<TVertex> GetMinCycle()
        {
            Cycle<TVertex> result = new Cycle<TVertex>(this.RootVertex);
            // удаляем все рёбра, которые присутствуют в исходном цикле более чем 1 раз
            foreach (Physical2DGraphEdge<TVertex> edge in this)
            {
                if (result.Contains(edge))
                {
                    result.Remove(edge);
                }
                else
                {
                    result.Add(edge);
                }

            }


            return result;
        }

        internal bool? isTree = null;

        public bool IsTree
        {
            get { return GetIsTree(); }
        }

        private bool GetIsTree()
        {
            if (isTree == null)
            {
                throw new NotImplementedException();
                // TODO: проверить вырожденный или нет цикл
                //return false;
            }
            else
            {
                return (bool)isTree;
            }
        }


        /// <summary>
        /// Вставка новой последовательности path после узла vertex. Если vertex не обнаружен, то вставка
        /// осуществляется в конец исходного цикла
        /// </summary>
        /// <param name="vertex"></param>
        /// <param name="path"></param>
        internal void InsertAfterFirstVertex(Physical2DGraphVertex vertex, Cycle<TVertex> path)
        {
            int i = 0;
            TVertex current = this.RootVertex;

            foreach (Physical2DGraphEdge<TVertex> edge in this)
            {
                if (current.Equals(edge.Source))
                {
                    current = edge.Target; // следующий узел
                }
                else
                {
                    current = edge.Source;
                }

                i = i + 1;

                if (current.Equals(vertex))
                {
                    break;
                }

            }

            this.InsertRange(i, path);
        }

        /// <summary>
        /// Замена участка цикла от firstVertex до lastVertex, новой последовательностю subPath.        
        /// </summary>
        /// <param name="firstTerminalVertex"></param>
        /// <param name="lastVertex"></param>
        /// <param name="path"></param>
        /// <remarks></remarks>
        internal void Replace(TVertex firstVertex, TVertex lastVertex, Cycle<TVertex> subPath)
        {
            // firstVertex может быть расположен в списке рёбер позже lastVertex
            // TODO: проверить, существуют ли firstVertex и lastVertex

            int firstVertexPosition = -1;
            TVertex current = this.RootVertex;

            // ищем позицию firstVertex
            foreach (Physical2DGraphEdge<TVertex> edge in this)
            {
                if (current.Equals(edge.Source))
                {
                    current = edge.Target; // следующий узел
                }
                else
                {
                    current = edge.Source;
                }

                firstVertexPosition = firstVertexPosition + 1;

                if (current.Equals(firstVertex))
                {
                    break;
                }

            };

            // ищем позицию lastVertex [индекс ребра, конец которого по направлению совпадает с узлом lastVertex]
            int lastVertexPosition = -1;
            current = this.RootVertex;
            foreach (Physical2DGraphEdge<TVertex> edge in this)
            {
                if (current.Equals(edge.Source))
                {
                    current = edge.Target; // следующий узел
                }
                else
                {
                    current = edge.Source;
                }

                lastVertexPosition = lastVertexPosition + 1;

                if (current.Equals(lastVertex))
                {
                    break;
                }

            };

            // удаляем участок начиная от firstVertexPosition до узла lastVertex или до конца (если узел lastVertex расположен раньше firstVertex)
            if (lastVertexPosition > firstVertexPosition)
            {
                // удаляем:
                this.RemoveRange(firstVertexPosition + 1, lastVertexPosition - firstVertexPosition);
                // вставляем
                InsertRange(firstVertexPosition + 1, subPath);
            }
            else
            {
                // удаляем:
                // после firstVertexPosition до конца
                if (firstVertexPosition < this.Count - 1) { this.RemoveRange(firstVertexPosition + 1, this.Count - firstVertexPosition - 1); };
                // с начала и до lastVertexPosition
                this.RemoveRange(0, lastVertexPosition + 1);

                // вставляем в конец:

                this.InsertRange(this.Count, subPath);
                this.RootVertex = lastVertex;



            }






        }
    }

    /// <summary>
    /// "Путешественник" - используется для алгоритма нахождения кратчайших двунаправленных циклов.
    /// Имеет текущее положение (вершина) и пройденный путь (цикл).
    /// Может проходить по грани в двух направления, помечая пройденость грани двумя поетками для каждого направления
    /// </summary>
    public class TwoDirectionWalker /*: ICloneable */
    {
        Physical2DGraph Owner = null;
        public Physical2DGraphVertex CurrentPosition = null;
        public Cycle<Physical2DGraphVertex> PassedPath = null;

        public TwoDirectionWalker(Physical2DGraph owner, Physical2DGraphVertex startPosition)
        {
            Owner = owner;
            CurrentPosition = startPosition;
            PassedPath = new Cycle<Physical2DGraphVertex>(startPosition);
            //PassedPath.RootVertex = startPosition;
        }

        /// <summary>
        /// Перемещение "Путешественника" на новую позицию. При этом грань помечается как пройденная в соотв. направлении.
        /// </summary>
        /// <param name="newPosition"></param>
        public void StepTo(Physical2DGraphVertex newPosition)
        {
            // 1. Находим общее ребро

            Physical2DGraphEdge<Physical2DGraphVertex> commonEdge;
            if (Owner.FindCommonEdge(CurrentPosition, newPosition, out commonEdge))
            {
                StepOver(commonEdge);
            }
            else
            {

                // TODO: raise exception  QuickGraphException
            }



        }

        /// <summary>
        /// Перемещение "Путешественника" на новую позицию. При этом грань помечается как пройденная в соотв. направлении.
        /// </summary>
        /// <param name="newPosition"></param>
        public void StepOver(Physical2DGraphEdge<Physical2DGraphVertex> edge)
        {
            // проверяем, принадлежит ли вершина текущего положения ребру:
            System.Diagnostics.Debug.Assert((edge.Target.Equals(CurrentPosition) || edge.Source.Equals(CurrentPosition)));


            if (CurrentPosition == edge.Source)
            {
                // 1. помечаем, что ребро пройдено путешественником в направлении от текущей вершины
                edge.IsPassedFromSourceToTarget = true;
                CurrentPosition = edge.Target;
            }
            else
            {
                edge.IsPassedFromTargetToSource = true;
                CurrentPosition = edge.Source;
            }
            PassedPath.Add(edge);
        }

        /// <summary>
        /// Проверка: может ли путешественник перемещаться на новую позицию. 
        /// </summary>
        /// <param name="newPosition"></param>
        /// <returns></returns>
        public bool CanStepTo(Physical2DGraphVertex newPosition)
        {
            // 1. Находим общее ребро
            bool result = false;

            Physical2DGraphEdge<Physical2DGraphVertex> commonEdge;

            if (Owner.FindCommonEdge(CurrentPosition, newPosition, out commonEdge))
            {
                result = CanStepOver(commonEdge);
            }
            else
            {

                // TODO: raise exception  QuickGraphException
            }


            return result;
        }

        public bool CanStepOver(Physical2DGraphEdge<Physical2DGraphVertex> edge)
        {


            System.Diagnostics.Debug.Assert((edge.Target.Equals(CurrentPosition) || edge.Source.Equals(CurrentPosition)));

            // 1. Находим общее ребро
            bool result = true;

            // 1. Проверяем, помечено ли ребро как пройденное (прохождение ребра не запрещено)
            if (edge.IsPassedFrom(CurrentPosition) || edge.IsEmbededFrom(CurrentPosition))
            {
                result = false;
            }




            // 2. Проверяем - не равно ли ребро предыдущему (последнему) --или первому

            if ((result == true) && (PassedPath.Count > 0))
            {
                Physical2DGraphEdge<Physical2DGraphVertex> lastEdge = PassedPath[PassedPath.Count - 1];
                if (lastEdge.Equals(edge))
                {
                    result = false;
                }

                //if (PassedPath.Count > 1)
                //{
                //    Physical2DGraphEdge<Physical2DGraphVertex> firstEdge = PassedPath[0];
                //    if (firstEdge.Equals(edge))
                //    {
                //        result = false;
                //    }
                //}
            }




            return result;
        }

        public bool С(Physical2DGraphEdge<Physical2DGraphVertex> edge)
        {
            // 3. Проверяем, не отмечена ли противолежащая вершина как запрещённая?
            Physical2DGraphVertex oppVertex;
            if (CurrentPosition == edge.Source)
            {
                oppVertex = edge.Target;
            }
            else
            {
                oppVertex = edge.Source;
            };

            return true;
        }


        public object Clone()
        {
            TwoDirectionWalker result;
            result = new TwoDirectionWalker(Owner, this.CurrentPosition);
            result.PassedPath = (Cycle<Physical2DGraphVertex>)PassedPath.Clone();


            return result;

        }






    }

    /// <summary>
    /// "Путешественник" - испольлзуется для алгоритма проверки шарнирности вершины
    /// Имеет текущее положение.
    /// Может проходить грань только в одном направлении, помечая проеденности вершины isMarked = -1;
    /// </summary>


    internal sealed class SubgraphFasetPair
    {
        internal readonly Physical2DGraph Subgraph;

        internal readonly Cycle<Physical2DGraphVertex> Faset;

        internal SubgraphFasetPair(Physical2DGraph subgraph, Cycle<Physical2DGraphVertex> faset)
        {
            Subgraph = subgraph;
            Faset = faset;
        }
    }



    public class Physical2DGraph : QuickGraph.UndirectedGraph<Physical2DGraphVertex, Physical2DGraphEdge<Physical2DGraphVertex>>
    {

        /// <summary>
        /// расстояние между точками 
        /// </summary>
        /// <param name="p1"></param>
        /// <param name="p1"></param>
        /// <returns></returns>
        internal double VectorMath_Distance(Point p1, Point p2)
        {

            //Vector sub = Vector.Subtract((Vector)p1, (Vector)p2);
            //// sub.Length ?!
            //return Math.Sqrt(Vector.Multiply(sub, sub));            
            return Vector.Subtract((Vector)p1, (Vector)p2).Length;
        }

        ///// <summary>
        ///// Развёртывание графа на плоскости, для отображения. Используется "физическая модель"
        ///// </summary>
        ///// <param name="iterationCount"></param>
        //public void Unfold(NumericalMethodParametres parametres)
        //{

        //    //NumericalMethodParametres parametres = new NumericalMethodParametres();
        //    //// TODO: использовать в качестве входного параметра NumericalMethodParametres parametres
        //    //parametres.OneProcessIterationCount = 20;
        //    //parametres.OneIterationTimeStep = 0.20;

        //    //for (int i = 0; i < iterationCount - 1; i++)
        //    //{
        //    //    this.ProcessUnfoldingModel(parametres);
        //    //}

        //    //parametres.DontCalculateRemoteNodesGravitation = false;
        //    //this.ProcessUnfoldingModel(parametres);

            
        //}

        ///// <summary>
        ///// Развёртывание графа на плоскости, для отображения. Используется "физическая модель".
        ///// Перед развёртыванием начальное положение узлов устанавливается в окрестности точки centerPosition
        ///// </summary>
        ///// <param name="iterationCount"></param>
        //public void Unfold(Point centerPosition, int iterationCount = 20)
        //{
        //    Random rnd = new Random((Int32)DateTime.Now.Ticks);
        //    foreach (Physical2DGraphVertex v in this.Vertices)
        //    {
        //        v.Position = new Point(centerPosition.X * 0.5 + this.PhysicalModelParametres.FreeConnectionLength * rnd.NextDouble(),
        //                               centerPosition.Y * 0.5 + this.PhysicalModelParametres.FreeConnectionLength * rnd.NextDouble()
        //                                );
        //    };



        //    this.Unfold(iterationCount);
        //}



        /// <summary>
        /// Поиск общего ребра двух вершин (предполагается, что паралельные рёбра отсутсвуют)
        /// </summary>
        /// <param name="fromVertex"></param>
        /// <param name="toVertex"></param>
        /// <returns></returns>
        public bool FindCommonEdge(Physical2DGraphVertex fromVertex, Physical2DGraphVertex toVertex, out Physical2DGraphEdge<Physical2DGraphVertex> result)
        {
            result = null;

            try
            {
                result = this.AdjacentEdges(fromVertex).First
                                    (delegate(Physical2DGraphEdge<Physical2DGraphVertex> edge)
                                        {
                                            return (edge.GetOtherVertex(fromVertex) == toVertex);
                                        }
                                    );
            }
            catch (System.InvalidOperationException)
            {
            }


            return (result != null);
        }

        /// <summary>
        /// Поиск минимального двунаправленного цикла (ребро можно пройти в двух направлениях).
        /// При этом двунаправленный цикл может быть вырожденным (деревом, все рёбра пройдены в двух направлениях) или частично вырожденным - быть односвязным с вершиной,
        /// лишь некоторые рёбра пройдены в двух направлениях.
        /// directionEdge - указывает направление поиска
        /// Если directionEdge == null, то поиск минимального цикла начинается с вершины rootVertex по всем направлениям
        /// </summary>
        /// <param name="rootVertex"></param>
        /// <param name="directionEdge"></param>
        /// <param name="cycle"></param>
        /// <returns></returns>
        /// <remarks>
        /// Используется:
        /// 1) в начале при инициализации итеративного алгоритма планаризации для поиска минимального цикла (либо детектирования дерева)
        /// 2) в итерации алгоритма планаризации при встраивании субграфа в грань для поиска внутреннего минимального цикла (с учётом терминальных узлов грани)
        /// </remarks>
        public bool FindShortestTwoWayCycle(Physical2DGraphVertex rootVertex, Physical2DGraphEdge<Physical2DGraphVertex> directionEdge, out Cycle<Physical2DGraphVertex> cycle)
        {
            bool isUnconfluentCycleFound = false;
            cycle = null;

            ClearPassedMarks();

            /*
             * Будем использовать "путешественников". Путешественники хранят позицию (TVertex position), в которой они находятся и пройденный путь  (OrientedCycle passed) из корневого узла
             * 
             * Начальные условия:
             * Создаём "Путешественника", в вершине directionEdge, путь: rootVertex, directionEdge.
             * Добавляем его в коллекцию путешественников.
             * 
             * 
             * Для каждого путешественника Walker (пока они есть):
             *  Находим все рёбра edgesCanWalked  по которым ещё можно ходить.
             *      условия: (old)это ребро - не начальное (directionEdge)
             *               (new)это ребро - не соседнее (directionEdge)
             *               ребро не помечено, как пройденное из вершины Walker.position (без разницы кем - ведь мы ищем минимальный путь)
             *      
             *  для каждого ребра edge is edgesCanWalked:
             *      создаём клон Walker'a (для первого ребра можно не создавать, а использовать существующий) newWalker, копируя пройденный путь
             *      перемещаем newWalker в узел противолежащий текущему положению через грань edge:
             *          edge помечается как пройденная от узла (текущее положение)
             *          новое положение - противолежащий узел
             *          путь увеличивается на edge
             *      если новое положение newWalker == rootVertex - то пройденный путь и есть икомый минимальный ориентированный цикл.Выходим из цикла
             *  конец цикла по edge
             *  если некуда было идти (рёбра не найдены) - то удаляем путешественника из коллекции путешественников
             * если путешественники закончились, а минимальный цикл не найден - возвращаем "минимальный цикл из rootVertex по направлению directionEdge несуществует"
             *          
             *      
             */

            // список путешественников:
            List<TwoDirectionWalker> walkers = new List<TwoDirectionWalker>();

            // создаём стартового путешественника
            TwoDirectionWalker firstWalker = new TwoDirectionWalker(this, rootVertex);

            // "пускаем" путешественника вдоль стартового ребра, если оно указано:
            if (directionEdge != null)
            {
                firstWalker.StepOver(directionEdge);
            }

            walkers.Add(firstWalker);

            // вспомогательный список: добавленные путешественники:
            List<TwoDirectionWalker> walkersToAdd = new List<TwoDirectionWalker>();
            // вспомогательный список: удалённые путешественники:
            List<TwoDirectionWalker> walkersToDelete = new List<TwoDirectionWalker>();

            while ((walkers.Count > 0) && (isUnconfluentCycleFound == false))
            {
                // Для каждого путешественника Walker (пока они есть):
                for (int i = 0; i < walkers.Count; i++)
                {


                    TwoDirectionWalker walker = walkers[i];

                    /*  Находим все рёбра edgesCanWalked  по которым ещё можно ходить.
                     *      условия: это ребро - не начальное (directionEdge)
                     *               ребро не помечено, как пройденное из вершины Walker.position (без разницы кем - ведь мы ищем минимальный путь)
                     */
                    if (this.AdjacentEdges(walker.CurrentPosition).Any((delegate(Physical2DGraphEdge<Physical2DGraphVertex> edge)
                                        { return walker.CanStepOver(edge); }
                                    )))
                    {

                        List<Physical2DGraphEdge<Physical2DGraphVertex>> edgesCanToWalk =
                            //Enumerable.Where(
                                    this.AdjacentEdges(walker.CurrentPosition).Where(
                                    (delegate(Physical2DGraphEdge<Physical2DGraphVertex> edge)
                                        { return walker.CanStepOver(edge); }
                                    )
                                    ).ToList();

                        // для каждого ребра создаём клон путешественника и перемещаем его вдоль грани
                        // (для первой грани используем прежнего путешественника)
                        for (int edgeInd = 0; edgeInd < edgesCanToWalk.Count; edgeInd++)
                        {
                            TwoDirectionWalker newWalker = (TwoDirectionWalker)walker.Clone(); ;
                            walkersToAdd.Add(newWalker);

                            /*
                             * перемещаем newWalker в узел противолежащий текущему положению через грань edge:
                             *          edge помечается как пройденная от узла (текущее положение)
                             *          новое положение - противолежащий узел
                             *          путь увеличивается на edge
                             *      если новое положение newWalker == rootVertex - то пройденный путь и есть икомый минимальный ориентированный цикл.Выходим из цикла
                             */

                            newWalker.StepOver(edgesCanToWalk[edgeInd]);

                            // цикл либо замыкается начальной позицией либо терминальным узлом
                            if ((newWalker.CurrentPosition == rootVertex) || (this.TerminalVertices.Contains(newWalker.CurrentPosition)))
                            {
                                // Проверяем, есть ли уже точно такая же грань, но инвертированная
                                // при условии, что граф не является кольцом.
                                //if (this.IsIvertedCycleExists(newWalker.PassedPath))
                                //{
                                //}
                                //else
                                {
                                    // найден цикл. Раз он найден первым, то это минимальный цикл.
                                    isUnconfluentCycleFound = true;
                                    cycle = newWalker.PassedPath;
                                    cycle.isTree = false;
                                    break;
                                }
                            }
                        }

                        walkersToDelete.Add(walker);
                    }
                    else
                    {
                        // путешественник в тупике - просто удаляем его
                        walkersToDelete.Add(walker);
                    }



                }

                // удаляем путешественников, которым некуда идти
                foreach (TwoDirectionWalker w in walkersToDelete)
                {
                    walkers.Remove(w);
                }

                // добавляем новых путешественников:
                foreach (TwoDirectionWalker w in walkersToAdd)
                {
                    walkers.Add(w);
                }

                walkersToAdd.Clear();
                walkersToDelete.Clear();



            }

            if (!isUnconfluentCycleFound)
            {
                // весь граф обойдён (путники закончились попав в тупики), но цикл не найден: это дерево. Формируем вырожденный цикл дерево
                this.ClearPassedMarks();
                ConvertTreeToTwoWayCycle(rootVertex, directionEdge, out cycle);
            }



            return isUnconfluentCycleFound;

        }

        public bool FindShortestTwoWayCycle(Physical2DGraphVertex rootVertex, out Cycle<Physical2DGraphVertex> cycle)
        {
            return FindShortestTwoWayCycle(rootVertex, null, out cycle);
        }

        /// <summary>
        /// Преобразует дерево (часть графа)
        /// rootVertex - указывает вершину, с кото
        /// directionEdge - указывает направление поиска. Используется если нужно преобразовать лишь часть графа ("деревянную")
        /// Если directionEdge == null, то заполнение цикла начинается с вершины rootVertex по любому направлениям.
        /// Направлений может быть несколько, если rootVertext - корень и не листья дерева.
        /// </summary>
        /// <param name="rootVertex"></param>
        /// <param name="directionEdge"></param>
        /// <param name="cycle"></param>
        /// <returns></returns>
        public void ConvertTreeToTwoWayCycle(Physical2DGraphVertex rootVertex, Physical2DGraphEdge<Physical2DGraphVertex> directionEdge, out Cycle<Physical2DGraphVertex> cycle)
        {

            cycle = null;

            /*
             * Будем использовать "путешественника". Путешественники хранят позицию (TVertex position), 
             *  в которой они находятся и пройденный путь  (OrientedCycle passed) из корневого узла

             * 
             * Начальные условия:
             * Создаём "Путешественника", в вершине directionEdge, путь: rootVertex, directionEdge.                             
             * 
             * Очищаем все пометки пройденности.
             * 
             *                  
             *   Отталкиваясь от текущей вершины, находим все рёбра edgesCanWalked  по которым ещё можно ходить.
             *      условия: 
             *               ребро не помечено, как пройденное из вершины Walker.position 
             *                           *      
             *  Выбираем  ребро edge is edgesCanWalked
             *           (!) отдавае предпочтение тому ребру, которое ещё не обходилось в обоих направлениях:
             *           из узла нельзя двигатья по ребру в обратном направлении, если у этого ребра есть ребро ни разу не обойдённое
             *           
             *          edge помечается как пройденная от узла (текущее положение)
             *          новое положение - противолежащий узел
             *          путь увеличивается на edge
             *      если указан directionEdge (обходиться деревянный сегмент графа) и если новое положение newWalker == rootVertex 
             *            - то пройденный путь и есть икомый минимальный ориентированный цикл дереве.Выходим из цикла                 
             *      если  directionEdge не указан, то спокойно можем проходить через rootVertex
             *          
             *  конец цикла по edge
             *  если некуда больше идти (рёбра не найдены) - возвращаем процеднный путешественником путь
             *  (если некоторые рёбра остались не помеченными как пройденные хотя бы  в одном направлении, то это было не дерево)                  
             */

            // очищаем пометки пройденности:
            this.ClearPassedMarks();

            // создаём путешественника
            TwoDirectionWalker walker = new TwoDirectionWalker(this, rootVertex);

            // "пускаем" путешественника вдоль стартового ребра, если оно указано:
            if (directionEdge != null)
            {
                walker.StepOver(directionEdge);
            }




            /*  Находим все рёбра edgesCanWalked  по которым ещё можно ходить.
             *               ребро не помечено, как пройденное из вершины Walker.position 
                     
             */
            while (this.AdjacentEdges(walker.CurrentPosition).Any(delegate(Physical2DGraphEdge<Physical2DGraphVertex> edge)
                {
                    return !edge.IsPassedFrom(walker.CurrentPosition) && !edge.IsEmbededFrom(walker.CurrentPosition);
                }
                    )
                )
            {

                List<Physical2DGraphEdge<Physical2DGraphVertex>> edgesCanToWalk =
                    //Enumerable.Where(
                            this.AdjacentEdges(walker.CurrentPosition).Where(
                            (delegate(Physical2DGraphEdge<Physical2DGraphVertex> edge)
                            { return !edge.IsPassedFrom(walker.CurrentPosition) && !edge.IsEmbededFrom(walker.CurrentPosition); }
                            )
                            ).ToList();
                // Если среди edgeCanToWalk есть ребро не пройденное в обоих направлениях (т.е. не пройденное в обратном направлении и не пройденное в прямом), 
                // то выбираем это ребро для дальнейшего путешествия.
                // иначе вынуждены выбрать первое попавшееся ребро (уже пройденое в обратном направлении, но не в прямом)

                Physical2DGraphEdge<Physical2DGraphVertex> NextEdgeToWalk;

                if (edgesCanToWalk.Any((delegate(Physical2DGraphEdge<Physical2DGraphVertex> edge)
                    {

                        return !edge.IsPassedTo(walker.CurrentPosition) && !edge.IsEmbededTo(walker.CurrentPosition);
                    }
                        ))
                    )
                {
                    // есть ребро, ни разу не пройденное - его и выбираем для дальнейшено путешествия
                    NextEdgeToWalk = edgesCanToWalk.First((delegate(Physical2DGraphEdge<Physical2DGraphVertex> edge)
                            {

                                return !edge.IsPassedTo(walker.CurrentPosition) && !edge.IsEmbededTo(walker.CurrentPosition);
                            }
                        ));

                }
                else
                {
                    // нет ребра, ни разу не пройденного (все пройдены хотя бы в одном направлении, но не пройдены в прямом)
                    // берём первое попавшееся
                    NextEdgeToWalk = edgesCanToWalk.First();
                };


                walker.StepOver(NextEdgeToWalk);

                if ((directionEdge != null) && (walker.CurrentPosition == rootVertex))
                {
                    break;
                }

            };

            System.Diagnostics.Debug.Assert(walker.CurrentPosition == rootVertex);
            cycle = walker.PassedPath;
            cycle.isTree = true;

        }



        /// <summary>
        /// Планаризация графа. Возвращает список граней графа, которые могут быть уложены на плоскость, а также
        /// флаг - планарный граф или нет.
        /// Используется для генетического алгоритма поиска некоторых топологий
        /// </summary>
        /// <returns>Возвращает true, если граф планарный</returns>
        public bool Planarize()
        {
            // Алгоритм планаризации:
            // 1. Находим простой цикл. 
            //   1.1 Если нельзя найти - то перед нами дерево и мы возвращаем единственную его грань
            //   1.2 Если цикл найден - то  перед нами две грани (внутрицикла/вне цикла). В которые мы вписываем оставшуюся часть графа
            // 2. Укладываем очередную оставшуюся часть графа
            //   2.1 Определяем субграфы в оставшейся части графа
            //     2.1.2 Если субграфов больше нет, то выход
            //   2.2 для каждого субграфа определяем одну или несколько граней из уже уложеннных
            //     2.1.2 если остались субграфы, которые нельзя вместить ни в одну грань, то граф не планарныый.
            //   2.3 для каждой из пар ("субграф", "грань") - находим простой смежный цикл
            //      2.3.1 делим грань на две или одну (если субграф - дерево)
            // Переходим к пункту 2. 

            ClearEmededAndPassedMarks();


            // список уже уложеных граней (пока пустой)
            fasets = new List<Cycle<Physical2DGraphVertex>>();


            // 1. Пытаемся найти простой цикл
            // Начиная с первого попавшегося узла - ищем кратчайший замкнутый путь.
            // Если не удаётся - берём следующий узел
            // Если узлы закончились - то перед нами - дерево
            //bool isMinCycleFound = false;
            Cycle<Physical2DGraphVertex> cycle = null;

            ////foreach (Physical2DGraphVertex v in this.Vertices)
            ////{
            ////    // ищем кратчайший путь из узла V
            ////    this.ClearPassedMarks();
            ////    if (this.FindShortestTwoWayCycle(v, out cycle))
            ////    {
            ////        isMinCycleFound = true;
            ////        break;
            ////    }
            ////}

            if (this.AdjacentEdges(this.Vertices.First()).Count() == 0)
            {
                // граф либо не связный, либо состоит только из одного узла
                isPlanar = true;
            }
            else
            {


                this.FindShortestTwoWayCycle(this.Vertices.First(), this.AdjacentEdges(this.Vertices.First()).First(), out cycle);

                //if (cycle.IsTree)
                //{


                //    isPlanar = true;


                //    //Cycle<Physical2DGraphVertex> faset;
                //    //Physical2DGraphVertex v = this.Vertices.First();

                //    //ConvertTreeToTwoWayCycle(v, null, out faset);
                //    cycle.MarkAllEdgesAsEmbedded();
                //    fasets.Add(cycle);

                //}
                //else
                {
                    //   1.2 Если цикл найден - то  перед нами две грани (внутрицикла/вне цикла). В которые мы вписываем оставшуюся часть графа

                    Cycle<Physical2DGraphVertex> faset1 = cycle;
                    faset1.MarkAllEdgesAsEmbedded();
                    fasets.Add(faset1);

                    // получаем инвертированный цикл из исходного. Если исходный цикл - дерево, то инвертированный цикл будет пустым
                    Cycle<Physical2DGraphVertex> faset2 = (Cycle<Physical2DGraphVertex>)cycle.Clone();
                    faset2.Invert();
                    faset2.MarkAllEdgesAsEmbedded();
                    if (faset2.Count != 0)
                    {
                        fasets.Add(faset2);
                    }

                    // утв.: Найдены первые две грани, можно запускать итеративный алгоритм планеризации

                    // 2. Укладываем очередную оставшуюся часть графа

                    // 2.1 Определяем субграфы в оставшейся части графа
                    // (субграф храним в виде нового экземпляра объекта граф + храним список терминальных узлов)


                    List<Physical2DGraph> subgraphs = this.FindSubgraphes(fasets);

                    //MessageBox.Show(String.Format("количество субграфов: {0}", subgraphs.Count));

                    isPlanar = true;

                    // для отладки:
                    int maxiterationCount = 10000;
                    int stepsCount = 0;
                    while ((subgraphs.Count > 0) && (stepsCount < maxiterationCount))
                    {
                        stepsCount++;
                        //   2.2 для каждого субграфа определяем одну или несколько граней из уже уложеннных
                        //     2.1.2 если остались субграфы, которые нельзя вместить ни в одну грань, то граф не планарныый.
                        #region Старая версия некорректно работающая для "конфликтных" субграфов которые могут потенциально вмещаться в одну и туже грань
                        ////// Алгоритм: для субграфа выбирает ту грань, из уложенных и ранее не выбранных для другого субграфа,
                        ////// которой принадлежат все вершины субграфа. Таки граней может быть несколько (в т.ч. ниодной)
                        ////// (алгоритм можно оптимизировать, позволяя вмещать несколько неконфликтующих субграфов в одну и ту же грань)


                        //List<Cycle<Physical2DGraphVertex>> alreadySelectedFasets = new List<Cycle<Physical2DGraphVertex>>();



                        //foreach (Physical2DGraph subgraph in subgraphs)
                        //{

                        //    HashSet<Cycle<Physical2DGraphVertex>> commonFasets = null;

                        //    // для каждой из терминальных вершин графа получаем множество уложенных граней. Пересечение таких множеств и
                        //    // будет искомым множеством граней, в которые можно уложить граф.
                        //    foreach (Physical2DGraphVertex v in subgraph.TerminalVertices)
                        //    {
                        //        HashSet<Cycle<Physical2DGraphVertex>> fasetsOfVertex = new HashSet<Cycle<Physical2DGraphVertex>>
                        //            (fasets.Where
                        //                (delegate(Cycle<Physical2DGraphVertex> c)
                        //                    {
                        //                        return c.Vertices.Contains(v);
                        //                    }
                        //                )
                        //            ) ;

                        //        if (commonFasets == null)
                        //        {
                        //            // первая терминальня вершина из рассматриваемых (так как каждая терминальная вершина принадлежит хотя бы одной грани или даже двум)
                        //            // - заполняем множество найденых граней
                        //             commonFasets = new HashSet<Cycle<Physical2DGraphVertex>>(fasetsOfVertex);
                        //        }
                        //        else
                        //        {
                        //            commonFasets.IntersectWith(fasetsOfVertex);
                        //        }
                        //    }


                        //    //MessageBox.Show(commonFasets.ToString());
                        //    // выбиарем первую попавшуюся грань из найденного множества общих граней, при условии что она не выбрана.
                        //    // (добавляем в список выбранных)

                        //    if (commonFasets.Any(delegate(Cycle<Physical2DGraphVertex> f) { return !alreadySelectedFasets.Contains(f); }))
                        //    {
                        //        Cycle<Physical2DGraphVertex> selectedFaset = commonFasets.First(delegate(Cycle<Physical2DGraphVertex> f) { return !alreadySelectedFasets.Contains(f); });                                                        
                        //        alreadySelectedFasets.Add(selectedFaset);
                        //        EmbedSubgraphToFaset(subgraph, selectedFaset, fasets);
                        //    }
                        #endregion


                        // список пар "субграф-грань" которые потенциально могут быть обработаны (субграф вписан в граф)
                        List<SubgraphFasetPair> subgraphesAndFasetsForEmbeddingList = new List<SubgraphFasetPair>();

                        // 2.1. Находим для каждого графа - список граней, в которые он может потенциально вместиться ("вписаться")
                        foreach (Physical2DGraph subgraph in subgraphs)
                        {

                            //MessageBox.Show(String.Format( "Subgraph:", subgraph.ToString()));
                            HashSet<Cycle<Physical2DGraphVertex>> fasetsForEmbeddingSet = null;

                            // для каждой из терминальных вершин графа получаем множество уложенных граней. Пересечение таких множеств и
                            // будет искомым множеством граней, в которые можно уложить граф.
                            foreach (Physical2DGraphVertex v in subgraph.TerminalVertices)
                            {
                                HashSet<Cycle<Physical2DGraphVertex>> fasetsOfVertex = new HashSet<Cycle<Physical2DGraphVertex>>
                                    (fasets.Where
                                        (delegate(Cycle<Physical2DGraphVertex> c)
                                            {
                                                return c.Vertices.Contains(v);
                                            }
                                        )
                                    );

                                if (fasetsForEmbeddingSet == null)
                                {
                                    // первая терминальня вершина из рассматриваемых (так как каждая терминальная вершина принадлежит хотя бы одной грани или даже двум)
                                    // - заполняем множество найденых граней
                                    fasetsForEmbeddingSet = new HashSet<Cycle<Physical2DGraphVertex>>(fasetsOfVertex);
                                }
                                else
                                {
                                    fasetsForEmbeddingSet.IntersectWith(fasetsOfVertex);
                                }
                            }

                            foreach (Cycle<Physical2DGraphVertex> faset in fasetsForEmbeddingSet)
                            {
                                subgraphesAndFasetsForEmbeddingList.Add(new SubgraphFasetPair(subgraph, faset));
                            }


                        }




                        /* 2.1.2.Обрабатываем получившиеся пары "субграф-грань":
                         * 2.1.2.0 если список пустой, а субграфы таки осталисть - то граф не планарный. Выход.
                         * 2.1.2.1 Обрабатываем односвязанные субграфы: Последовательно выбираем те из пар "субграф-грань", у которых количество терминальных узлов в субграфе = 1
                         *  При каждом выборе пары "субграф-грань" - вмещаем субграф в грань и у даляем те пары "субграф-грань" из исходного списка, которые содержат обработанную
                         *  грань или граф (за итерацию обрабатывается только одна грань
                         * 2.1.2.2 Обрабатываем неоднносвязанные/неконфликтующие субграфы: Последовательно выбираем те из пар "субграф-грань" грань которых содержит только один субграф
                         *  При каждом выборе пары - аналогично п. 2.1
                         * 2.1.2.3 Обрабатываем конфликтные субграфы: из оставшегося списка для каждого субграфа на угад выбираем подходящую грань                     
                         */


                        if (subgraphesAndFasetsForEmbeddingList.Count == 0)
                        {
                            isPlanar = false;
                            break;
                        }

                        // temp: оставляем самую большую  грань.

                        var maxPair = (from x in subgraphesAndFasetsForEmbeddingList orderby x.Faset.Count descending select x).First();

                        subgraphesAndFasetsForEmbeddingList.RemoveAll(x => x != maxPair);

                        #region Обрабатываем однозвязанные графы

                        // при этом для лучшей определённости и односвязанные графы добавляем в самую большую грань

                        var pairsQuery = from x in subgraphesAndFasetsForEmbeddingList
                                         where x.Subgraph.TerminalVertices.Count == 1
                                         orderby x.Faset.Count descending
                                         select x;



                        while (pairsQuery.Count() > 0)
                        {

                            SubgraphFasetPair pairForEmbedding = pairsQuery.First();

                            EmbedSubgraphToFaset(pairForEmbedding.Subgraph, pairForEmbedding.Faset, fasets);

                            //var pairsToRemove = from x in subgraphesAndFasetsForEmbeddingList
                            //                    where (x.Subgraph == pair.Subgraph) || (x.Faset == pair.Faset)                                        
                            //                    select x;
                            //subgraphesAndFasetsForEmbeddingList.RemoveAll(delegate(SubgraphFasetPair pair) { return (pair.Subgraph.Equals(pairForEmbedding.Subgraph)) || (pair.Faset.Equals(pairForEmbedding.Faset)); });


                            /// для обеспечения Silverlight совместимости:
                            //subgraphesAndFasetsForEmbeddingList.RemoveAll(delegate(SubgraphFasetPair pair) { return (pair.Subgraph.Equals(pairForEmbedding.Subgraph)); });
                            var pairsWithSubgrapQuery = from pair in subgraphesAndFasetsForEmbeddingList where pair.Subgraph.Equals(pairForEmbedding.Subgraph) select pair;
                            foreach (SubgraphFasetPair p in pairsWithSubgrapQuery.ToList<SubgraphFasetPair>())
                            {
                                subgraphesAndFasetsForEmbeddingList.Remove(p);
                            }



                            pairsQuery = from x in subgraphesAndFasetsForEmbeddingList
                                         where x.Subgraph.TerminalVertices.Count == 1
                                         orderby x.Faset.Count descending
                                         select x;

                        }
                        #endregion


                        #region обрабатываем не обдосвязанные/неконфликтующие графы
                        var NonconflictPairsQuery =
                            from part in
                                (from x in subgraphesAndFasetsForEmbeddingList
                                 group x by x.Faset into partition
                                 select new
                                 {
                                     Faset = partition.Key,
                                     Group = partition,
                                     Count = partition.Count()
                                 })
                            where part.Count == 1
                            orderby part.Faset.Count descending
                            select part
                            ;



                        while (NonconflictPairsQuery.Count() > 0)
                        {

                            SubgraphFasetPair pairForEmbedding = NonconflictPairsQuery.First().Group.First();

                            EmbedSubgraphToFaset(pairForEmbedding.Subgraph, pairForEmbedding.Faset, fasets);


                            //subgraphesAndFasetsForEmbeddingList.RemoveAll(delegate(SubgraphFasetPair pair) { return (pair.Subgraph.Equals(pairForEmbedding.Subgraph)) || (pair.Faset.Equals(pairForEmbedding.Faset)); });


                            var pairsToRemove = from pair in subgraphesAndFasetsForEmbeddingList where pair.Subgraph.Equals(pairForEmbedding.Subgraph) || pair.Faset.Equals(pairForEmbedding.Faset) select pair;
                            foreach (SubgraphFasetPair p in pairsToRemove.ToList<SubgraphFasetPair>())
                            {
                                subgraphesAndFasetsForEmbeddingList.Remove(p);
                            }






                            NonconflictPairsQuery = from part in
                                                        (from x in subgraphesAndFasetsForEmbeddingList
                                                         group x by x.Faset into partition
                                                         select new
                                                         {
                                                             Faset = partition.Key,
                                                             Group = partition,
                                                             Count = partition.Count()
                                                         })
                                                    where part.Count == 1
                                                    orderby part.Faset.Count descending
                                                    select part;

                        }

                        #endregion

                        #region Обрабатывае "конфликтные" субграфы

                        var ConflictPairsQuery = from x in subgraphesAndFasetsForEmbeddingList orderby x.Faset.Count descending select x;


                        if (ConflictPairsQuery.Count() > 0)
                        {

                            SubgraphFasetPair pairForEmbedding = ConflictPairsQuery.First();

                            EmbedSubgraphToFaset(pairForEmbedding.Subgraph, pairForEmbedding.Faset, fasets);

                            //var pairsToRemove = from x in subgraphesAndFasetsForEmbeddingList
                            //                    where (x.Subgraph == pair.Subgraph) || (x.Faset == pair.Faset)                                        
                            //                    select x;
                            // subgraphesAndFasetsForEmbeddingList.RemoveAll(delegate(SubgraphFasetPair pair) { return (pair.Subgraph.Equals(pairForEmbedding.Subgraph)) || (pair.Faset.Equals(pairForEmbedding.Faset)); });

                            var pairsToRemove = from pair in subgraphesAndFasetsForEmbeddingList where (pair.Subgraph.Equals(pairForEmbedding.Subgraph)) || (pair.Faset.Equals(pairForEmbedding.Faset)) select pair;
                            foreach (SubgraphFasetPair p in pairsToRemove.ToList<SubgraphFasetPair>())
                            {
                                subgraphesAndFasetsForEmbeddingList.Remove(p);
                            }


                            //--pairsQuery = from x in subgraphesAndFasetsForEmbeddingList select x;
                            ConflictPairsQuery = from x in subgraphesAndFasetsForEmbeddingList orderby x.Faset.Count descending select x;

                        }

                        #endregion

                        // 2.3. поиск следующих субграфов:
                        subgraphs = this.FindSubgraphes(fasets);

                    }
                }
            }

            // (заглушка)


            ClearEmededAndPassedMarks();

            /// Пост обработка и кэширование

            // 1. запоминаем длину минимального цикла, для каждой грани
            foreach (Cycle<Physical2DGraphVertex> f in fasets)
            {
                f.MinCycleLengthCache();

                foreach (Physical2DGraphVertex v in f.Vertices)
                {
                    if (!v.fasets.Contains(f))
                    {
                        v.fasets.Add(f);
                    }
                }
            }

            // 2. для каждой вершины - запоминаем к каким граням она пренадлежит (количество которых =  степени вершины)



            isPlanar = (this.VertexCount - this.EdgeCount + fasets.Count == 2); // Формула Эейлера: V(G) | − | E(G) | + | F(G) | = 2,

            return isPlanar;
        }

        /// <summary>
        /// Размещён ли граф на 2D плоскости без пересечений рёбер 
        /// </summary>
        /// <returns></returns>
        /// <remarks>
        /// Планарный граф всегда можно уложить на плоскости без пересечений, но можно уложить и "кое-как" - с пересечениями. Не планарный граф нельзя уложить на плоскости
        /// </remarks>
        /// <seealso cref="IsPlanar"/>
        /// <seealso cref="IsPlanarized"/>
        public bool IsPlanned()
        {
            // Unfold(new Point(0, 0), 4); // затратно!

            Physical2DGraphEdge<Physical2DGraphVertex>[] edges = this.Edges.ToArray();

            bool result = true;

            for (int i = 0; i < edges.Length - 1; i++)
            {
                for (int j = i + 1; j < edges.Length; j++)
                {
                    if ((edges[i].Source != edges[j].Target) && (edges[i].Source != edges[j].Source)
                       && (edges[i].Target != edges[j].Target) && (edges[i].Target != edges[j].Source))
                    {

                        Point p11 = edges[i].Source.Position;
                        Point p12 = edges[i].Target.Position;

                        Point p21 = edges[j].Source.Position;
                        Point p22 = edges[j].Target.Position;


                        if (Geometry.Geometer2D.IsLinesCross(p11, p12, p21, p22))
                        {
                            result = false;
                            break;

                        }
                    }
                }
                if (result == false)
                {
                    break;
                }

            }
            return result;
        }



        /// <summary>
        /// возвращает true, если граф двусвязный (без шарниров)
        /// </summary>
        /// <returns></returns>
        public bool IsBeconnected()
        {
            bool result = true;

            /*
             * Алгоритм: проверяем каждую вершину - является ли она шарниром.
             * Вершин
             * 
             */
            foreach (Physical2DGraphVertex vertex in this.Vertices)
            {
                if (IsVertexHinge(vertex))
                {
                    result = false;
                    break;
                }
            }


            return result;
        }

        /// <summary>
        /// возвращает true, если указанная вершина является шарниром
        /// </summary>
        /// <param name="vertex"></param>
        /// <returns></returns>
        public bool IsVertexHinge(Physical2DGraphVertex vertex)
        {
            bool result = false;


            /*
             * Алгоритм: 
             * 1. очищаем пометки всех вершин
             * 2. vertex метим как "-1";
             * 3. метим всех её соседей в "1" и запоминаем список
             * 4. для каждого из её соседей, проверяем достижимы ли остальные соседи, если нельзя посещать вершину vertex
             * 4.1 отправляем путника от соседа.
             *  - путник может посещать только 0 и 1. -1 не может посещать. Также не может посещать пройденные рёбра
             *  - на развилке путник размножается (и путники 
             *  - если доходит до вершины помеченной в 1 - то она исключачается из списка соседей для достижения (путник может обходит граф дальше)
             *  - если путнику некуда идти - то он уничтожается
             *  - если все путники закончились - то финальная проверка.
             *  - финальная проверка:
             *      - если остались ещё в списке недостигнутые соседи - знаит vertex - является шарниром
             */

            if (this.AdjacentEdges(vertex).Count() == 1)
            {
                result = false;
                return result;
            }



            this.ClearVertexMarks();

            vertex.isMarkedForAlgorithms = -1;


            List<Physical2DGraphVertex> neighbToReach = new List<Physical2DGraphVertex>();

            foreach (Physical2DGraphEdge<Physical2DGraphVertex> edge in this.AdjacentEdges(vertex))
            {
                Physical2DGraphVertex opp = edge.GetOtherVertex(vertex);
                if (!neighbToReach.Contains(opp))
                {
                    opp.isMarkedForAlgorithms = 1;
                    neighbToReach.Add(opp);
                }
            };



            Physical2DGraphVertex[] vertices = this.Vertices.ToArray<Physical2DGraphVertex>();
            List<Physical2DGraphVertex> wolkers = new List<Physical2DGraphVertex>();

            // Стартуем из одной вершины соседней "шарниру" и пробуем достичь все остальных вершин.
            Physical2DGraphVertex startVertex = neighbToReach.First();
            neighbToReach.Remove(startVertex);
            startVertex.isMarkedForAlgorithms = -1;
            wolkers.Add(startVertex);

            while ((wolkers.Count > 0) && (neighbToReach.Count() > 0))
            {
                Physical2DGraphVertex currentNode = wolkers.First<Physical2DGraphVertex>();

                foreach (var edgeToNeighbor in this.AdjacentEdges(currentNode))
                {
                    Physical2DGraphVertex neighbor = edgeToNeighbor.GetOtherVertex(currentNode);

                    // если этот узел neighbor ещё не помечен как обойдённый
                    if (neighbor.isMarkedForAlgorithms >= 0)
                    {

                        if (neighbor.isMarkedForAlgorithms == 1)
                        {
                            neighbToReach.Remove(neighbor);
                        }

                        neighbor.isMarkedForAlgorithms = -1;
                        wolkers.Add(neighbor);

                    }

                }
                wolkers.Remove(currentNode);
            }

            // финальная проверка
            if (neighbToReach.Count() > 0)
            {
                // весь граф обойдён, но не все вершины достигнуты
                result = true;
            }



            return result;
        }

        private void ClearVertexMarks()
        {
            foreach (Physical2DGraphVertex v in Vertices)
            {
                v.isMarkedForAlgorithms = 0;
            }
        }






        /// <summary>
        /// Вмещаем субграф в уже уложенную грань. По сути преобразуем их для передачи на следующую итерацию.
        /// В результате "вмещения" количество уложенных рёбер возрастает, а уложенная грань может разделиться на две уложенных
        /// /// </summary>
        /// <param name="subgraph"></param>
        /// <param name="faset"></param>
        private void EmbedSubgraphToFaset(Physical2DGraph subgraph, Cycle<Physical2DGraphVertex> faset, List<Cycle<Physical2DGraphVertex>> fasets)
        {
            /*
             * Для "вмещения" достаточно найти "внутренний"  путь, начиная с любой терминальной вершины, заканчивая другой терминальной вершиной          
             * Если субграф может имеет только одну терминальную вершину (дерево или односвязный субграф), грань лишь дополняется новыми внутренними рёбрами.
             * Иначе, грань разделяется на две, границей между нимя является найденный путь.
             * 
             * Для поиска внутреннего кратчайшего пути, можно использовать функцию FindShortestTwoWayCycle, обобщив её для случая непустых терминальных узлов
             * 
             */
            //MessageBox.Show(String.Format(" Embed subgraph {0} to faset {1}", subgraph, faset));
            // Ищем внутренний цикл, начиная с любого терминального узла subgraph'а
            Cycle<Physical2DGraphVertex> path;
            Physical2DGraphVertex firstTerminalVertex = subgraph.TerminalVertices.First();
            if (subgraph.FindShortestTwoWayCycle(firstTerminalVertex, out path))
            {
                // цикл найден, причём начальный и конечный узел path - это терминальные узлы.
                if (path.Vertices.First().Equals(path.Vertices.Last()))
                {
                    /* Если начальный и конечный узел цикла path - один и тот же узел, то 
                     *  следует модифицировать грань faset, вставив (и отметить как уложенный на плоскость) туда новый найденный путь path.
                     * 
                     */

                    //MessageBox.Show(path.ToString());
                    faset.InsertAfterFirstVertex(firstTerminalVertex, path);
                    faset.MarkAllEdgesAsEmbedded();

                    path.Invert();
                    if (path.Count > 0)
                    {
                        fasets.Add(path);
                        path.MarkAllEdgesAsEmbedded();
                    }




                }
                else
                {
                    //MessageBox.Show(String.Format(" Embed path {0} to faset {1}", path, faset));
                    /* общий случай - исходная грань делиться найденным циклом на две
                     * первая грань F1 = Path + F[Path_end --> Path_start], склейка path и участка исходной грани от конечного до начально узла path
                     * вторая грань F2 = -path + F[Path_start --> Path_end], склейка инвертированной path и участка исходной грани от начального до конечного узла path
                     */
                    Cycle<Physical2DGraphVertex> faset2 = (Cycle<Physical2DGraphVertex>)faset.Clone();

                    Physical2DGraphVertex lastVertex = path.Vertices.Last();

                    faset.Replace(firstTerminalVertex, lastVertex, path);
                    faset.MarkAllEdgesAsEmbedded();

                    path.Invert();
                    faset2.Replace(lastVertex, firstTerminalVertex, path);
                    faset2.MarkAllEdgesAsEmbedded();
                    fasets.Add(faset2);




                }

            }
            else
            {
                /* цикл не найден. Т.е. subgraph является деревом. Поэтому следует обойти дерего и 
                 * модифицировать грань faset, вставив туда новый найденный путь
                 */
                // модифицируем цикл faset в точке firstTerminalVertex вставляя туда path (дерево)
                // 

                // 1. ищем позцию грань, которая заканчивается(!) терминальным узлом firstTerminalVertex
                faset.InsertAfterFirstVertex(firstTerminalVertex, path);

                faset.MarkAllEdgesAsEmbedded();
            }


        }



        /// <summary>
        /// Поиск субграфов относительно уже уложенной части графа
        /// Уже уложенные рёбра помечены как уложенные.
        /// </summary>
        /// <param name="fasets"></param>
        /// <returns></returns>
        private List<Physical2DGraph> FindSubgraphes(List<Cycle<Physical2DGraphVertex>> fasets)
        {
            /* Алгоритм поиска субграфов:
             * 1. Для всех уже уложенных рёбер находим узлы, которые связаны с ещё не уложенными рёбрами (терминальные узлы -
             * каждый терминальный узел может принадлежать к нескольким субграфам)            
             * 2. Если есть такие узлы, то:                  
             *      для каждого ещё не уложенного ребра E терминального узла, и ещё не помеченного, как добавленного в субграф (как пройденный путешественником):
             *          2.1- создаём новый субграф, добавляем его в список найденых графов
             *          2.2 - запускаем "Путешественников" из терминального узла в ребро E.
             *            -- этот первый терминальный узел добавляем в список терминальный узлов субграфа
             *          2.3- обходим все рёбра пока есть ещё куда идти. Путешественники заканчивают путь, если 
             *              - попадут в тупик 
             *              -- или закончаться все не обойдённые рёбра 
             *              -- достигнут терминального узла
             *                  -- терминальный узел добавляем в список терминальных узлов субграфа
             *          - при этом рёбра помечаются как добавленные в субграф и добавляются в созданный новый (очередной) субграф
             *          
             * 3. в итоге на выходе мы получили cписок субграфов
             */

            List<Physical2DGraph> result = new List<Physical2DGraph>();

            ClearPassedMarks();

            // 1. Для всех уже уложенных рёбер находим узлы, которые связаны с ещё не уложенными рёбрами (терминальные узлы -
            // каждый терминальный узел может принадлежать к нескольким субграфам)            



            foreach (Cycle<Physical2DGraphVertex> faset in fasets)
            {
                foreach (Physical2DGraphVertex vertex in faset.Vertices)
                {

                    foreach (Physical2DGraphEdge<Physical2DGraphVertex> edge in
                            this.AdjacentEdges(vertex).Where
                                (
                                    delegate(Physical2DGraphEdge<Physical2DGraphVertex> edge)
                                    {
                                        // при поиске очередного субграфа, пройденные рёбра и вследующий раз не должны попадать в выборку 
                                        return ((!edge.IsEmbeded())
                                                && (!edge.IsPassed()));
                                    }))
                    {

                        Physical2DGraph subgraph = GetSubgraphOnDirection(vertex, edge);
                        result.Add(subgraph);
                    };


                };
            }


            return result;
        }

        Physical2DGraph GetSubgraphOnDirection(Physical2DGraphVertex rootVertex, Physical2DGraphEdge<Physical2DGraphVertex> directionEdge)
        {
            Physical2DGraph result = new Physical2DGraph();


            /*
             * Будем использовать "путешественников". Путешественники хранят позицию (TVertex position), в которой они находятся и пройденный путь  (OrientedCycle passed) из корневого узла
             * 
             * Начальные условия:
             * Создаём "Путешественника", в вершине directionEdge, путь: rootVertex, directionEdge.
             * "Проходим" им
             * Добавляем его в коллекцию путешественников.
             * 
             * 
             * Для каждого путешественника Walker (пока они есть):
             *  Находим все рёбра edgesCanWalked  по которым ещё можно ходить.
             *      условия: это ребро - не начальное (directionEdge)
             *               ребро не помечено, как пройденное (без разницы кем и в каком направлении - ведь мы ищем минимальный путь)
             *               (new: GetSubgraphOnDirection)  текущее положение не является терминальным узлом (терминальность можно определит по правилу: хотя бы одно его ребро уже уложено)
             *  для каждого ребра edge is edgesCanWalked:
             *      создаём клон Walker'a (для первого ребра можно не создавать, а использовать существующий) newWalker, копируя пройденный путь
             *      перемещаем newWalker в узел противолежащий текущему положению через грань edge:
             *          edge помечается как пройденная от узла (текущее положение)
             *          новое положение - противолежащий узел
             *          путь увеличивается на edge
             *          (new: GetSubgraphOnDirection) - добавляем ребро в субграф         
             *      
             *  конец цикла по edge
             *  если некуда идти (рёбра не найдены) - то удаляем путешественника из коллекции путешественников
             * если путешественники закончились, возращаем "собранный" субграф
             *      
             */

            result.TerminalVertices.Add(rootVertex);

            // список путешественников:
            List<TwoDirectionWalker> walkers = new List<TwoDirectionWalker>();

            // создаём стартового путешественника
            TwoDirectionWalker firstWalker = new TwoDirectionWalker(this, rootVertex);

            // "пускаем" путешественника вдоль стартового ребра, если оно указано:
            if (directionEdge != null)
            {
                firstWalker.StepOver(directionEdge);
                result.AddVerticesAndEdge(directionEdge);
            }

            walkers.Add(firstWalker);

            // вспомогательный список: добавленные путешественники:
            List<TwoDirectionWalker> walkersToAdd = new List<TwoDirectionWalker>();
            // вспомогательный список: удалённые путешественники:
            List<TwoDirectionWalker> walkersToDelete = new List<TwoDirectionWalker>();

            while ((walkers.Count > 0))
            {
                // Для каждого путешественника Walker (пока они есть):
                for (int i = 0; i < walkers.Count; i++)
                {

                    TwoDirectionWalker walker = walkers[i];

                    /*  Находим все рёбра edgesCanWalked  по которым ещё можно ходить.
                     *      ребро не помечено, как пройденное (без разницы кем и в каком направлении - ведь мы ищем минимальный путь)
                     *      (new: GetSubgraphOnDirection)  текущее положение не является терминальным узлом (терминальность можно определит по правилу: хотя бы одно его ребро уже уложено)
                     */
                    bool PositionIsTerminal = this.AdjacentEdges(walker.CurrentPosition).Any(delegate(Physical2DGraphEdge<Physical2DGraphVertex> edge) { return edge.IsEmbeded(); });
                    bool PositionHaveUnpassedEdges = this.AdjacentEdges(walker.CurrentPosition).Any(delegate(Physical2DGraphEdge<Physical2DGraphVertex> edge) { return !edge.IsPassed(); });
                    if (PositionHaveUnpassedEdges && !PositionIsTerminal)
                    {

                        List<Physical2DGraphEdge<Physical2DGraphVertex>> edgesCanToWalk =
                                    this.AdjacentEdges(walker.CurrentPosition).Where(delegate(Physical2DGraphEdge<Physical2DGraphVertex> edge) { return !edge.IsPassed(); }).ToList();


                        // для каждого ребра создаём клон путешественника и перемещаем его вдоль грани
                        // (для первой грани используем прежнего путешественника)
                        for (int edgeInd = 0; edgeInd < edgesCanToWalk.Count; edgeInd++)
                        {
                            TwoDirectionWalker newWalker = (TwoDirectionWalker)walker.Clone(); ;
                            walkersToAdd.Add(newWalker);

                            /*
                                * перемещаем newWalker в узел противолежащий текущему положению через грань edge:
                                *          edge помечается как пройденная от узла (текущее положение)
                                *          новое положение - противолежащий узел
                                *          путь увеличивается на edge
                                *      (new: GetSubgraphOnDirection) - добавляем ребро в субграф        
                                */

                            newWalker.StepOver(edgesCanToWalk[edgeInd]);
                            result.AddVerticesAndEdge(edgesCanToWalk[edgeInd]);

                        }

                        walkersToDelete.Add(walker);
                    }
                    else
                    {
                        // путешественник в тупике - просто удаляем его
                        walkersToDelete.Add(walker);

                        // если его текущая позиция - терминальный узел, то 
                        // вставляем его в список уникальных терминальных узлов, если он в этом списке ещё отсутствует
                        if (PositionIsTerminal)
                        {
                            int ind = result.TerminalVertices.BinarySearch(walker.CurrentPosition);
                            if (ind < 0)
                            {
                                result.TerminalVertices.Insert(-ind - 1, walker.CurrentPosition);
                            }
                        }
                    }



                }

                // удаляем путешественников, которым некуда идти
                foreach (TwoDirectionWalker w in walkersToDelete)
                {
                    walkers.Remove(w);
                }

                // добавляем новых путешественников:
                foreach (TwoDirectionWalker w in walkersToAdd)
                {
                    walkers.Add(w);
                }

                walkersToAdd.Clear();
                walkersToDelete.Clear();

            }

            return result;
        }

        /// <summary>
        /// Список терминальных узлов. Используется при планаризации графа
        /// </summary> 
        protected List<Physical2DGraphVertex> TerminalVertices = new List<Physical2DGraphVertex>();


        protected void ClearPassedMarks()
        {
            foreach (Physical2DGraphEdge<Physical2DGraphVertex> edge in this.Edges)
            {
                edge.IsPassedFromSourceToTarget = false;
                edge.IsPassedFromTargetToSource = false;
            }
        }

        protected void ClearEmededAndPassedMarks()
        {
            foreach (Physical2DGraphEdge<Physical2DGraphVertex> edge in this.Edges)
            {
                edge.IsEmbededFromSourceToTarget = false;
                edge.IsEmbededFromTargetToSource = false;
                edge.IsPassedFromSourceToTarget = false;
                edge.IsPassedFromTargetToSource = false;
            }
        }


        protected List<Cycle<Physical2DGraphVertex>> fasets;

        /// <summary>
        /// Грани графа (заполнены после проверки на планарность)
        /// </summary>
        public List<Cycle<Physical2DGraphVertex>> Fasets { get { return fasets; } }




        /// <summary>
        /// Параметры случайного расположения новых узлов на 2D плоскости
        /// </summary>
        private NewVertexSeedModel newVertexSeedParametres;
        public NewVertexSeedModel NewVertexSeedParametres { get { return newVertexSeedParametres; } }

        private PhysicalModelParametres physicalModelParametres;
        public PhysicalModelParametres PhysicalModelParametres { get { return physicalModelParametres; } }

        // заранее расчитанная обратная величина инерциальной массы, пригодится для всех итераций  - для скорости.
        private double nodeInertialMassRation;

        // Среднее рассхождение между итерациями
        public double difference;
        public double difference_priorStep;
        public double difference_priorPriorStep;
        public Point massCenter;        // центр масс узла - для компенсации смещения из-за неустойчивости алгоритмка развёртывания
        public double avgAngleVelocity; // для компенсации вращения из-за неустойчивости алгоритма развёртывания

        // кластеры, для расчёта гравитационных сил
        protected internal List<Physical2DGraphVertexCluster> clusters = new List<Physical2DGraphVertexCluster>();


        /// <summary>
        /// Режим автоматическоро разрезания движущихся рёбер, проходящих в процессе движения через точку разрезания
        /// </summary>
        public Boolean isCuttingUpMode = false;
        /// <summary>
        /// Точка разрезания
        /// </summary>
        public Point cuttingUpPoint = new Point();





        public Physical2DGraph()
        {
            newVertexSeedParametres = new NewVertexSeedModel();
            physicalModelParametres = new PhysicalModelParametres();
        }

        public void Reset()
        {
            difference = 0.0D;
            difference_priorStep = 0.0D;
            difference_priorPriorStep = 0.0D;
            massCenter = new Point(0, 0);
        }


        /// <summary>
        /// Развёртывание графа на плоскости, для отображения. Используется "физическая модель".
        /// Перед развёртыванием начальное положение узлов устанавливается в окрестности точки centerPosition
        /// </summary>
        /// <param name="iterationCount"></param>
        public void ProcessUnfoldingModel(NumericalMethodParametres parametres, Point centerPosition)
        {
            Random rnd = new Random((Int32)DateTime.Now.Ticks);
            foreach (Physical2DGraphVertex v in this.Vertices)
            {
                v.Position = new Point(centerPosition.X * 0.5 + this.PhysicalModelParametres.FreeConnectionLength * rnd.NextDouble(),
                                       centerPosition.Y * 0.5 + this.PhysicalModelParametres.FreeConnectionLength * rnd.NextDouble()
                                        );
            };



            this.ProcessUnfoldingModel(parametres);
        }

        public void ProcessUnfoldingModel(NumericalMethodParametres parametres)
        {

            bool dontCalculateRemoteNodesGravitationSavedValue = parametres.DontCalculateRemoteNodesGravitation;

            //заранее расчитываем обратную величину инерциальной массы  - для скорости.
            nodeInertialMassRation = 1 / physicalModelParametres.NodeInertialMass;

            massCenter = CalcMassCenter();

            if (parametres.DontCalculateRemoteNodesGravitation)
            {
                // Разбиваем множество узлов на кластеры
                // Т.е. разбиение на кластеры  топологическое и состав кластеров не меняется от положения узлов (меняются лишь их центры масс) 
                // то делаем один раз для всех внешних итераций - ра
                this.DivideByCluster(parametres.NearestNodesCountThreshold);
            }


            // Если количество внешних итераций > 1 то их считаем с оптимизацией расчёта гравитационного взаимодействия, кроме последней 
            // TODO^
            if (parametres.OuterIterationCount > 1)
            {
                parametres.DontCalculateRemoteNodesGravitation = true;
            }

            for (int i = 0; i < parametres.OuterIterationCount; i++)
            {
                if ((i > 0) && (i == parametres.OuterIterationCount - 1))
                {
                    parametres.DontCalculateRemoteNodesGravitation = dontCalculateRemoteNodesGravitationSavedValue;
                }

                #region Расчёт гравитационной силы далёких узлов
                if (parametres.DontCalculateRemoteNodesGravitation)
                {

                    // Для каждого кластера рассчитываем центр масс;
                    foreach (var cluster in clusters)
                    {
                        cluster.SumGravitationMass = cluster.Vertices.Count;
                        cluster.Center.X = 0;
                        cluster.Center.Y = 0;

                        foreach (var v in cluster.Vertices)
                        {
                            cluster.Center.X = cluster.Center.X + v.Position.X;
                            cluster.Center.Y = cluster.Center.Y + v.Position.Y;
                        }
                        cluster.Center.X = cluster.Center.X / cluster.Vertices.Count;
                        cluster.Center.Y = cluster.Center.Y / cluster.Vertices.Count;

                    }

                    // теперь определяем список ближайших узлов и гравитационную силу остальных узлов (приближённая оценка)
                    foreach (var vertex in Vertices)
                    {
                        // 10 ближайших узлов и фиксируем их список ближайших узлов
                        // 
                        vertex.FillNearesNodes(parametres.NearestNodesCountThreshold);
                    }







                }


                #endregion

                

                for (int iterationIndex = 0; iterationIndex < parametres.OneProcessIterationCount; iterationIndex++)
                {
                    //ProcessOneStep_Euler(parametres);
                    switch (parametres.Method)
                    {
                        case NumericalMethod.Euler:
                            ProcessOneStep_Euler(parametres);
                            break;
                        case NumericalMethod.NewtonRelax:
                            //ProcessOneStep_Potencial_LocalNewton(parametres as NewtonNumericalMethodParametres);
                            break;
                        case NumericalMethod.RungeKutta4:
                            //ProcessOneStep_RungeKutta4(parametres);
                            break;
                        default:
                            break;
                    }
                }

            }

                //parametres.DontCalculateRemoteNodesGravitation = false;
                //this.ProcessUnfoldingModel(parametres);







            // восстанавливаем исходные параметры
            parametres.DontCalculateRemoteNodesGravitation = dontCalculateRemoteNodesGravitationSavedValue;
            
            // центрируем и выравниваем граф, который сместился и "раскрутился" в результате погрешностей неустойчивого метода
            MoveMassCenterTo(massCenter);
            SetAvgRotationVelocityToZero(massCenter); // компенсируем вращательное движение из-за погрешности оптимизированного алгоритма
            avgAngleVelocity = CalcAvgRotationVelocity(massCenter); // скорость после компенсации

        }


        /// <summary>
        /// Метод гравитационного упругого моделирования
        /// с вычислением по методу Эйлера
        /// </summary>
        /// <param name="parametres"></param>
        protected void ProcessOneStep_Euler(NumericalMethodParametres parametres)
        {
            //    1. сначала для каждого узла рассчитываем суммы действующих на него сил
            //    2. Рассчитываем ускорение узла (прирост скорости)
            //    3. Рассчитываем новое значение скорости
            //    4. Рассчитываем новое положение узла}

            double timeInterval = parametres.OneIterationTimeStep;


            if (parametres.AutoDecreaseTimeStep)
            {
                if (difference > 0)
                {
                    timeInterval = Math.Min(timeInterval, 0.2D / (Math.Max(Math.Max(difference, difference_priorStep), difference_priorPriorStep)));
                }
            }
            difference_priorPriorStep = difference_priorStep;
            difference_priorStep = difference;


            foreach (var vertex in Vertices)
            {
                //  --- 1. рассчитываем действующие на узел силы 
                vertex.force = new Vector(0, 0);

                // -- 1.2. сила гравитации
                vertex.force = Vector.Add(vertex.force, CalcGravitationForce(vertex, parametres.DontCalculateRemoteNodesGravitation));

                // -- 1.2. сила упругости стержней  
                vertex.force = Vector.Add(vertex.force, CalcStiffnessForce(vertex));

                //  -- 1.3. сила сопротивления движению  }
                vertex.force = Vector.Add(vertex.force, CalcDissipativeForce(vertex));

                //  --- 2. Рассчитываем ускорение узла (прирост скорости) }                
                /// ----- old
                Vector vAcseleration = Vector.Multiply(vertex.force, PhysicalModelParametres.NodeInertialMass);
                vertex._newVelocity = Vector.Add(vertex.velocity, Vector.Multiply(vAcseleration, timeInterval));
                vertex._newPosition = (Point)Vector.Add((Vector)vertex.position, Vector.Multiply(vertex._newVelocity, timeInterval));
                /// -----


                //Vector vVelocityHalf = Vector.Add(vertex._velocity, Vector.Multiply(vertex._AcselerationPrior, 0.5D * timeInterval));
                //vertex._newPosition = (Point)Vector.Add((Vector)vertex._position, Vector.Multiply(vVelocityHalf, timeInterval));

                //Vector vAcseleration = Vector.Multiply(vertex._force, PhysicalModelParametres.NodeInertialMass);
                //vertex._newVelocity = Vector.Add(vVelocityHalf, Vector.Multiply(vAcseleration, 0.5D * timeInterval));                

                //vertex._AcselerationPrior = vAcseleration;

            }

            CalcDifference();
            UpdateNewProperties();



            // глючит: много лишних вертикальных удалений
            //if (IsCuttingUpMode)
            //{
            //    List<Physical2DGraphEdge<Physical2DGraphVertex>> edgesToDelete = new List<Physical2DGraphEdge<Physical2DGraphVertex>>();

            //    // Удаляем рёбра, если в четырёх угольник, образованный старым и новым положением ребра, попадает точка разреза
            //    Point[] points = new Point[4];

            //    foreach (var edge in this.Edges)
            //    {
            //        points[0] = edge.Source._oldPosition;
            //        points[1] = edge.Target._oldPosition;
            //        points[2] = edge.Target._position;
            //        points[3] = edge.Source._position; 

            //        if (PointInPolygon(CuttingUpPoint, points))
            //        {
            //            edgesToDelete.Add(edge);
            //        }
            //    }


            //    //this.RemoveEdges(edgesToDelete);
            //}
        }

        /// <summary>
        /// Разрезание графа. Метод может вызываться в ответ на действия пользователя (курсором мыши)
        /// </summary>
        /// <param name="edge"></param>
        public virtual void CutUpEdge(Physical2DGraphEdge<Physical2DGraphVertex> edge)
        {
            try
            {
                this.RemoveEdge(edge);
            }
            catch
            {
            }
        }

        static bool PointInPolygon(Point p, Point[] poly)
        {
            Point p1, p2;

            bool inside = false;

            if (poly.Length < 3)
            {
                return inside;
            }

            Point oldPoint = new Point(poly[poly.Length - 1].X, poly[poly.Length - 1].Y);

            for (int i = 0; i < poly.Length; i++)
            {
                Point newPoint = new Point(poly[i].X, poly[i].Y);
                if (newPoint.X > oldPoint.X)
                {
                    p1 = oldPoint;
                    p2 = newPoint;
                }
                else
                {
                    p1 = newPoint;
                    p2 = oldPoint;
                }

                if ((newPoint.X < p.X) == (p.X <= oldPoint.X)
                    && ((long)p.Y - (long)p1.Y) * (long)(p2.X - p1.X)
                     < ((long)p2.Y - (long)p1.Y) * (long)(p.X - p1.X))
                {
                    inside = !inside;
                }

                oldPoint = newPoint;
            }

            return inside;
        }


        /// <summary>
        /// Расчет итерации методом поиска минимума потенциала (сил), локально, методом Ньютона.
        /// 
        /// Для всей системы - существует функция потенциальной энергии в зависиомсти от положений всех вершин.
        /// (Не будем решать задачу от всех переменных (положений  всех точек = 2N)
        /// а будем на каждой итерации решать локально, для каждой точки.
        /// Если заморозить все точки кроме одной, то существует функция потенциала
        /// зависещей от положения этой точки.
        /// Математически задача сводится к нахождению миниму этого потенциала,
        /// или что тоже самое нулю всех действующих сил.
        /// (В отличии от метода Эйлера, и физической модели - мы не будем опираться на время,
        /// а используя производную не только потенциала (его градиент и есть сила),
        /// а еще и как эта сила изменяется в пространстве (производная поля силы) - сразу прикинем через какое время она станет равной нулю
        ///  - то есть методом Ньютона решения нелинейных уравнений (в данном случае от двух переменных и векторного значения - системы)
        ///  http://ru.wikipedia.org/wiki/Метод_Ньютона
        /// 
        /// U(x,y) - локальный потенциал потенциальных сил (скаляр)
        /// U(x,y) = U_гравитации + U_упругости
        /// F(x, y) = grad (U) - действующая сила (вектор), записываем формулу.
        /// dF(x, y) = [[d Fx/dx, d Fx/dy ], [d Fy/dx, d Fy/dy ] - матрица из столбцов производных по направлению.
        /// 
        /// а) Тут можно применить Метод Ньютона для каждой составляющей силы (чтоб она была равна 0)
        /// 
        /// x_нов = x_стар - Fx/(dFx/dx)   ; решение Fx(x, y)=0, где x переменная, y - параметр
        /// y_нов = y_стар - Fy/(dFy/dy)   ; наоброт
        /// 
        /// б) Решать сразу систему двух уравнений, относительно (x,y)
        /// http://ru.wikipedia.org/wiki/Метод_Ньютона - многомерный случай
        /// 
        /// Тут придется вычислять детерминант матрицы производных по направлению,
        ///  зато на каждой итерации окажемся ближе к верному решению (для данной точки).
        ///  
        /// в) Решать систему сразу для большого числа точек в окрестности (конгломерата) - это на будущее.
        /// 
        /// </summary>
        /// <param name="isSimple">True - более простым методом (вариант а), False - с помощью системы</param>
        ///     
        protected void ProcessOneStep_Potencial_LocalNewton(NewtonNumericalMethodParametres parametres)
        {
            //    1. сначала для каждого узла рассчитываем суммы действующих на него сил
            //    2. Рассчитываем частные производные
            //    3. Рассчитываем новое положение узла}
            //       а) откладываем примененние изменения (хуже)
            //       б) сразу двигаем точку (лучше)

            // диагностика погрешности , для варианта б)
            if (parametres.IsMoveNow)
            {
                difference = 0;
            }


            foreach (var vertex in Vertices)
            {
                //  --- 1. рассчитываем действующие на узел силы 
                vertex.force = CalcForce(vertex, parametres.DontCalculateRemoteNodesGravitation);

                double dx = 0;
                double dy = 0;
                double d = 0;
                if (parametres.isTwoCoords)
                {
                    // 2. Рассчитываем частные производные
                    Vector Fdx = CalcForceDx(vertex, parametres, vertex.force);
                    Vector Fdy = CalcForceDy(vertex, parametres, vertex.force);
                    // 3. Рассчитываем новое положение узла. Решаем  
                    //
                    //    F+A(dP)=0,
                    //
                    //    dp = -F*A^(-1);
                    // где dP = [dx, dy]^T; (столбец)
                    //     A = [[Fdx.X, Fdy.X], [Fdx.Y, Fdy.Y]]
                    // и используем формулу для обратной матрицы
                    // [a  b]-1     1    [ d  -b]
                    //          = -----  
                    // [c  d]     ad-bc  [-c   a]
                    double detA = Fdx.X * Fdy.Y - Fdy.X * Fdx.Y;
                    if (detA != 0)
                    {
                        dx = -(vertex.force.X * Fdy.Y - vertex.force.Y * Fdy.X) / detA;
                        dy = -(-vertex.force.X * Fdx.Y + vertex.force.Y * Fdx.X) / detA;
                    }
                }
                else
                {
                    // Обычный метод ньютона - отдельно по каждой кординате.
                    // 2. Рассчитываем частные производные
                    Vector Fdx = CalcForceDx(vertex, parametres, vertex.force);
                    // 3. Рассчитываем новое положение узла.
                    dx = -vertex.force.X / Fdx.X;
                    dx = dx * parametres.BrakeFactor;


                    Vector Fdy = CalcForceDy(vertex, parametres, vertex.force);
                    dy = -vertex.force.Y / Fdy.Y;
                    dy = dy * parametres.BrakeFactor;
                }

                // обрабатываем максимально перемещение узла
                double s2 = dx * dx + dy * dy;
                d = Math.Sqrt(s2);

                if (parametres.MaxShift < d)
                {
                    dx = dx * parametres.MaxShift / d;
                    dy = dy * parametres.MaxShift / d;
                    d = parametres.MaxShift; // обновлляем, т.к. пригодится протом для погроешности
                }

                vertex._newPosition.X = vertex.position.X + dx;
                vertex._newPosition.Y = vertex.position.Y + dy;

                if (parametres.IsMoveNow)
                {
                    // б) сразу двигаем точку (лучше)
                    // не забываем вычислить погрешность.
                    difference += Math.Sqrt(d);

                    vertex.position = vertex._newPosition;
                }
                // а) откладываем примененние изменения (хуже)
            }

            if (parametres.IsMoveNow)
            {
                // только для варианта б)
                difference = difference / Vertices.Count();
            }
            else
            {
                // только для варианта a)
                CalcDifference();
                UpdateNewProperties();
            }
        }

        /// <summary>
        /// Методом Рунге Кутта 4-го порядка
        /// 
        /// Приведем задачу к системе ОДУ первого порядка
        /// Для этого раздели уравнения движение (второго порядка) на два уравнения первого порядка
        /// путем введения скорости, по каждым координатам, по каждым вершинам
        /// 
        /// x' = Vx
        /// Vx' = Fx(...)
        /// y' = Vy
        /// Vy' = Fy(...)
        /// ...
        /// (для всех вершин)
        /// 
        /// (' - производные по времени,
        /// индексы x, y - соответсвующие координатам компоненты )
        /// Получилися система ОДУ 1-го порядка из 4N уравнений
        /// 
        /// Далее применяем метод РК
        /// http://ru.wikipedia.org/wiki/Метод_Рунге_—_Кутта
        /// к этой системе
        /// http://alglib.sources.ru/diffequations/rungekuttasys.php
        /// 
        /// </summary>
        /// <param name="parametres"></param>
        protected void ProcessOneStep_RungeKutta4(NumericalMethodParametres parametres)
        {
            double dt = parametres.OneIterationTimeStep;
            // первый этап
            foreach (var vertex in Vertices)
            {
                // запоминаем начальные значения
                vertex._oldPosition = vertex.position;
                vertex._oldVelocity = vertex.velocity;
                // вычисление силы 
                Vector F = CalcForce(vertex, parametres.DontCalculateRemoteNodesGravitation);
                F = Vector.Add(F, CalcDissipativeForce(vertex));

                // вычисления коэффициентов первого этапа
                vertex._xk1 = vertex.velocity.X * dt;
                vertex._xm1 = F.X * dt;

                vertex._yk1 = vertex.velocity.Y * dt;
                vertex._ym1 = F.Y * dt;
            }

            // подготавливаем аргументы для второго этапа
            foreach (var vertex in Vertices)
            {
                vertex.position.X = vertex._oldPosition.X + vertex._xk1 / 2;
                vertex.velocity.X = vertex._oldVelocity.X + vertex._xm1 / 2;

                vertex.position.Y = vertex._oldPosition.Y + vertex._yk1 / 2;
                vertex.velocity.Y = vertex._oldVelocity.Y + vertex._ym1 / 2;
            }

            // второй этап
            foreach (var vertex in Vertices)
            {
                Vector F = CalcForce(vertex, parametres.DontCalculateRemoteNodesGravitation);
                F = Vector.Add(F, CalcDissipativeForce(vertex));
                // вычисления коэффициентов этапа
                vertex._xk2 = vertex.velocity.X * dt;
                vertex._xm2 = F.X * dt;

                vertex._yk2 = vertex.velocity.Y * dt;
                vertex._ym2 = F.Y * dt;
            }

            // подготавливаем аргументы для третьего этапа
            foreach (var vertex in Vertices)
            {
                vertex.position.X = vertex._oldPosition.X + vertex._xk2 / 2;
                vertex.velocity.X = vertex._oldVelocity.X + vertex._xm2 / 2;

                vertex.position.Y = vertex._oldPosition.Y + vertex._yk2 / 2;
                vertex.velocity.Y = vertex._oldVelocity.Y + vertex._ym2 / 2;
            }

            // третий этап
            foreach (var vertex in Vertices)
            {
                Vector F = CalcForce(vertex, parametres.DontCalculateRemoteNodesGravitation);
                F = Vector.Add(F, CalcDissipativeForce(vertex));

                // вычисления коэффициентов этапа
                vertex._xk3 = vertex.velocity.X * dt;
                vertex._xm3 = F.X * dt;

                vertex._yk3 = vertex.velocity.Y * dt;
                vertex._ym3 = F.Y * dt;
            }

            // подготавливаем аргументы для четвертого этапа
            foreach (var vertex in Vertices)
            {
                vertex.position.X = vertex._oldPosition.X + vertex._xk3;
                vertex.velocity.X = vertex._oldVelocity.X + vertex._xm3;

                vertex.position.Y = vertex._oldPosition.Y + vertex._yk3;
                vertex.velocity.Y = vertex._oldVelocity.Y + vertex._ym3;
            }

            // четвертый этап
            foreach (var vertex in Vertices)
            {
                Vector F = CalcForce(vertex, parametres.DontCalculateRemoteNodesGravitation);
                F = Vector.Add(F, CalcDissipativeForce(vertex));
                // вычисления коэффициентов этапа
                vertex._xk4 = vertex.velocity.X * dt;
                vertex._xm4 = F.X * dt;

                vertex._yk4 = vertex.velocity.Y * dt;
                vertex._ym4 = F.Y * dt;
            }

            // окончательный результат + возвращаем старые значения (для последующей оценки погрешности)
            // ...
            // можно сразу засовывать в значения _position, а старые записываем как _newPosition
            // так как функции CalcDifference все равно (симемтрична относительно них)
            // (но не применять UpdateNewProperties)
            foreach (var vertex in Vertices)
            {
                vertex._newPosition.X = vertex._oldPosition.X + (vertex._xk1 + 2 * vertex._xk2 + 2 * vertex._xk3 + vertex._xk4) / 6;
                vertex._newPosition.Y = vertex._oldPosition.Y + (vertex._yk1 + 2 * vertex._yk2 + 2 * vertex._yk3 + vertex._yk4) / 6;
                vertex._newVelocity.X = vertex._oldVelocity.X + (vertex._xm1 + 2 * vertex._xm2 + 2 * vertex._xm3 + vertex._xm4) / 6;
                vertex._newVelocity.Y = vertex._oldVelocity.Y + (vertex._ym1 + 2 * vertex._ym2 + 2 * vertex._ym3 + vertex._ym4) / 6;
                vertex.position = vertex._oldPosition;
                vertex.velocity = vertex._oldVelocity;
            }

            CalcDifference();
            UpdateNewProperties();

        }

        #region Вспомогательные вычисления модели (Силы, Гравитация, Упругость, производные)
        /// <summary>
        /// Вычесление всех сил (гравитационной и упругости стержней)
        /// </summary>
        /// <param name="vertex"></param>
        /// <returns></returns>
        private Vector CalcForce(Physical2DGraphVertex vertex, bool dontCalculateRemoteNodesGravitation)
        {
            Vector force = new Vector(0, 0);
            force = Vector.Add(force, CalcGravitationForce(vertex, dontCalculateRemoteNodesGravitation));
            force = Vector.Add(force, CalcStiffnessForce(vertex));
            return force;
        }


        /// <summary>
        /// Расчет силы гравитации, действующей на данную вершину
        /// </summary>
        /// <param name="vertex"></param>
        /// <returns></returns>
        private Vector CalcGravitationForce(Physical2DGraphVertex vertex, bool dontCalculateRemoteNodesGravitation)
        {
            Vector gravitationForce = new Vector(0, 0);
            //return gravitationForce;

            //var dfs = new UndirectedDepthFirstSearchAlgorithm<Physical2DGraphVertex,Edge<Physical2DGraphVertex>>(this);

            //dfs.MaxDepth = 2;
            //dfs.Compute(vertex);
            //dfs.VisitedGraph.VertexCount



            if (physicalModelParametres.IsWithConnectedNodesGravitationInteractionOnly)
            {
                #region рассчитываем силу гравитации только для топологически близких узлов

                List<Physical2DGraphVertex> verticesLevel1 = new List<Physical2DGraphVertex>();

                // находим соседей первого уровня:
                foreach (Edge<Physical2DGraphVertex> ed in this.AdjacentEdges(vertex))
                {
                    // определили соседей первого уровня
                    Physical2DGraphVertex vertexTo = ed.GetOtherVertex<Physical2DGraphVertex, Edge<Physical2DGraphVertex>>(vertex);

                    // узел может быть vertexTo связан с vertex и двумя рёбрами, и поэтому уже находится в списке verticesLevel1
                    if (verticesLevel1.IndexOf(vertexTo) < 0)
                    {
                        verticesLevel1.Add(vertexTo);
                    }

                };

                // определяем соседей перового и второго уровня:
                // добавляем только не существующих
                List<Physical2DGraphVertex> verticesLevel2 = new List<Physical2DGraphVertex>();
                verticesLevel2.AddRange(verticesLevel1);
                foreach (Physical2DGraphVertex v1 in verticesLevel1)
                {
                    foreach (Edge<Physical2DGraphVertex> ed in this.AdjacentEdges(v1))
                    {
                        // определили соседей первого уровня
                        Physical2DGraphVertex vertexTo = ed.GetOtherVertex<Physical2DGraphVertex, Edge<Physical2DGraphVertex>>(vertex);
                        if (verticesLevel2.IndexOf(vertexTo) < 0)
                        {
                            verticesLevel2.Add(vertexTo);
                        }
                    }
                }
                // verticesLevel2 - содержит соседей первого и второго уровня
                foreach (Physical2DGraphVertex vertexTo in verticesLevel2)
                {
                    if (vertexTo != vertex)
                    {

                        double distance = VectorMath_Distance(vertex.Position, vertexTo.Position);
                        Vector direction = Vector.Subtract((Vector)vertex.Position, (Vector)vertexTo.Position);
                        double GFToSingleNode_Value = PhysicalModelParametres.NodeGravitationCoefficient / (distance * distance);
                        Vector vGFToSingleNode = Vector.Multiply(direction, GFToSingleNode_Value);
                        gravitationForce = Vector.Add(gravitationForce, vGFToSingleNode);
                    }
                }

                #endregion
            }
            else
            {
                if (dontCalculateRemoteNodesGravitation)
                {
                    #region рассчитываем силу гравитации для ближайших узлов и прибавляем её к заранее рассчитанному влиянию далёких

                    gravitationForce.X = vertex._remoteNodesGravitationForce.X;
                    gravitationForce.Y = vertex._remoteNodesGravitationForce.Y;

                    foreach (Physical2DGraphVertex vertexTo in vertex.NearestNodes)
                    {
                        double distance = VectorMath_Distance(vertex.Position, vertexTo.Position);
                        Vector direction = Vector.Subtract((Vector)vertex.Position, (Vector)vertexTo.Position);
                        double GFToSingleNode_Value = PhysicalModelParametres.NodeGravitationCoefficient / (distance * distance);
                        Vector vGFToSingleNode = Vector.Multiply(direction, GFToSingleNode_Value);
                        gravitationForce = Vector.Add(gravitationForce, vGFToSingleNode);
                    }
                    #endregion
                }
                else
                {
                    #region рассчитываем силу гравитации для всех узлов

                    foreach (Physical2DGraphVertex vertexTo in this.Vertices)
                    {
                        if (vertexTo != vertex)
                        {

                            double distance = VectorMath_Distance(vertex.Position, vertexTo.Position);
                            Vector direction = Vector.Subtract((Vector)vertex.Position, (Vector)vertexTo.Position);
                            double GFToSingleNode_Value = PhysicalModelParametres.NodeGravitationCoefficient / (distance * distance);
                            Vector vGFToSingleNode = Vector.Multiply(direction, GFToSingleNode_Value);
                            gravitationForce = Vector.Add(gravitationForce, vGFToSingleNode);
                        }
                    }

                    #endregion
                }


            }
            return gravitationForce;
        }

        /// <summary>
        /// Расчёт силы упругости стержней действующих на выделенную вершину
        /// </summary>
        /// <param name="vertex"></param>
        /// <returns></returns>
        private Vector CalcStiffnessForce(Physical2DGraphVertex vertex)
        {
            Vector vStiffnessForce = new Vector(0, 0);
            foreach (Edge<Physical2DGraphVertex> edge in this.AdjacentEdges(vertex))
            {
                Physical2DGraphVertex nearNode = edge.GetOtherVertex(vertex);
                if (nearNode != vertex)
                {
                    double distance = this.VectorMath_Distance(vertex.Position, nearNode.Position);
                    Vector direction = Vector.Subtract((Vector)vertex.Position, (Vector)nearNode.Position);
                    double SFToSingleNode_Value = PhysicalModelParametres.ConectionGripFactor * (PhysicalModelParametres.FreeConnectionLength - distance);
                    Vector vSFToSingleNode = Vector.Multiply(direction, SFToSingleNode_Value);
                    vStiffnessForce = Vector.Add(vStiffnessForce, vSFToSingleNode);
                }
            }
            return vStiffnessForce;
        }
        /// <summary>
        /// расчет силы сопротивления движению
        /// </summary>
        /// <param name="vertex"></param>
        /// <returns></returns>
        private Vector CalcDissipativeForce(Physical2DGraphVertex vertex)
        {
            //      vVelocity_Value := R(vNode.fVelocity.x, vNode.fVelocity.y); // модуль скорости
            double velocityValue = Math.Sqrt(vertex.velocity.X * vertex.velocity.X + vertex.velocity.Y * vertex.velocity.Y);
            double dissipativeForce_Value = PhysicalModelParametres.ResistingForceFactor_0 + velocityValue * PhysicalModelParametres.ResistingForceFactor_1;
            Vector dissipativeForce = Vector.Multiply(vertex.velocity, -dissipativeForce_Value);
            return dissipativeForce;
        }


        /// <summary> Вычисление производной силы (численный метод) по направлению X;
        /// </summary>
        /// <param name="vertex"></param>
        /// <returns></returns>
        private Vector CalcForceDx(Physical2DGraphVertex vertex, NewtonNumericalMethodParametres p)
        {
            Vector f0 = CalcForce(vertex, p.DontCalculateRemoteNodesGravitation); // сила в начальеной точке (возможно уже вычисленная(
            Point p0 = vertex.position;  // запоминаем начальное положение 
            vertex.position.X += p.Delta; // смещаем положение на малую величину в опр. напр.
            Vector f = CalcForce(vertex, p.DontCalculateRemoteNodesGravitation);  // вычисляем изменившееся значение силы

            vertex.position = p0; // восстанавливаем значение

            // вычисляем произвордную
            Vector result = Vector.Subtract(f, f0);
            result = Vector.Divide(result, p.Delta);
            return result;
        }

        /// <summary> Вычисление производной силы (численный метод) по направлению X;
        /// </summary>
        /// <param name="vertex"></param>
        /// <returns></returns>
        private Vector CalcForceDx(Physical2DGraphVertex vertex, NewtonNumericalMethodParametres p, Vector f0)
        {
            //Vector f0 = CalcForce(vertex); // сила в начальеной точке (возможно уже вычисленная(
            Point p0 = vertex.position;  // запоминаем начальное положение 
            vertex.position.X += p.Delta; // смещаем положение на малую величину в опр. напр.
            Vector f = CalcForce(vertex, p.DontCalculateRemoteNodesGravitation);  // вычисляем изменившееся значение силы

            vertex.position = p0; // восстанавливаем значение

            // вычисляем производную
            Vector result = Vector.Subtract(f, f0);
            result = Vector.Divide(result, p.Delta);
            return result;
        }

        /// <summary> Вычисление производной силы (численный метод) по направлению Y;
        /// </summary>
        /// <param name="vertex"></param>
        /// <returns></returns>
        private Vector CalcForceDy(Physical2DGraphVertex vertex, NewtonNumericalMethodParametres p, Vector f0)
        {
            //Vector f0 = CalcForce(vertex); // сила в начальеной точке (возможно уже вычисленная)
            Point p0 = vertex.position;  // запоминаем начальное положение 
            vertex.position.Y += p.Delta; // смещаем положение на малую величину в опр. напр.
            Vector f = CalcForce(vertex, p.DontCalculateRemoteNodesGravitation);  // вычисляем изменившееся значение силы

            vertex.position = p0; // восстанавливаем значение

            // вычисляем произвордную
            Vector result = Vector.Subtract(f, f0);
            result = Vector.Divide(result, p.Delta);
            return result;
        }

        /// <summary>Вычисление производной силы (численный метод) по направлению (заданому);
        /// </summary>
        /// <param name="vertex"></param>
        /// <returns></returns>
        private Vector CalcForceDy(Physical2DGraphVertex vertex, Vector direction, NewtonNumericalMethodParametres p, Vector f0)
        {
            //Vector f0 = CalcForce(vertex); // сила в начальеной точке (возможно уже вычисленная)
            Point p0 = vertex.position;  // запоминаем начальное положение 
            vertex.position.X += p.Delta * direction.X; // смещаем положение на малую величину в опр. напр.
            vertex.position.Y += p.Delta * direction.Y; // 
            Vector f = CalcForce(vertex, p.DontCalculateRemoteNodesGravitation);  // вычисляем изменившееся значение силы
            vertex.position = p0; // восстанавливаем значение
            // вычисляем произвордную
            Vector result = Vector.Subtract(f, f0);
            result = Vector.Divide(result, p.Delta);
            return result;

        }

        /// <summary> Вычисление производной силы (численный метод) по направлению Y;
        /// </summary>
        /// <param name="vertex"></param>
        /// <returns></returns>
        private Vector CalcForceDy(Physical2DGraphVertex vertex, NewtonNumericalMethodParametres p)
        {
            Vector f0 = CalcForce(vertex, p.DontCalculateRemoteNodesGravitation); // сила в начальеной точке (возможно уже вычисленная)
            Point p0 = vertex.position;  // запоминаем начальное положение 
            vertex.position.Y += p.Delta; // смещаем положение на малую величину в опр. напр.
            Vector f = CalcForce(vertex, p.DontCalculateRemoteNodesGravitation);  // вычисляем изменившееся значение силы

            vertex.position = p0; // восстанавливаем значение

            // вычисляем произвордную
            Vector result = Vector.Subtract(f, f0);
            result = Vector.Divide(result, p.Delta);
            return result;
        }

        /// <summary>Вычисление производной силы (численный метод) по направлению (заданому);
        /// </summary>
        /// <param name="vertex"></param>
        /// <returns></returns>
        private Vector CalcForceDy(Physical2DGraphVertex vertex, Vector direction, NewtonNumericalMethodParametres p)
        {
            Vector f0 = CalcForce(vertex, p.DontCalculateRemoteNodesGravitation); // сила в начальеной точке (возможно уже вычисленная)
            Point p0 = vertex.position;  // запоминаем начальное положение 
            vertex.position.X += p.Delta * direction.X; // смещаем положение на малую величину в опр. напр.
            vertex.position.Y += p.Delta * direction.Y; // 
            Vector f = CalcForce(vertex, p.DontCalculateRemoteNodesGravitation);  // вычисляем изменившееся значение силы
            vertex.position = p0; // восстанавливаем значение
            // вычисляем произвордную
            Vector result = Vector.Subtract(f, f0);
            result = Vector.Divide(result, p.Delta);
            return result;

        }


        #endregion

        #region Вспомогательный процедуры для итерации

        // 
        /// <summary>
        /// расчет длины между точками (использовать или VectorMath_Distance или Length или это, что быстрее)
        /// </summary>
        /// <param name="p1">первая точка</param>
        /// <param name="p2">вторая</param>
        /// <returns>Расстояние между ними</returns>
        private double getPointsDistance(Point p1, Point p2)
        {
            double dx = p1.X - p2.X;
            double dy = p1.Y - p2.Y;
            double s2 = dx * dx + dy * dy;
            return Math.Sqrt(s2);
        }

        /// <summary>
        /// обновление новых координат и скоростей 
        /// </summary>
        private void UpdateNewProperties()
        {
            foreach (var vertex in Vertices)
            {
                ///обновляем скорость только в том случае, если она не превышает параметр
                if (vertex._newVelocity.Length < PhysicalModelParametres.MaxVelocity)
                {
                    vertex.velocity = vertex._newVelocity;
                }
                vertex._oldPosition = vertex.position;
                vertex.position = vertex._newPosition;
            }
        }

        /// <summary>
        ///  оценка изменения (разницы) смещения (запускать ДО обновления координат) 
        ///  Нужно запустить не в каждой итерации, а только на предпоследней в серии.
        ///  или запускать по требованию (для этого в классе вершина,
        ///  сохранять старые значения положений даже после обновления)
        /// </summary>
        private void CalcDifference()
        {
            difference = 0;
            // суммируем длины каждого смещения, и делим на числе вершин
            foreach (var vertex in Vertices)
            {
                difference += getPointsDistance(vertex.position, vertex._newPosition);
            }
            difference = difference / Vertices.Count();
        }

        /// <summary>
        /// Вычисление центра масс
        /// </summary>
        private Point CalcMassCenter()
        {
            Point p = new Point(0, 0);
            double number = Vertices.Count();
            foreach (var vertex in Vertices)
            {
                p.X += vertex.position.X / number;
                p.Y += vertex.position.Y / number;
            }
            return p;
        }

        /// <summary>
        /// Вычисление угла вращения 
        /// </summary>
        private double CalcAvgRotationVelocity(Point centerPoint)
        {
            double sum = 0;

            foreach (var vertex in Vertices)
            {
                Vector vectorCP = (Vector)centerPoint - (Vector)vertex.position;
                double R = vectorCP.Length;
                double gamma = Vector.AngleBetween(vectorCP, vertex.velocity);

                if (R != 0) sum += Math.Sin(gamma * Math.PI / 180) * vertex.velocity.Length / R;

            }
            return sum / Vertices.Count();
        }

        /// <summary>
        /// Компенсация вращения
        /// </summary>
        /// <param name="centerPoint"></param>
        private void SetAvgRotationVelocityToZero(Point centerPoint)
        {

            RotateTransform rTransform = new RotateTransform();
            Double d = CalcAvgRotationVelocity(centerPoint) * (180 / Math.PI);
            if (d < 1000)
            {
                rTransform.Angle = d;
                rTransform.CenterX = centerPoint.X;
                rTransform.CenterY = centerPoint.Y;

                foreach (var vertex in Vertices)
                {
                    vertex.position = rTransform.Transform(vertex.Position);
                }
            }

        }

        /// <summary>
        /// Компенсация смещения. Перемещение графа, чтоб центр масс был в заданной точке
        /// </summary>
        /// <param name="p"></param>
        public void MoveMassCenterTo(Point p)
        {
            Point nowP = CalcMassCenter();
            double dx = p.X - nowP.X;
            double dy = p.Y - nowP.Y;
            foreach (var vertex in Vertices)
            {
                vertex.position.X += dx;
                vertex.position.Y += dy;
            }
        }


        #endregion


        protected override void OnVertexAdded(Physical2DGraphVertex args)
        {
            base.OnVertexAdded(args);
            args._ownerGraph = this;
        }

        protected override void OnVertexRemoved(Physical2DGraphVertex args)
        {
            base.OnVertexRemoved(args);
        }

        protected override void OnEdgeRemoved(Physical2DGraphEdge<Physical2DGraphVertex> args)
        {
            base.OnEdgeRemoved(args);
        }

        protected void SetNewVertexPosition(Physical2DGraphVertex vertex)
        {
            vertex.position = this.NewVertexSeedParametres.GetNextPosition();
        }

        protected void SetNewVertexPositionNearOldVertex(Physical2DGraphVertex vertex, Physical2DGraphVertex oldVertex)
        {

            vertex.position = this.NewVertexSeedParametres.GetNextPositionNear(oldVertex.position);
        }



        //procedure tGeometry2DNet.SetNewNodePosition(xNewNode: tNode);
        //begin
        //  if xNewNode is tGeometry2DNode then
        //    begin
        //    tGeometry2DNode(xNewNode).fPosition := PointF2(random*GN_SeedAreaSize + GN_SeedAreaCenter, random*GN_SeedAreaSize + GN_SeedAreaCenter);
        //    end;
        //end;                                 

        //procedure tGeometry2DNet.SetNewNodePositionNearOldNode(xNewNode,
        //  xOldNode: tNode);
        //begin
        //  if (xNewNode is tGeometry2DNode) and (xOldNode is tGeometry2DNode) then
        //    begin
        //    tGeometry2DNode(xNewNode).fPosition := PointF2(tGeometry2DNode(xOldNode).Position.x + random*GN_ToBornNewNodeNearByDistance,
        //                                           tGeometry2DNode(xOldNode).Position.y + random*GN_ToBornNewNodeNearByDistance);
        //    end;

        //end;

        /// <summary>
        /// Разбиение на класстеры: для оптимизации силы гравитации (уход от сложности  o(n^2) )
        /// </summary>
        /// <param name="nodesCountInCluster"></param>
        public void DivideByCluster(int nodesCountInCluster)
        {
            clusters.Clear();

            if (this.Vertices.Count() == 0)
            {
                return;
            }

            foreach (var v in this.Vertices)
            {
                v.ClusterID = -1;
            }

            // текущий кластер
            int currentClusterID = 0;
            Physical2DGraphVertexCluster cluster = new Physical2DGraphVertexCluster();
            clusters.Add(cluster);
            cluster.ID = currentClusterID;


            // кандидаты на рассмотрение
            List<Physical2DGraphVertex> candidates = new List<Physical2DGraphVertex>();
            //int indCandidates = -1; // идекск



            // выбираем первый попавшийся узел
            Physical2DGraphVertex currentNode;


            currentNode = this.Vertices.First<Physical2DGraphVertex>();
            currentNode.ClusterID = cluster.ID;
            cluster.Vertices.Add(currentNode);

            candidates.Add(currentNode);


            // 1. ищем топологически nodesCountInCluster узлов ближайшие к текущему узлу, но без определённого кластера 
            int foundCount = 1;

            //candidates.Clear();

            while (candidates.Count > 0) // ещё не все узлы перебраны
            {
                currentNode = candidates.Last<Physical2DGraphVertex>();


                foreach (Edge<Physical2DGraphVertex> ed in this.AdjacentEdges(currentNode))
                {

                    Physical2DGraphVertex v = ed.GetOtherVertex(currentNode);
                    if ((v.ClusterID == -1) && (foundCount < nodesCountInCluster))
                    {
                        candidates.Add(v);
                        v.ClusterID = currentClusterID;
                        cluster.Vertices.Add(v);
                        foundCount++;
                    }

                    if (foundCount >= nodesCountInCluster)
                    {
                        currentClusterID++;
                        foundCount = 0;
                        cluster = new Physical2DGraphVertexCluster();
                        cluster.ID = currentClusterID;
                        clusters.Add(cluster);
                    }
                };


                candidates.Remove(currentNode);
            }

            // чтобы не было бага с "пустым" кластером
            if (cluster.Vertices.Count == 0)
            {
                clusters.Remove(cluster);
            }
        }



        /// <summary>
        /// Граф планаризован (частично или полностью). Т.е. найдены все грани которые можно уложить на плоскость. Если все узлы уложены, то граф планарный
        /// </summary>
        public bool IsPlanarized { get { return (fasets != null); } }


        private bool isPlanar;
        public bool IsPlanar { get { if (!IsPlanarized) { return Planarize(); } else { return isPlanar; } } }



        /// <summary>
        /// Возвращает распределение узлов по их степени
        /// </summary>
        /// <param name="graph"></param>
        /// <returns></returns>
        public Dictionary<int, int> GetVerticiesDistributionByDegree()
        {

            var vertParts = from x in this.Vertices
                            group x by this.AdjacentEdges(x).Count() into part
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
        /// возвращает распределение граней по их степени.
        /// </summary>
        /// <param name="graph"></param>
        /// <returns></returns>
        public Dictionary<int, int> GetFasetsDistributionByLength()
        {
            if (!IsPlanarized) { this.Planarize(); }
            
            var fasetParts = from x in fasets
                             //group x by x.Count
                                 group x by x.MinCycleLength
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
    }
}

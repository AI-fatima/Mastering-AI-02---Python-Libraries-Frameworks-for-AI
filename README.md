# Mastering AI 02 - Python Libraries/Frameworks for AI

Welcome to the `Mastering AI 02 - Python Libraries/Frameworks for AI` repository! This repository aims to provide a detailed and structured overview of essential Python libraries and frameworks for AI, including data handling, machine learning, deep learning, and more.

## Table of Contents

1. [Core Libraries for Data Handling](#21-core-libraries-for-data-handling)
   1. [Pandas](#211-pandas)
   2. [NumPy](#212-numpy)
   3. [Dask](#213-dask)
2. [Mathematics and Scientific Computation](#22-mathematics-and-scientific-computation)
   1. [SciPy](#221-scipy)
   2. [SymPy](#222-sympy)
3. [Machine Learning Libraries](#23-machine-learning-libraries)
   1. [Scikit-learn](#231-scikit-learn)
   2. [XGBoost](#232-xgboost)
   3. [LightGBM](#233-lightgbm)
4. [Deep Learning Frameworks](#24-deep-learning-frameworks)
   1. [TensorFlow](#241-tensorflow)
   2. [PyTorch](#242-pytorch)
   3. [Keras](#243-keras)
5. [Visualization Libraries](#25-visualization-libraries)
   1. [Matplotlib](#251-matplotlib)
   2. [Seaborn](#252-seaborn)
   3. [Plotly](#253-plotly)
6. [Natural Language Processing Libraries](#26-natural-language-processing-libraries)
   1. [NLTK](#261-nltk)
   2. [SpaCy](#262-spacy)
   3. [Hugging Face Transformers](#263-hugging-face-transformers)
7. [Computer Vision Libraries](#27-computer-vision-libraries)
   1. [OpenCV](#271-opencv)
   2. [Pillow](#272-pillow)
8. [Big Data and Distributed Computing Libraries](#28-big-data-and-distributed-computing-libraries)
   1. [PySpark](#281-pyspark)
   2. [Ray](#282-ray)
9. [MLOps and Model Deployment](#29-mlops-and-model-deployment)
   1. [MLflow](#291-mlflow)
   2. [Docker](#292-docker)
   3. [Kubernetes](#293-kubernetes)
10. [Glossary of AI-Specific Terminologies](#glossary-of-ai-specific-terminologies-libraries-context)

---

## 1. Core Libraries for Data Handling

### 1.1 Pandas
- **1.1.1 Data Structures**
  - **1.1.1.1 Series**: One-dimensional labeled array.
  - **1.1.1.2 DataFrame**: Two-dimensional labeled data structure.
- **1.1.2 Data Manipulation**
  - **1.1.2.1 Indexing & Selection**: Accessing data using labels or boolean conditions.
  - **1.1.2.2 Data Alignment**: Handling data with mismatched indexes.
  - **1.1.2.3 GroupBy**: Aggregating data by grouping.
- **1.1.3 Data Operations**
  - **1.1.3.1 Merging/Joining**: Combining multiple DataFrames.
  - **1.1.3.2 Reshaping**: Pivot tables and data reshaping.
  - **1.1.3.3 Data Cleaning**: Handling missing data, duplicates.

### 1.2 NumPy
- **1.2.1 Core Concepts**
  - **1.2.1.1 ndarray**: N-dimensional array.
  - **1.2.1.2 Data Types**: Numeric and non-numeric data types.
- **1.2.2 Array Operations**
  - **1.2.2.1 Mathematical Operations**: Element-wise operations.
  - **1.2.2.2 Aggregation**: Sum, mean, std, etc.
  - **1.2.2.3 Broadcasting**: Operations with arrays of different shapes.
- **1.2.3 Linear Algebra**
  - **1.2.3.1 Matrix Multiplication**: Dot product and matrix multiplication.
  - **1.2.3.2 Decomposition**: Eigenvalues, SVD.

### 1.3 Dask
- **1.3.1 Core Features**
  - **1.3.1.1 DataFrames**: Scalable DataFrame operations.
  - **1.3.1.2 Arrays**: Distributed array computing.
- **1.3.2 Parallel Computing**
  - **1.3.2.1 Task Scheduling**: Parallelize tasks across cores.
  - **1.3.2.2 Lazy Evaluation**: Deferred computation strategy.
- **1.3.3 Integration**
  - **1.3.3.1 Integration with Pandas**: Convert between Dask and Pandas DataFrames.
  - **1.3.3.2 Compatibility**: Use with other libraries for distributed computing.

## 2. Mathematics and Scientific Computation

### 2.1 SciPy
- **2.1.1 Core Submodules**
  - **2.1.1.1 linalg**: Linear algebra functions.
  - **2.1.1.2 optimize**: Optimization routines.
  - **2.1.1.3 signal**: Signal processing tools.
- **2.1.2 Optimization**
  - **2.1.2.1 Minimization**: Find local minima.
  - **2.1.2.2 Least Squares**: Fit data to a model.
- **2.1.3 Special Functions**
  - **2.1.3.1 Bessel Functions**: Common in signal processing.
  - **2.1.3.2 Gamma Functions**: Used in probability distributions.

### 2.2 SymPy
- **2.2.1 Symbolic Computation**
  - **2.2.1.1 Algebraic Expressions**: Manipulate algebraic formulas.
  - **2.2.1.2 Simplification**: Simplify expressions.
- **2.2.2 Calculus**
  - **2.2.2.1 Differentiation**: Compute derivatives.
  - **2.2.2.2 Integration**: Compute integrals.
- **2.2.3 Equation Solvers**
  - **2.2.3.1 Algebraic Equations**: Solve polynomial equations.
  - **2.2.3.2 Differential Equations**: Solve ODEs and PDEs.

## 3. Machine Learning Libraries

### 3.1 Scikit-learn
- **3.1.1 Supervised Learning**
  - **3.1.1.1 Classification**: Algorithms like SVM, decision trees.
  - **3.1.1.2 Regression**: Linear regression, ridge regression.
- **3.1.2 Unsupervised Learning**
  - **3.1.2.1 Clustering**: K-means, hierarchical clustering.
  - **3.1.2.2 Dimensionality Reduction**: PCA, t-SNE.
- **3.1.3 Model Evaluation**
  - **3.1.3.1 Cross-Validation**: Techniques for model validation.
  - **3.1.3.2 Metrics**: Accuracy, precision, recall, F1 score.

### 3.2 XGBoost
- **3.2.1 Gradient Boosting**
  - **3.2.1.1 Tree-Based Methods**: Boosted decision trees.
  - **3.2.1.2 Regularization**: Techniques to prevent overfitting.
- **3.2.2 Optimization**
  - **3.2.2.1 Hyperparameter Tuning**: Adjust boosting parameters.
  - **3.2.2.2 Early Stopping**: Stop training when performance plateaus.

### 3.3 LightGBM
- **3.3.1 Boosting Methods**
  - **3.3.1.1 Leaf-Wise Tree Growth**: Optimizes leaf-wise nodes.
  - **3.3.1.2 Histogram-Based Splitting**: Efficient binning for splits.
- **3.3.2 Performance**
  - **3.3.2.1 Speed and Efficiency**: Reduced training time and memory usage.
  - **3.3.2.2 Handling Large Datasets**: Efficient processing of large datasets.

## 4. Deep Learning Frameworks

### 4.1 TensorFlow
- **4.1.1 Core Components**
  - **4.1.1.1 Tensors**: Fundamental data structure.
  - **4.1.1.2 Computational Graphs**: Represent operations and dependencies.
- **4.1.2 Model Building**
  - **4.1.2.1 tf.keras**: High-level API for building models.
  - **4.1.2.2 Estimators**: Predefined models and training loops

.
- **4.1.3 Deployment**
  - **4.1.3.1 TensorFlow Serving**: Deploy models in production.
  - **4.1.3.2 TensorFlow Lite**: Optimize models for mobile and embedded devices.

### 4.2 PyTorch
- **4.2.1 Core Features**
  - **4.2.1.1 Tensors**: Multi-dimensional data structures.
  - **4.2.1.2 Autograd**: Automatic differentiation for gradient calculation.
- **4.2.2 Model Building**
  - **4.2.2.1 nn.Module**: Base class for building neural networks.
  - **4.2.2.2 DataLoader**: Efficient data loading and batching.
- **4.2.3 Advanced Topics**
  - **4.2.3.1 Transfer Learning**: Fine-tuning pre-trained models.
  - **4.2.3.2 Distributed Training**: Train models across multiple GPUs.

### 4.3 Keras
- **4.3.1 Core Concepts**
  - **4.3.1.1 Sequential API**: Simple model building.
  - **4.3.1.2 Functional API**: Complex model architectures.
- **4.3.2 Advanced Features**
  - **4.3.2.1 Custom Layers**: Define and use custom neural network layers.
  - **4.3.2.2 Transfer Learning**: Reuse models for new tasks.

## 5. Visualization Libraries

### 5.1 Matplotlib
- **5.1.1 Basic Plotting**
  - **5.1.1.1 Line Plots**: Visualize data trends.
  - **5.1.1.2 Scatter Plots**: Show relationships between variables.
- **5.1.2 Advanced Plotting**
  - **5.1.2.1 Subplots**: Multiple plots in a single figure.
  - **5.1.2.2 Customization**: Styles, labels, and annotations.

### 5.2 Seaborn
- **5.2.1 Statistical Plots**
  - **5.2.1.1 Histograms**: Distribution of data.
  - **5.2.1.2 KDE Plots**: Kernel density estimation for distribution visualization.
- **5.2.2 Categorical Plots**
  - **5.2.2.1 Box Plots**: Summarize data distributions.
  - **5.2.2.2 Violin Plots**: Combine box plot and KDE.

### 5.3 Plotly
- **5.3.1 Interactive Visualization**
  - **5.3.1.1 3D Plots**: Visualize three-dimensional data.
  - **5.3.1.2 Dashboards**: Create interactive web-based dashboards.
- **5.3.2 Animations**
  - **5.3.2.1 Animated Charts**: Display changes in data over time.
  - **5.3.2.2 Transitions**: Smoothly transition between different visualizations.

## 6. Natural Language Processing Libraries

### 6.1 NLTK
- **6.1.1 Text Processing**
  - **6.1.1.1 Tokenization**: Splitting text into words or sentences.
  - **6.1.1.2 Lemmatization**: Reducing words to their base form.
- **6.1.2 Linguistic Analysis**
  - **6.1.2.1 Part-of-Speech Tagging**: Identify grammatical categories.
  - **6.1.2.2 Named Entity Recognition**: Extract entities from text.

### 6.2 SpaCy
- **6.2.1 Core Features**
  - **6.2.1.1 Pre-trained Models**: Fast and efficient models for various languages.
  - **6.2.1.2 Dependency Parsing**: Analyze grammatical structure.
- **6.2.2 Customization**
  - **6.2.2.1 Custom Pipelines**: Build and customize processing pipelines.
  - **6.2.2.2 Training**: Train custom models on user-defined data.

### 6.3 Hugging Face Transformers
- **6.3.1 Model Architectures**
  - **6.3.1.1 GPT (Generative Pre-trained Transformer)**: Text generation models.
  - **6.3.1.2 BERT (Bidirectional Encoder Representations from Transformers)**: Contextual understanding models.
- **6.3.2 Tools and Libraries**
  - **6.3.2.1 Transformers Library**: Interface for pre-trained models.
  - **6.3.2.2 Tokenizers Library**: Efficient tokenization for transformer models.

## 7. Computer Vision Libraries

### 7.1 OpenCV
- **7.1.1 Image Processing**
  - **7.1.1.1 Basic Operations**: Cropping, resizing, rotating.
  - **7.1.1.2 Filters and Edge Detection**: Applying filters, detecting edges.
- **7.1.2 Object Detection**
  - **7.1.2.1 Haar Cascades**: Simple object detection method.
  - **7.1.2.2 DNN Module**: Use deep learning models for object detection.

### 7.2 Pillow
- **7.2.1 Image Manipulation**
  - **7.2.1.1 Opening and Saving Images**: Read and write image files.
  - **7.2.1.2 Image Filters**: Apply image enhancements and effects.
- **7.2.2 Image Operations**
  - **7.2.2.1 Cropping and Resizing**: Adjust image dimensions.
  - **7.2.2.2 Image Drawing**: Add text and shapes to images.

## 8. Big Data and Distributed Computing Libraries

### 8.1 PySpark
- **8.1.1 Core Concepts**
  - **8.1.1.1 RDD (Resilient Distributed Dataset)**: Fault-tolerant distributed data structure.
  - **8.1.1.2 DataFrames**: Distributed tabular data structure.
- **8.1.2 Data Processing**
  - **8.1.2.1 Transformations**: Apply transformations to RDDs/DataFrames.
  - **8.1.2.2 Actions**: Retrieve results from RDDs/DataFrames.
- **8.1.3 Integration**
  - **8.1.3.1 Integration with Hadoop**: Use with Hadoop ecosystems.
  - **8.1.3.2 Compatibility with Other Libraries**: Integration with libraries like NumPy and Pandas.

### 8.2 Ray
- **8.2.1 Core Features**
  - **8.2.1.1 Distributed Execution**: Parallelize Python functions.
  - **8.2.1.2 Actor Model**: Object-oriented concurrency model.
- **8.2.2 Libraries**
  - **8.2.2.1 Ray Tune**: Hyperparameter tuning and optimization.
  - **8.2.2.2 Ray RLlib**: Reinforcement learning library.

## 9. MLOps and Model Deployment

### 9.1 MLflow
- **9.1.1 Experiment Tracking**
  - **9.1.1.1 Logging Metrics**: Track performance metrics.
  - **9.1.1.2 Artifacts**: Store model files, training data.
- **9.1.2 Model Registry**
  - **9.1.2.1 Model Versioning**: Track different versions of models.
  - **9.1.2.2 Deployment**: Manage and deploy models to production.

### 9.2 Docker
- **9.2.1 Containerization**
  - **9.2.1.1 Images**: Create reusable images with dependencies.
  - **9.2.1.2 Containers**: Run applications in isolated environments.
- **9.2.2 Deployment**
  - **9.2.2.1 Orchestration**: Manage container deployment with Docker Compose.
  - **9.2.2.2 Security**: Best practices for securing containers.

### 9.3 Kubernetes
- **9.3.1 Orchestration**
  - **9.3.1.1 Pods**: Basic deployable units in Kubernetes.
  - **9.3.1.2 Services**: Define and expose network services.
- **9.3.2 Scalability**
  - **9.3.2.1 Auto-scaling**: Automatically scale applications based on load.
  - **9.3.2.2 Load Balancing**: Distribute traffic across multiple instances.

## 10. Glossary of AI-Specific Terminologies

This section will provide definitions and explanations for key AI terminologies relevant to the libraries and frameworks discussed in this repository. 

- **Machine Learning**: A type of AI that allows systems to learn from data.
- **Deep Learning**: A subset of machine learning using neural networks with many layers.
- **Supervised Learning**: Training a model on labeled data.
- **Unsupervised Learning**: Training a model on unlabeled data to find patterns.
- **Reinforcement Learning**: Training models to make decisions through rewards and penalties.
- **Optimization**: The process of making a model or algorithm more efficient.
- **Regularization**: Techniques to prevent overfitting in machine learning models.

---


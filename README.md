# 🤖 MLGenie - Open Source Machine Learning Platform

![MLGenie Dashboard](https://img.shields.io/badge/MLGenie-Dashboard-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## ✨ Modern, Reactive Machine Learning Platform

MLGenie is a comprehensive, open-source machine learning platform built with Streamlit, featuring a modern reactive dashboard with beautiful hover animations and comprehensive ML capabilities.

### 🚀 Features

#### 🎨 **Modern Reactive Dashboard**
- **Glassmorphism Design**: Beautiful translucent cards with backdrop blur effects
- **Hover Animations**: React-style transform animations with scale and translate effects
- **Real-time Statistics**: Global platform metrics updated in real-time
- **Responsive Layout**: Works perfectly on desktop and mobile devices

#### 🔧 **Core ML Capabilities**
- **Data Visualization**: Interactive charts and plots with Plotly
- **Feature Engineering**: Advanced preprocessing and feature transformation
- **ML Training**: Support for 28+ machine learning algorithms
- **Deep Learning**: Neural network training and optimization
- **Model Export**: Save and deploy trained models

#### 📊 **Platform Analytics**
- **Global Statistics**: Platform-wide metrics and usage data
- **Algorithm Distribution**: Visual breakdown of algorithm usage
- **Performance Trends**: Model accuracy trends over time
- **System Health**: Real-time monitoring of platform components

### 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/jatinnathh/MLGenie.git
cd MLGenie
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run home.py
```

### 📋 Requirements

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
plotly>=5.10.0
scikit-learn>=1.1.0
tensorflow>=2.10.0
torch>=1.12.0
seaborn>=0.11.0
matplotlib>=3.5.0
```

### 🎯 Usage

#### **Dashboard**
- View global platform statistics
- Monitor system health and performance
- Analyze algorithm usage trends
- Track platform activity in real-time

#### **Data Visualization**
- Upload CSV datasets
- Create interactive visualizations
- Explore data distributions and correlations
- Generate publication-ready plots

#### **Feature Engineering**
- Handle missing values
- Scale and normalize features
- Create new features
- Apply dimensionality reduction

#### **Model Training**
- Train classification and regression models
- Compare multiple algorithms
- Hyperparameter optimization
- Model performance evaluation

### 🏗️ Architecture

```
MLGenie/
├── modules/                    # Core ML modules
│   ├── modern_dashboard.py    # Reactive dashboard
│   ├── visualize.py          # Data visualization
│   ├── Feature_eng.py        # Feature engineering
│   ├── ML_training.py        # Machine learning
│   └── DL_training.py        # Deep learning
├── utils/                     # Utility functions
│   ├── shared.py            # Common utilities
│   ├── sidebar.py           # Navigation
│   └── styles.py            # Custom CSS
├── images/                   # Sample datasets
└── home.py                  # Main application
```

### 🎨 Design Features

#### **Modern CSS Animations**
```css
/* Hover Effects */
.metric-card:hover {
    transform: translateY(-12px) scale(1.03);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

/* Glassmorphism */
.metric-card {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}
```

#### **Interactive Elements**
- Smooth cubic-bezier transitions
- Progressive enhancement on hover
- Status indicators with pulse animations
- Gradient overlays and shadows

### 🌟 Platform Statistics

The dashboard displays real-time global statistics:

- **1,834+ Models Trained** across all users
- **247+ Datasets Processed** with TB of data
- **5,692+ Experiments** run on the platform
- **28 Different Algorithms** available
- **156 Active Projects** currently running

### 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web framework
- [Plotly](https://plotly.com/) for interactive visualizations
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- [TensorFlow](https://tensorflow.org/) and [PyTorch](https://pytorch.org/) for deep learning

### 📞 Support

- **Documentation**: [Wiki](https://github.com/jatinnathh/MLGenie/wiki)
- **Issues**: [GitHub Issues](https://github.com/jatinnathh/MLGenie/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jatinnathh/MLGenie/discussions)

---

<p align="center">
  <strong>Made with ❤️ for the ML community</strong><br>
  <em>MLGenie - Democratizing Machine Learning</em>
</p>
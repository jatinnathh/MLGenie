# MLGenie - Production Ready AutoML Platform

A comprehensive AutoML platform built with Streamlit, featuring data visualization, feature engineering, and both traditional ML and deep learning capabilities.

## 🚀 Quick Start

### Local Development
```bash
# Clone repository
git clone <your-repo-url>
cd MLGenie

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run home.py
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# Access at http://localhost:8501
```

### AWS EC2 Deployment
```bash
# Run on EC2 instance
chmod +x deployment/deploy-ec2.sh
./deployment/deploy-ec2.sh
```

## 📊 Features

- **Data Visualization**: Interactive charts including scatter plots, histograms, box plots, and radar charts
- **Feature Engineering**: Automated feature selection, encoding, and transformation
- **Machine Learning**: Support for classification, regression, and clustering
- **Deep Learning**: Neural network training with customizable architectures
- **Dashboard**: Real-time training progress and model performance tracking

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly
- **ML/DL**: Scikit-learn, TensorFlow, PyTorch
- **Data Processing**: Pandas, NumPy
- **Deployment**: Docker, AWS EC2

## 📁 Project Structure

```
MLGenie/
├── home.py                 # Main application entry point
├── modules/
│   ├── dashboard.py        # Dashboard functionality
│   ├── visualize.py        # Data visualization
│   ├── Feature_eng.py      # Feature engineering
│   ├── ML_training.py      # Machine learning training
│   └── DL_training.py      # Deep learning training
├── utils/
│   ├── shared.py           # Shared utilities
│   ├── sidebar.py          # Sidebar components
│   └── styles.py           # UI styling
├── deployment/
│   ├── deploy-ec2.sh       # AWS deployment script
│   └── README.md           # Deployment guide
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
└── docker-compose.yml     # Docker Compose setup
```

## 🔧 Development

### Prerequisites
- Python 3.11+
- Docker (for containerized deployment)
- Git

### Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run home.py`

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📦 Deployment

### Production Configuration
- Environment variables in `.env` file
- Docker health checks enabled
- Nginx reverse proxy configured
- Automated backups scheduled
- SSL/TLS certificates (Let's Encrypt)

### Monitoring
- Application logs via Docker Compose
- System monitoring with built-in tools
- Health check endpoints

## 🔒 Security

- Firewall configuration included
- SSL/TLS encryption supported
- Regular security updates automated
- Backup and recovery procedures

## 📈 Scaling

- Horizontal scaling with load balancers
- Database integration ready
- Cloud storage compatibility
- Microservices architecture support

## 🆘 Support

For deployment issues:
- Check logs: `docker-compose logs -f`
- Review documentation in `deployment/README.md`
- Monitor system resources

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with modern ML/AI frameworks and deployment best practices for production environments.

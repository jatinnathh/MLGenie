# MLGenie Enhanced Dashboard Setup Guide

Welcome to the enhanced MLGenie platform with comprehensive dashboard tracking!

## New Features Added ✨

### 📊 Dashboard Overview
- **Key Metrics Cards**: Datasets uploaded, feature engineering steps, models trained, best scores
- **Recent Activity Log**: Real-time tracking of all platform activities
- **Model Leaderboard**: Performance ranking of all trained models
- **Pending Tasks**: Queue management and progress tracking
- **System Resources**: CPU/Memory usage monitoring
- **Model Deployments**: Track deployed models and endpoints

### 🎯 What's New
1. **Professional Interface**: Clean, card-based design with no emojis
2. **Activity Tracking**: Every action is logged and displayed
3. **Performance Metrics**: Comprehensive model comparison
4. **Database Integration**: Optional MySQL backend for persistence
5. **Theme Toggle**: Light/Dark mode support
6. **Export Reports**: Download dashboard data as JSON

## Installation Instructions

### Option 1: Quick Start (Session-Based Tracking)
The dashboard will work immediately with session-state tracking:

```bash
# No additional setup required
# Just run MLGenie as usual
streamlit run home.py
```

### Option 2: Full Setup (MySQL Database Tracking)
For persistent tracking across sessions:

#### Step 1: Install Dependencies
```bash
pip install -r requirements_dashboard.txt
```

#### Step 2: Setup MySQL Database
```bash
# Install MySQL Server (if not already installed)
# Then run the database setup script:
python setup_database.py
```

Follow the prompts to:
- Enter MySQL connection details
- Create database and tables
- Insert sample data

#### Step 3: Update Database Configuration
Edit `utils/database.py` and update the connection parameters:

```python
# Update these lines in the connect() method:
self.connection = mysql.connector.connect(
    host="your_host",          # Usually "localhost"
    user="your_username",      # Usually "root"
    password="your_password",  # Your MySQL password
    database="mlgenie",        # Database name
    autocommit=True
)
```

## Usage Guide

### Dashboard Navigation
1. **Homepage**: Overview of all platform statistics
2. **Key Metrics**: Quick performance indicators
3. **Recent Activity**: Last 15 actions with timestamps
4. **Model Leaderboard**: Top performing models ranked by score
5. **Resource Monitor**: System usage tracking

### Activity Tracking
The system automatically tracks:
- ✅ Dataset uploads (filename, size, dimensions)
- ✅ Feature engineering operations (scaling, encoding, etc.)
- ✅ Model training (algorithm, metrics, duration)
- ✅ Prediction requests (input/output tracking)
- ✅ Export operations (model downloads)

### Dashboard Features

#### Metric Cards
- **Datasets Uploaded**: Total count with file info
- **Feature Engineering Steps**: Operations performed
- **Models Trained**: ML + DL model counts
- **Best Score**: Highest performing model
- **Jobs in Progress**: Active/queued tasks

#### Activity Feed
- Real-time activity log
- Status indicators (success/failed/in-progress)
- Timestamp tracking
- Activity type badges

#### Model Leaderboard
- Performance ranking (accuracy/R² score)
- Training duration comparison
- Algorithm distribution
- Interactive charts

#### System Resources
- CPU usage gauge
- Memory utilization
- GPU availability status
- Disk space monitoring

### Best Practices

#### For Development
1. Use session-based tracking for quick testing
2. Enable MySQL for production/multi-user scenarios
3. Check dashboard after each major operation
4. Export reports for analysis

#### For Production
1. Set up MySQL database for persistence
2. Configure proper user management
3. Monitor system resources regularly
4. Set up automated backups

## Troubleshooting

### Common Issues

#### Database Connection Failed
```
Error: Could not connect to database
Solution: Check MySQL server is running and credentials are correct
```

#### Missing Dependencies
```
Error: Module 'mysql.connector' not found
Solution: pip install mysql-connector-python
```

#### Dashboard Not Updating
```
Issue: Metrics not refreshing
Solution: Click "Refresh Dashboard" button or restart application
```

### Performance Tips

1. **Database Optimization**
   - Index frequently queried columns
   - Archive old activity logs periodically
   - Use connection pooling for high traffic

2. **UI Responsiveness**
   - Cache dashboard queries (built-in)
   - Limit activity log display (default: 15 items)
   - Use background tasks for heavy operations

3. **Memory Management**
   - Clear old session state periodically
   - Limit model storage in memory
   - Use file-based model persistence

## Advanced Configuration

### Custom Themes
Edit the CSS in `modules/dashboard.py` to customize:
- Color schemes
- Card layouts
- Typography
- Animations

### Database Schema
Modify `utils/database.py` to:
- Add custom tables
- Extend activity types
- Include additional metrics
- Support multi-tenancy

### Integration
Connect with external systems:
- REST APIs for model serving
- Cloud storage for datasets
- Monitoring systems
- CI/CD pipelines

## Support

### Documentation
- Check individual module docstrings
- Review database schema in `setup_database.py`
- Examine sample configurations

### Development
- All dashboard code is in `modules/dashboard.py`
- Database utilities in `utils/database.py`
- Activity logging in individual modules

### Customization
The dashboard is fully customizable:
- Add new metric cards
- Create custom visualizations
- Implement additional tracking
- Extend database schema

## Next Steps

1. **Run the Setup**: Choose session-based or database tracking
2. **Explore Dashboard**: Upload data and train models to see tracking
3. **Customize**: Modify colors, layouts, and metrics as needed
4. **Scale**: Implement database backend for production use

Enjoy your enhanced MLGenie experience! 🚀

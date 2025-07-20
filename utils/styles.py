# utils/styles.py
def get_global_styles():
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-blue: #1976d2;
        --secondary-blue: #42a5f5;
        --success-green: #4caf50;
        --warning-orange: #ff9800;
        --error-red: #f44336;
        --background: #fafafa;
        --surface: #ffffff;
        --text-primary: #212121;
        --text-secondary: #757575;
        --border-light: #e0e0e0;
    }
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-container {
        background: var(--surface);
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    .feature-card {
        background: var(--surface);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid var(--border-light);
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--primary-blue), var(--secondary-blue));
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    </style>
    """

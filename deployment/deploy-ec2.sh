#!/bin/bash

# MLGenie AWS EC2 Deployment Script
# Run this script on your EC2 instance

set -e

echo "🚀 Starting MLGenie deployment on AWS EC2..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Docker
echo "🐳 Installing Docker..."
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
sudo apt install -y docker-ce

# Install Docker Compose
echo "🔧 Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add user to docker group
sudo usermod -aG docker $USER

# Create application directory
echo "📁 Setting up application directory..."
sudo mkdir -p /opt/mlgenie
sudo chown -R $USER:$USER /opt/mlgenie
cd /opt/mlgenie

# Clone or copy application code (modify this based on your source)
echo "📥 Deploying application code..."
# If using git:
# git clone https://github.com/yourusername/mlgenie.git .
# Or copy files directly

# Create data directories
mkdir -p data models logs

# Set up environment
echo "🔐 Setting up environment..."
cat > .env << EOF
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENVIRONMENT=production
EOF

# Build and start the application
echo "🏗️  Building Docker container..."
docker-compose build

echo "🎬 Starting MLGenie application..."
docker-compose up -d

# Set up systemd service for auto-restart
echo "⚙️  Setting up systemd service..."
sudo tee /etc/systemd/system/mlgenie.service > /dev/null <<EOF
[Unit]
Description=MLGenie Docker Compose
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/mlgenie
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable mlgenie
sudo systemctl daemon-reload

# Set up nginx reverse proxy (optional)
echo "🌐 Setting up Nginx reverse proxy..."
sudo apt install -y nginx

sudo tee /etc/nginx/sites-available/mlgenie > /dev/null <<EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/mlgenie /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx

# Set up firewall
echo "🔥 Configuring firewall..."
sudo ufw allow ssh
sudo ufw allow http
sudo ufw allow https
sudo ufw --force enable

# Create backup script
echo "💾 Setting up backup system..."
cat > /opt/mlgenie/backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf "/opt/mlgenie/backups/mlgenie_backup_${DATE}.tar.gz" \
    /opt/mlgenie/data \
    /opt/mlgenie/models \
    --exclude="/opt/mlgenie/backups"
find /opt/mlgenie/backups -name "*.tar.gz" -mtime +7 -delete
EOF

chmod +x /opt/mlgenie/backup.sh
mkdir -p /opt/mlgenie/backups

# Add cron job for daily backup
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/mlgenie/backup.sh") | crontab -

echo "✅ MLGenie deployment completed successfully!"
echo ""
echo "🔗 Access your application at: http://$(curl -s ifconfig.me)"
echo "📊 Local access: http://localhost"
echo ""
echo "📋 Useful commands:"
echo "  - Check status: docker-compose ps"
echo "  - View logs: docker-compose logs -f"
echo "  - Restart: sudo systemctl restart mlgenie"
echo "  - Update: docker-compose pull && docker-compose up -d"
echo ""
echo "🛡️  Security reminders:"
echo "  - Change default passwords"
echo "  - Set up SSL/TLS certificates"
echo "  - Configure proper backup strategy"
echo "  - Monitor application logs"

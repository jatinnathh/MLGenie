# MLGenie AWS EC2 Deployment Guide

## Overview
This guide helps you deploy MLGenie AutoML platform on AWS EC2 with Docker containerization.

## Prerequisites
- AWS Account with EC2 access
- Basic knowledge of Linux/Ubuntu
- SSH key pair for EC2 access

## Step 1: Launch EC2 Instance

### Recommended Instance Configuration:
- **Instance Type**: t3.large or larger (minimum 2 vCPU, 8GB RAM)
- **AMI**: Ubuntu 22.04 LTS
- **Storage**: 30GB+ EBS storage
- **Security Group**: 
  - SSH (port 22) from your IP
  - HTTP (port 80) from anywhere
  - HTTPS (port 443) from anywhere
  - Custom port 8501 for direct Streamlit access (optional)

### Security Group Rules:
```
Type        Protocol    Port Range    Source
SSH         TCP         22           Your IP/0.0.0.0/0
HTTP        TCP         80           0.0.0.0/0
HTTPS       TCP         443          0.0.0.0/0
Custom      TCP         8501         0.0.0.0/0 (optional)
```

## Step 2: Connect to EC2 Instance

```bash
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

## Step 3: Deploy Application

### Option A: Automated Deployment (Recommended)
```bash
# Download and run deployment script
wget https://raw.githubusercontent.com/yourusername/mlgenie/main/deployment/deploy-ec2.sh
chmod +x deploy-ec2.sh
./deploy-ec2.sh
```

### Option B: Manual Deployment

1. **Update System:**
```bash
sudo apt update && sudo apt upgrade -y
```

2. **Install Docker:**
```bash
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update && sudo apt install -y docker-ce
```

3. **Install Docker Compose:**
```bash
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

4. **Setup Application:**
```bash
sudo mkdir -p /opt/mlgenie
sudo chown -R $USER:$USER /opt/mlgenie
cd /opt/mlgenie

# Copy your application files here
# Or clone from git: git clone https://github.com/yourusername/mlgenie.git .

# Create environment file
cat > .env << EOF
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENVIRONMENT=production
EOF
```

5. **Start Application:**
```bash
docker-compose build
docker-compose up -d
```

## Step 4: Access Your Application

- **Public URL**: `http://your-ec2-public-ip`
- **Direct Streamlit**: `http://your-ec2-public-ip:8501`

## Step 5: Production Optimizations

### SSL/TLS Setup with Let's Encrypt:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### Monitoring Setup:
```bash
# Install monitoring tools
sudo apt install htop iotop
docker-compose logs -f  # View application logs
```

### Auto-Restart Service:
```bash
sudo systemctl enable mlgenie
sudo systemctl start mlgenie
```

## Management Commands

### Application Management:
```bash
# View status
docker-compose ps

# View logs
docker-compose logs -f

# Restart application
docker-compose restart

# Update application
git pull
docker-compose build
docker-compose up -d
```

### System Management:
```bash
# Restart service
sudo systemctl restart mlgenie

# Check service status
sudo systemctl status mlgenie

# View system resources
htop
df -h
```

## Backup and Recovery

### Automated Backup:
The deployment script sets up daily backups at 2 AM:
```bash
# Manual backup
/opt/mlgenie/backup.sh

# View backups
ls -la /opt/mlgenie/backups/
```

### Restore from Backup:
```bash
cd /opt/mlgenie
tar -xzf backups/mlgenie_backup_YYYYMMDD_HHMMSS.tar.gz
docker-compose restart
```

## Troubleshooting

### Common Issues:

1. **Port 8501 not accessible:**
   - Check security group settings
   - Verify Docker container is running: `docker-compose ps`

2. **Application not starting:**
   - Check logs: `docker-compose logs`
   - Verify disk space: `df -h`
   - Check memory: `free -h`

3. **Permission errors:**
   - Fix ownership: `sudo chown -R $USER:$USER /opt/mlgenie`

4. **Docker issues:**
   - Restart Docker: `sudo systemctl restart docker`
   - Clean up: `docker system prune -f`

### Health Checks:
```bash
# Check if application is responding
curl http://localhost:8501

# Check Docker containers
docker-compose ps

# Check system resources
htop
```

## Cost Optimization

1. **Use Spot Instances** for development/testing
2. **Schedule shutdown** during non-business hours
3. **Monitor CloudWatch** for resource utilization
4. **Use EBS GP3** volumes for better price/performance

## Security Best Practices

1. **Regular Updates:**
```bash
sudo apt update && sudo apt upgrade -y
docker-compose pull
```

2. **Firewall Configuration:**
```bash
sudo ufw status
```

3. **SSH Key Management:**
   - Use strong SSH keys
   - Disable password authentication
   - Consider SSH bastion hosts

4. **Application Security:**
   - Use environment variables for secrets
   - Enable HTTPS with valid certificates
   - Regular security audits

## Support

For issues and support:
- Check application logs: `docker-compose logs`
- Review system logs: `sudo journalctl -u mlgenie`
- Monitor system resources: `htop`, `df -h`

## Next Steps

1. Set up domain name and SSL certificates
2. Configure automated backups to S3
3. Set up CloudWatch monitoring
4. Implement CI/CD pipeline
5. Configure load balancing for high availability

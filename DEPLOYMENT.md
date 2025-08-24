# Deployment Guide

This directory contains deployment files for Better Impuls Viewer.

## Frontend Deployment

Deploy the React frontend to a web server:

```bash
# Make the script executable (if not already)
chmod +x deploy-frontend.sh

# Deploy to /var/www/impuls-viewer
sudo ./deploy-frontend.sh
```

The script will:
1. Build the frontend using `npm run build`
2. Create backup of existing deployment
3. Copy files to `/var/www/impuls-viewer`
4. Set appropriate permissions for web server

Configure your web server (nginx/apache) to serve from `/var/www/impuls-viewer`.

### Example nginx configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    root /var/www/impuls-viewer;
    index index.html;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    # Proxy API requests to backend
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Backend Deployment

### Option 1: Docker (Recommended)

Build and run using Docker:

```bash
# Build the image
docker build -t better-impuls-viewer-backend .

# Run with docker-compose
docker-compose up -d

# Or run directly
docker run -d -p 8000:8000 --name impuls-backend better-impuls-viewer-backend
```

### Option 2: Direct Python

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Run the application
python backend/app.py --port 8000
```

## Full Deployment

1. **Deploy Backend**: Use Docker or direct Python method above
2. **Deploy Frontend**: Run `./deploy-frontend.sh`
3. **Configure Web Server**: Point to `/var/www/impuls-viewer` and proxy API to port 8000
4. **Test**: Visit your domain and verify both frontend and API work

## Notes

- Backend runs on port 8000 by default
- Frontend expects backend API to be available at `/api/` endpoint
- Ensure firewall allows access to port 8000 if using direct Python deployment
- Docker deployment is recommended for production use
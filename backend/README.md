# Docker Usage for Backend

## Building the Docker Image

From the repository root directory, run:

```bash
docker build -f backend/Dockerfile -t better-impuls-viewer-backend .
```

## Running the Container

Run the backend server in a Docker container:

```bash
docker run --rm --name backend -p 8000:8000 better-impuls-viewer-backend
```

The backend API will be available at `http://localhost:8000`

## API Endpoints

- `GET /` - Root endpoint
- `GET /stars` - List available star numbers
- `GET /telescopes/{star_number}` - Get telescopes for a star
- `GET /campaigns/{star_number}/{telescope}` - Get top 3 campaigns
- `GET /data/{star_number}/{telescope}/{campaign_id}` - Get processed light curve data
- `GET /periodogram/{star_number}/{telescope}/{campaign_id}` - Get Lomb-Scargle periodogram
- `GET /phase_fold/{star_number}/{telescope}/{campaign_id}?period={period}` - Get phase-folded data

## Testing the API

```bash
# Test basic connectivity
curl http://localhost:8000/

# Get available stars
curl http://localhost:8000/stars

# Get telescopes for star 1
curl http://localhost:8000/telescopes/1

# Get campaigns for star 1, hubble telescope
curl http://localhost:8000/campaigns/1/hubble
```
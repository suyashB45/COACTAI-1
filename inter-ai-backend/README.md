# Coaching API Deployment

## Docker Build & Run

### Build the image:
```bash
docker build -t coaching-api .
```

### Run the container:
```bash
docker run -p 8000:8000 \
  -e AZURE_SPEECH_KEY="your_speech_key" \
  -e AZURE_SPEECH_REGION="your_region" \
  -v $(pwd)/reports:/app/reports \
  coaching-api
```

### Using Docker Compose:
```bash
# Set environment variables
export AZURE_SPEECH_KEY="your_speech_key"
export AZURE_SPEECH_REGION="your_region"

# Run with compose
docker-compose up -d
```

## API Endpoints

- **POST** `/session/start` - Start a new coaching session
- **POST** `/api/session/{id}/chat` - Send message in session
- **GET** `/api/session/{id}` - Get session details
- **POST** `/api/session/{id}/complete` - Force complete session
- **GET** `/api/report/{id}` - Download session report

## Environment Variables

- `PORT` - Server port (default: 8000)
- `AZURE_SPEECH_KEY` - Azure Speech Services key
- `AZURE_SPEECH_REGION` - Azure Speech Services region

## Volume Mounts

- `/app/reports` - Generated PDF reports storage

## Health Check

The container includes a health check on `http://localhost:8000/`

# AI Job Queue Frontend

A modern React-based frontend for the AI Job Queue System, built with Vite, TypeScript, and Tailwind CSS.

## Features

- 🧠 **Training Job Submission**: Submit ML training jobs with various frameworks
- 🔮 **Inference Job Submission**: Run inference using trained models
- 📋 **Job Monitoring**: Real-time job status tracking and history
- 📊 **System Status**: Monitor workers, frameworks, and system health
- 🎨 **Modern UI**: Responsive design with Tailwind CSS
- ⚡ **Fast Development**: Vite for lightning-fast hot reloading
- 🔧 **TypeScript**: Full type safety and IntelliSense support

## Tech Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Axios** - HTTP client
- **React Router** - Client-side routing
- **Lucide React** - Icon library

## Quick Start

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Backend API running on `http://localhost:8000`

### Development

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

3. **Open browser:**
   Navigate to `http://localhost:3000`

### Production Build

1. **Build for production:**
   ```bash
   npm run build
   ```

2. **Preview production build:**
   ```bash
   npm run preview
   ```

## Project Structure

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── forms/          # Job submission forms
│   │   ├── jobs/           # Job monitoring components
│   │   ├── dashboard/      # System status components
│   │   └── ui/             # Reusable UI components
│   ├── lib/                # Utilities and API client
│   ├── types/              # TypeScript type definitions
│   ├── App.tsx             # Main app component
│   └── main.tsx            # App entry point
├── public/                 # Static assets
├── index.html              # HTML template
├── package.json            # Dependencies and scripts
├── vite.config.ts          # Vite configuration
├── tailwind.config.js      # Tailwind CSS configuration
└── Dockerfile              # Docker configuration
```

## API Integration

The frontend communicates with the backend API through:

- **Base URL**: `http://localhost:8000` (configurable via `VITE_API_URL`)
- **Endpoints**:
  - `POST /jobs/training` - Submit training jobs
  - `POST /jobs/inference` - Submit inference jobs
  - `GET /jobs/{job_id}/status` - Get job status
  - `GET /jobs` - List all jobs
  - `GET /health` - System health check
  - `GET /workers/status` - Worker status
  - `GET /frameworks` - Available frameworks

## Environment Variables

Create a `.env` file in the frontend directory:

```env
# API Configuration
VITE_API_URL=http://localhost:8000

# Development Configuration
VITE_DEV_MODE=true
```

## Docker Support

### Development with Docker

```bash
# Build and run with Docker Compose
docker-compose up frontend

# Or build manually
docker build -t ai-job-queue-frontend .
docker run -p 3000:80 ai-job-queue-frontend
```

### Production Deployment

The frontend is configured to work with the full stack via Docker Compose:

```bash
# Start complete system
docker-compose up -d

# Access frontend
open http://localhost:3000
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Features Overview

### Training Jobs

- **Model Selection**: Choose from BERT, GPT, ResNet, VGG, Inception, MobileNet, Random Forest, SVM, Logistic Regression
- **Framework Override**: Manually select framework or let the system auto-select
- **GPU Support**: Toggle GPU acceleration
- **Hyperparameters**: Configure model-specific training parameters
- **Data Path**: Specify training data location

### Inference Jobs

- **Model Selection**: Choose from completed training jobs
- **Input Data**: Provide data for inference
- **Parameters**: Configure inference settings (temperature, max length, etc.)

### Job Monitoring

- **Real-time Status**: Live updates of job progress
- **Job History**: View all submitted jobs
- **Detailed Results**: See training metrics and model information
- **Error Handling**: Clear error messages and debugging info

### System Status

- **Health Monitoring**: API and Redis connection status
- **Worker Status**: Available workers by type
- **Framework Info**: Supported frameworks and configurations

## Styling

The frontend uses Tailwind CSS with a custom design system:

- **Colors**: Primary, success, warning, error variants
- **Components**: Reusable UI components with consistent styling
- **Responsive**: Mobile-first responsive design
- **Dark Mode**: Ready for dark mode implementation

## Development Tips

1. **Hot Reloading**: Changes are reflected immediately in the browser
2. **Type Safety**: TypeScript provides full type checking
3. **API Mocking**: Use the backend API or mock responses for development
4. **Component Library**: Reusable UI components in `src/components/ui/`
5. **Error Handling**: Comprehensive error handling with user-friendly messages

## Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Ensure backend is running on `http://localhost:8000`
   - Check `VITE_API_URL` environment variable

2. **Build Errors**
   - Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`
   - Check TypeScript errors: `npm run lint`

3. **Styling Issues**
   - Ensure Tailwind CSS is properly configured
   - Check for conflicting CSS classes

### Debug Mode

Enable debug logging by setting `VITE_DEV_MODE=true` in your `.env` file.

## Contributing

1. Follow the existing code style and patterns
2. Add TypeScript types for new features
3. Update documentation for new components
4. Test with the backend API
5. Ensure responsive design works on all screen sizes

## License

This project is part of the AI Job Queue System and follows the same license terms.

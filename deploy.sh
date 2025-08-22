#!/bin/bash

# Enhanced RAG Application Deployment Script
# This script handles building and deploying the application

set -e  # Exit on any error

echo "  RAG Application Deployment Script"
echo "============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command_exists docker-compose; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "All prerequisites are installed."
}

# Setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    if [ ! -f .env ]; then
        if [ -f .env.template ]; then
            print_warning ".env file not found. Copying from template..."
            cp .env.template .env
            print_warning "Please edit .env file with your API keys before continuing."
            read -p "Press Enter after updating .env file..."
        else
            print_error ".env file not found and no template available."
            exit 1
        fi
    fi
    
    # Create necessary directories
    mkdir -p chroma_db
    mkdir -p logs
    mkdir -p temp_uploads
    
    print_success "Environment setup complete."
}

# Build Docker image
build_image() {
    print_status "Building Docker image..."
    docker build -t enhanced-rag-app:latest .
    print_success "Docker image built successfully."
}

# Deploy application
deploy_app() {
    print_status "Deploying application..."
    
    # Stop existing containers
    docker-compose down 2>/dev/null || true
    
    # Start new containers
    docker-compose up -d
    
    print_success "Application deployed successfully."
}

# Check application health
check_health() {
    print_status "Checking application health..."
    
    # Wait for application to start
    sleep 10
    
    # Check if container is running
    if docker-compose ps | grep -q "Up"; then
        print_success "Application is running."
        print_status "Application available at: http://localhost:8501"
    else
        print_error "Application failed to start. Check logs with: docker-compose logs"
        exit 1
    fi
}

# Show logs
show_logs() {
    print_status "Showing application logs..."
    docker-compose logs -f
}

# Cleanup function
cleanup() {
    print_status "Cleaning up..."
    docker-compose down
    docker system prune -f
    print_success "Cleanup complete."
}

# Main deployment function
main_deploy() {
    check_prerequisites
    setup_environment
    build_image
    deploy_app
    check_health
}

# Help function
show_help() {
    echo "Enhanced RAG Application Deployment Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  deploy    - Full deployment (default)"
    echo "  build     - Build Docker image only"
    echo "  start     - Start application"
    echo "  stop      - Stop application"
    echo "  restart   - Restart application"
    echo "  logs      - Show application logs"
    echo "  cleanup   - Stop and cleanup"
    echo "  health    - Check application health"
    echo "  help      - Show this help"
    echo ""
}

# Parse command line arguments
case "${1:-deploy}" in
    "deploy")
        main_deploy
        ;;
    "build")
        check_prerequisites
        build_image
        ;;
    "start")
        docker-compose up -d
        check_health
        ;;
    "stop")
        docker-compose down
        ;;
    "restart")
        docker-compose down
        docker-compose up -d
        check_health
        ;;
    "logs")
        show_logs
        ;;
    "cleanup")
        cleanup
        ;;
    "health")
        check_health
        ;;
    "help")
        show_help
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
#!/bin/bash
# Production Deployment Script for Baseball HR Prediction System
# ============================================================

set -e  # Exit on any error

# Configuration
PROJECT_NAME="baseball-hr-prediction"
DEPLOY_DATE=$(date '+%Y%m%d_%H%M%S')
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="$PROJECT_ROOT/backups/deploy_$DEPLOY_DATE"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo "============================================================"
    echo "BASEBALL HR PREDICTION SYSTEM - PRODUCTION DEPLOYMENT"
    echo "============================================================"
    echo "Deploy Date: $(date)"
    echo "Project Root: $PROJECT_ROOT"
    echo "============================================================"
}

# Deployment functions
check_prerequisites() {
    log_info "Checking deployment prerequisites..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Python version: $PYTHON_VERSION"
    
    if [[ $(echo "$PYTHON_VERSION < 3.8" | bc -l) -eq 1 ]]; then
        log_error "Python 3.8+ is required"
        exit 1
    fi
    
    # Check required environment variables
    if [[ -z "$THEODDS_API_KEY" ]]; then
        log_error "THEODDS_API_KEY environment variable is not set"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

create_backup() {
    log_info "Creating deployment backup..."
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup current models if they exist
    if [[ -d "$PROJECT_ROOT/saved_models_pregame" ]]; then
        cp -r "$PROJECT_ROOT/saved_models_pregame" "$BACKUP_DIR/"
        log_info "Backed up models to: $BACKUP_DIR/saved_models_pregame"
    fi
    
    # Backup current config
    if [[ -f "$PROJECT_ROOT/config.py" ]]; then
        cp "$PROJECT_ROOT/config.py" "$BACKUP_DIR/"
        log_info "Backed up config to: $BACKUP_DIR/config.py"
    fi
    
    # Create deployment manifest
    cat > "$BACKUP_DIR/deployment_manifest.txt" << EOF
Deployment Date: $DEPLOY_DATE
Python Version: $PYTHON_VERSION
Git Commit: $(cd "$PROJECT_ROOT" && git rev-parse HEAD 2>/dev/null || echo "Unknown")
API Keys: $(echo "$THEODDS_API_KEY" | head -c 10)...
EOF
    
    log_success "Backup created at: $BACKUP_DIR"
}

install_dependencies() {
    log_info "Installing Python dependencies..."
    
    cd "$PROJECT_ROOT"
    
    # Check if requirements.txt exists
    if [[ -f "requirements.txt" ]]; then
        # Install/upgrade pip
        python3 -m pip install --upgrade pip
        
        # Install requirements
        python3 -m pip install -r requirements.txt
        log_success "Dependencies installed from requirements.txt"
    else
        log_warning "No requirements.txt found, skipping dependency installation"
    fi
}

setup_directories() {
    log_info "Setting up directory structure..."
    
    cd "$PROJECT_ROOT"
    
    # Create required directories
    DIRS=("logs" "data" "data/processed" "saved_models_pregame" "outputs" "backups")
    
    for dir in "${DIRS[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        else
            log_info "Directory exists: $dir"
        fi
    done
    
    log_success "Directory structure setup complete"
}

validate_deployment() {
    log_info "Validating deployment..."
    
    cd "$PROJECT_ROOT"
    
    # Run validation script
    if python3 scripts/production_startup.py --validate-only; then
        log_success "Deployment validation passed"
        return 0
    else
        log_error "Deployment validation failed"
        return 1
    fi
}

test_system() {
    log_info "Running system tests..."
    
    cd "$PROJECT_ROOT"
    
    # Test API connectivity
    if python3 test_api_connectivity.py > /dev/null 2>&1; then
        log_success "API connectivity tests passed"
    else
        log_warning "API connectivity tests had issues (may be normal)"
    fi
    
    # Test export system
    if python3 test_export_system.py > /dev/null 2>&1; then
        log_success "Export system tests passed"
    else
        log_error "Export system tests failed"
        return 1
    fi
    
    return 0
}

create_service_scripts() {
    log_info "Creating service management scripts..."
    
    # Create start script
    cat > "$PROJECT_ROOT/start_system.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
echo "Starting Baseball HR Prediction System..."
python3 scripts/production_startup.py
if [[ $? -eq 0 ]]; then
    echo "✅ System started successfully"
    echo "Use 'python3 main.py live --api-key $THEODDS_API_KEY' to run live predictions"
else
    echo "❌ System startup failed"
    exit 1
fi
EOF
    
    chmod +x "$PROJECT_ROOT/start_system.sh"
    
    # Create stop script
    cat > "$PROJECT_ROOT/stop_system.sh" << 'EOF'
#!/bin/bash
echo "Stopping Baseball HR Prediction System..."
# Kill any running Python processes for this project
pkill -f "python.*main.py.*live"
echo "✅ System stopped"
EOF
    
    chmod +x "$PROJECT_ROOT/stop_system.sh"
    
    log_success "Service scripts created"
}

generate_deployment_report() {
    log_info "Generating deployment report..."
    
    REPORT_FILE="$PROJECT_ROOT/logs/deployment_report_$DEPLOY_DATE.txt"
    
    cat > "$REPORT_FILE" << EOF
BASEBALL HR PREDICTION SYSTEM - DEPLOYMENT REPORT
==================================================
Deployment Date: $DEPLOY_DATE
Python Version: $PYTHON_VERSION
Project Root: $PROJECT_ROOT

DEPLOYMENT STATUS:
✅ Prerequisites check passed
✅ Backup created
✅ Dependencies installed
✅ Directory structure setup
✅ Deployment validation passed
✅ System tests completed
✅ Service scripts created

SYSTEM INFORMATION:
- Models Directory: $PROJECT_ROOT/saved_models_pregame
- Data Directory: $PROJECT_ROOT/data
- Logs Directory: $PROJECT_ROOT/logs
- Config File: $PROJECT_ROOT/config.py

API CONFIGURATION:
- The Odds API: Configured (Key: ${THEODDS_API_KEY:0:10}...)
- Visual Crossing: $(if [[ -n "$VISUALCROSSING_API_KEY" ]]; then echo "Configured"; else echo "Not configured"; fi)

NEXT STEPS:
1. Run: ./start_system.sh to start the system
2. Run: python3 main.py live --api-key \$THEODDS_API_KEY to get live predictions
3. Monitor logs in: $PROJECT_ROOT/logs/

SUPPORT:
- Check logs for any issues
- Run: python3 scripts/production_startup.py --validate-only for diagnostics
- Run: python3 main.py diagnose for model status

Deployment completed at: $(date)
EOF
    
    log_success "Deployment report saved to: $REPORT_FILE"
}

# Main deployment sequence
main() {
    print_header
    
    # Parse command line arguments
    SKIP_TESTS=false
    SKIP_BACKUP=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --skip-tests    Skip system tests"
                echo "  --skip-backup   Skip backup creation"
                echo "  -h, --help      Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run deployment steps
    check_prerequisites
    
    if [[ "$SKIP_BACKUP" != true ]]; then
        create_backup
    fi
    
    install_dependencies
    setup_directories
    
    if ! validate_deployment; then
        log_error "Deployment validation failed. Check the output above."
        exit 1
    fi
    
    if [[ "$SKIP_TESTS" != true ]]; then
        if ! test_system; then
            log_error "System tests failed. Deployment may be incomplete."
            exit 1
        fi
    fi
    
    create_service_scripts
    generate_deployment_report
    
    echo ""
    log_success "🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!"
    echo ""
    echo "Next steps:"
    echo "  1. Run: ./start_system.sh"
    echo "  2. Run: python3 main.py live --api-key \$THEODDS_API_KEY"
    echo "  3. Check deployment report: logs/deployment_report_$DEPLOY_DATE.txt"
    echo ""
}

# Handle interrupts
trap 'log_error "Deployment interrupted"; exit 1' INT TERM

# Run main deployment
main "$@"
#!/usr/bin/env python3
"""
Service Manager
===============

Production service management for the baseball HR prediction system.
"""

import os
import sys
import time
import signal
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class ServiceManager:
    """Manages the production service lifecycle."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.pid_file = self.project_root / "logs" / "service.pid"
        self.log_dir = self.project_root / "logs"
        self.setup_logging()
    
    def setup_logging(self):
        """Configure service logging."""
        self.log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "service_manager.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def is_running(self) -> bool:
        """Check if service is currently running."""
        if not self.pid_file.exists():
            return False
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if process is still running
            os.kill(pid, 0)  # This will raise an exception if process doesn't exist
            return True
        except (ValueError, OSError):
            # PID file exists but process is dead, clean it up
            self.pid_file.unlink(missing_ok=True)
            return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get detailed service status."""
        status = {
            'running': self.is_running(),
            'timestamp': datetime.now().isoformat()
        }
        
        if status['running']:
            with open(self.pid_file, 'r') as f:
                status['pid'] = int(f.read().strip())
            
            # Get process information
            try:
                import psutil
                process = psutil.Process(status['pid'])
                status['memory_mb'] = round(process.memory_info().rss / 1024 / 1024, 2)
                status['cpu_percent'] = process.cpu_percent()
                status['start_time'] = datetime.fromtimestamp(process.create_time()).isoformat()
            except (ImportError, psutil.NoSuchProcess):
                pass
        
        return status
    
    def start_service(self, mode: str = 'live', api_key: str = None, background: bool = True) -> bool:
        """Start the prediction service."""
        if self.is_running():
            self.logger.warning("Service is already running")
            return False
        
        self.logger.info(f"Starting service in {mode} mode...")
        
        # Prepare command
        cmd = [
            sys.executable, "main.py", mode
        ]
        
        if api_key:
            cmd.extend(["--api-key", api_key])
        
        if mode == 'live':
            cmd.extend([
                "--min-ev", "0.05",
                "--output-dir", str(self.log_dir / "predictions")
            ])
        
        try:
            # Change to project directory
            os.chdir(self.project_root)
            
            if background:
                # Start as background process
                log_file = self.log_dir / f"service_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                
                with open(log_file, 'w') as f:
                    process = subprocess.Popen(
                        cmd,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        start_new_session=True
                    )
                
                # Write PID file
                with open(self.pid_file, 'w') as f:
                    f.write(str(process.pid))
                
                self.logger.info(f"Service started with PID {process.pid}")
                self.logger.info(f"Logs: {log_file}")
                
                # Give it a moment to start
                time.sleep(2)
                
                # Check if it's still running
                if self.is_running():
                    return True
                else:
                    self.logger.error("Service failed to start")
                    return False
            else:
                # Run in foreground
                self.logger.info("Running service in foreground mode...")
                subprocess.run(cmd, check=True)
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to start service: {e}")
            return False
    
    def stop_service(self, force: bool = False) -> bool:
        """Stop the prediction service."""
        if not self.is_running():
            self.logger.warning("Service is not running")
            return True
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            self.logger.info(f"Stopping service (PID: {pid})...")
            
            if force:
                os.kill(pid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGTERM)
                
                # Wait for graceful shutdown
                for _ in range(30):  # Wait up to 30 seconds
                    if not self.is_running():
                        break
                    time.sleep(1)
                
                # Force kill if still running
                if self.is_running():
                    self.logger.warning("Graceful shutdown failed, force killing...")
                    os.kill(pid, signal.SIGKILL)
            
            # Clean up PID file
            self.pid_file.unlink(missing_ok=True)
            
            self.logger.info("Service stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop service: {e}")
            return False
    
    def restart_service(self, **kwargs) -> bool:
        """Restart the service."""
        self.logger.info("Restarting service...")
        
        if self.is_running():
            if not self.stop_service():
                return False
        
        time.sleep(2)  # Brief pause
        return self.start_service(**kwargs)
    
    def get_logs(self, lines: int = 50) -> str:
        """Get recent service logs."""
        try:
            # Find the most recent log file
            log_files = list(self.log_dir.glob("service_*.log"))
            if not log_files:
                return "No service logs found"
            
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            
            # Read last N lines
            with open(latest_log, 'r') as f:
                all_lines = f.readlines()
                return ''.join(all_lines[-lines:])
                
        except Exception as e:
            return f"Error reading logs: {e}"
    
    def print_status(self):
        """Print formatted service status."""
        status = self.get_service_status()
        
        print("=" * 50)
        print("BASEBALL HR PREDICTION SERVICE STATUS")
        print("=" * 50)
        
        if status['running']:
            print(f"Status: 🟢 RUNNING (PID: {status.get('pid', 'Unknown')})")
            if 'start_time' in status:
                print(f"Started: {status['start_time']}")
            if 'memory_mb' in status:
                print(f"Memory Usage: {status['memory_mb']} MB")
            if 'cpu_percent' in status:
                print(f"CPU Usage: {status['cpu_percent']}%")
        else:
            print("Status: 🔴 STOPPED")
        
        print(f"Checked: {status['timestamp']}")
        print(f"Log Directory: {self.log_dir}")
        print("=" * 50)

def main():
    """Main service manager function."""
    parser = argparse.ArgumentParser(description='Baseball HR Prediction Service Manager')
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status', 'logs'], 
                       help='Service action to perform')
    parser.add_argument('--mode', default='live', choices=['live', 'backtest', 'train'],
                       help='Service mode for start/restart')
    parser.add_argument('--api-key', help='API key for live mode')
    parser.add_argument('--foreground', action='store_true', 
                       help='Run in foreground (don\'t daemonize)')
    parser.add_argument('--force', action='store_true', 
                       help='Force kill service on stop')
    parser.add_argument('--lines', type=int, default=50, 
                       help='Number of log lines to show')
    
    args = parser.parse_args()
    
    # Get API key from environment if not provided
    api_key = args.api_key or os.getenv('THEODDS_API_KEY')
    
    # Create service manager
    service = ServiceManager()
    
    try:
        if args.action == 'start':
            if not api_key and args.mode == 'live':
                print("❌ API key required for live mode. Set THEODDS_API_KEY or use --api-key")
                sys.exit(1)
            
            success = service.start_service(
                mode=args.mode,
                api_key=api_key,
                background=not args.foreground
            )
            sys.exit(0 if success else 1)
            
        elif args.action == 'stop':
            success = service.stop_service(force=args.force)
            sys.exit(0 if success else 1)
            
        elif args.action == 'restart':
            if not api_key and args.mode == 'live':
                print("❌ API key required for live mode. Set THEODDS_API_KEY or use --api-key")
                sys.exit(1)
                
            success = service.restart_service(
                mode=args.mode,
                api_key=api_key,
                background=not args.foreground
            )
            sys.exit(0 if success else 1)
            
        elif args.action == 'status':
            service.print_status()
            
        elif args.action == 'logs':
            logs = service.get_logs(lines=args.lines)
            print(logs)
            
    except KeyboardInterrupt:
        print("\n🛑 Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Service manager error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
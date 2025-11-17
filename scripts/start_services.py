#!/usr/bin/env python3
"""
Service startup script for Fresh Supply Chain Intelligence System
Starts all services in the correct order
"""

import subprocess
import time
import sys
import os
import signal
import threading
from pathlib import Path

class ServiceManager:
    """Manages multiple services"""
    
    def __init__(self):
        self.processes = {}
        self.running = True
        
    def start_service(self, name, command, cwd=None, env=None):
        """Start a service"""
        print(f"Loading: Starting {name}...")
        
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[name] = process
            print(f"[SUCCESS] {name} started with PID {process.pid}")
            
            # Start monitoring thread
            thread = threading.Thread(target=self._monitor_service, args=(name, process))
            thread.daemon = True
            thread.start()
            
            return process
            
        except Exception as e:
            print(f"[ERROR] Failed to start {name}: {e}")
            return None
    
    def _monitor_service(self, name, process):
        """Monitor a service for output and errors"""
        while self.running and process.poll() is None:
            try:
                # Read output
                output = process.stdout.readline()
                if output:
                    print(f"[{name}] {output.strip()}")
                
                # Read errors
                error = process.stderr.readline()
                if error:
                    print(f"[{name} ERROR] {error.strip()}")
                    
            except Exception as e:
                print(f"[{name} MONITOR ERROR] {e}")
                break
    
    def stop_service(self, name):
        """Stop a service"""
        if name in self.processes:
            process = self.processes[name]
            if process.poll() is None:
                print(f"🛑 Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                    print(f"[SUCCESS] {name} stopped")
                except subprocess.TimeoutExpired:
                    print(f"WARNING: {name} didn't stop gracefully, killing...")
                    process.kill()
                    process.wait()
                    print(f"[SUCCESS] {name} killed")
            del self.processes[name]
    
    def stop_all(self):
        """Stop all services"""
        print("\n🛑 Stopping all services...")
        self.running = False
        
        for name in list(self.processes.keys()):
            self.stop_service(name)
        
        print("[SUCCESS] All services stopped")
    
    def wait_for_services(self):
        """Wait for all services to complete"""
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal...")
            self.stop_all()

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking: Checking dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'pandas', 'numpy', 'torch', 
        'plotly', 'dash', 'redis', 'sqlalchemy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"[ERROR] Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("[SUCCESS] All dependencies found")
    return True

def check_database():
    """Check database connectivity"""
    print("Checking: Checking database connectivity...")
    
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config.database_config import test_database_connection
        
        if test_database_connection():
            print("[SUCCESS] Database connection successful")
            return True
        else:
            print("[ERROR] Database connection failed")
            print("Please check your database configuration in .env file")
            return False
            
    except Exception as e:
        print(f"[ERROR] Database check failed: {e}")
        return False

def start_services():
    """Start all services"""
    
    print("🥬 Fresh Supply Chain Intelligence System")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check database
    if not check_database():
        print("WARNING: Database not available, some features may not work")
    
    # Create service manager
    manager = ServiceManager()
    
    # Set up signal handlers
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}")
        manager.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    
    # Start services
    services = [
        {
            'name': 'API Server',
            'command': ['python', 'api/main.py'],
            'cwd': str(project_root)
        },
        {
            'name': 'Dashboard',
            'command': ['python', 'dashboard/app.py'],
            'cwd': str(project_root)
        }
    ]
    
    # Start each service
    for service in services:
        manager.start_service(**service)
        time.sleep(2)  # Give each service time to start
    
    print("\nSUCCESS: All services started!")
    print("\nAccess points:")
    print("  • API Documentation: http://localhost:8000/docs")
    print("  • Dashboard: http://localhost:8050")
    print("  • Health Check: http://localhost:8000/health")
    print("\n[STOP] Press Ctrl+C to stop all services")
    
    # Wait for services
    try:
        manager.wait_for_services()
    except KeyboardInterrupt:
        manager.stop_all()

def start_docker_services():
    """Start services using Docker Compose"""
    print("🐳 Starting services with Docker Compose...")
    
    project_root = Path(__file__).parent.parent
    
    try:
        # Start services
        subprocess.run(['docker-compose', 'up', '--build'], cwd=project_root)
    except KeyboardInterrupt:
        print("\n🛑 Stopping Docker services...")
        subprocess.run(['docker-compose', 'down'], cwd=project_root)
    except FileNotFoundError:
        print("[ERROR] Docker Compose not found. Please install Docker and Docker Compose")
        sys.exit(1)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Fresh Supply Chain Intelligence System")
    parser.add_argument("--docker", action="store_true", 
                       help="Start services using Docker Compose")
    parser.add_argument("--api-only", action="store_true", 
                       help="Start only the API server")
    parser.add_argument("--dashboard-only", action="store_true", 
                       help="Start only the dashboard")
    
    args = parser.parse_args()
    
    if args.docker:
        start_docker_services()
    elif args.api_only:
        print("Loading: Starting API server only...")
        subprocess.run(['python', 'api/main.py'])
    elif args.dashboard_only:
        print("Loading: Starting dashboard only...")
        subprocess.run(['python', 'dashboard/app.py'])
    else:
        start_services()

if __name__ == "__main__":
    main()
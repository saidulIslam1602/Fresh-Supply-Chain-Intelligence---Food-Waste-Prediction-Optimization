#!/usr/bin/env python3
"""
Test runner script for Fresh Supply Chain Intelligence System
Runs all tests with coverage reporting
"""

import sys
import os
import subprocess
import argparse

def run_tests(test_type="all", coverage=True, verbose=False):
    """Run tests with specified options"""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory
    cmd.append("tests/")
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
    
    # Add specific test type
    if test_type == "unit":
        cmd.extend(["-m", "not integration"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    
    # Add test discovery
    cmd.extend(["--tb=short", "--strict-markers"])
    
    print(f"Running tests: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run tests
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
        if coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

def run_linting():
    """Run code linting"""
    
    print("Running code linting...")
    print("=" * 40)
    
    # Run black (code formatting)
    print("Running black...")
    result = subprocess.run(["python", "-m", "black", ".", "--check"], 
                          cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if result.returncode != 0:
        print("❌ Code formatting issues found. Run 'black .' to fix.")
        return False
    
    # Run flake8 (style checking)
    print("Running flake8...")
    result = subprocess.run(["python", "-m", "flake8", ".", "--max-line-length=100"], 
                          cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if result.returncode != 0:
        print("❌ Style issues found.")
        return False
    
    # Run mypy (type checking)
    print("Running mypy...")
    result = subprocess.run(["python", "-m", "mypy", ".", "--ignore-missing-imports"], 
                          cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if result.returncode != 0:
        print("⚠️  Type checking issues found (non-critical).")
    
    print("✅ Linting completed!")
    return True

def run_security_scan():
    """Run security scanning"""
    
    print("Running security scan...")
    print("=" * 40)
    
    # Run bandit (security linting)
    try:
        result = subprocess.run(["python", "-m", "bandit", "-r", ".", "-f", "json"], 
                              cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if result.returncode != 0:
            print("⚠️  Security issues found. Check bandit output.")
        else:
            print("✅ No security issues found!")
    except FileNotFoundError:
        print("⚠️  Bandit not installed. Install with: pip install bandit")

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Run tests for Fresh Supply Chain Intelligence System")
    parser.add_argument("--type", choices=["all", "unit", "integration", "fast"], 
                       default="all", help="Type of tests to run")
    parser.add_argument("--no-coverage", action="store_true", 
                       help="Disable coverage reporting")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--lint", action="store_true", 
                       help="Run code linting")
    parser.add_argument("--security", action="store_true", 
                       help="Run security scan")
    parser.add_argument("--all", action="store_true", 
                       help="Run all checks (tests, linting, security)")
    
    args = parser.parse_args()
    
    if args.all:
        print("🚀 Running all checks...")
        print("=" * 60)
        
        # Run linting
        if not run_linting():
            print("❌ Linting failed!")
            sys.exit(1)
        
        # Run security scan
        run_security_scan()
        
        # Run tests
        run_tests(args.type, not args.no_coverage, args.verbose)
        
    elif args.lint:
        if not run_linting():
            sys.exit(1)
    
    elif args.security:
        run_security_scan()
    
    else:
        # Run tests
        run_tests(args.type, not args.no_coverage, args.verbose)

if __name__ == "__main__":
    main()
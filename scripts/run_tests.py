#!/usr/bin/env python3
"""
Enhanced Test Runner for Fresh Supply Chain Intelligence System
"""

import os
import sys
import argparse
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestRunner:
    """Enhanced test runner with comprehensive reporting"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
    def run_command(self, command, capture_output=True):
        """Run a command and return the result"""
        logger.info(f"Running: {' '.join(command)}")
        return subprocess.run(command, cwd=self.project_root, capture_output=capture_output, text=True)
    
    def setup_environment(self):
        """Setup test environment"""
        logger.info("Setting up test environment...")
        directories = ['test_reports', 'test_reports/coverage', 'test_reports/junit']
        for directory in directories:
            (self.project_root / directory).mkdir(parents=True, exist_ok=True)
    
    def run_unit_tests(self, verbose=False, coverage=True):
        """Run unit tests"""
        logger.info("Running unit tests...")
        command = [
            sys.executable, '-m', 'pytest', 'tests/unit/',
            '-v' if verbose else '-q',
            '--junitxml=test_reports/junit/unit_tests.xml'
        ]
        if coverage:
            command.extend(['--cov=api', '--cov=models', '--cov=data'])
        
        result = self.run_command(command, capture_output=False)
        self.test_results['unit'] = {'success': result.returncode == 0}
        return self.test_results['unit']
    
    def run_integration_tests(self, verbose=False):
        """Run integration tests"""
        logger.info("Running integration tests...")
        command = [
            sys.executable, '-m', 'pytest', 'tests/integration/',
            '-v' if verbose else '-q',
            '--junitxml=test_reports/junit/integration_tests.xml'
        ]
        result = self.run_command(command, capture_output=False)
        self.test_results['integration'] = {'success': result.returncode == 0}
        return self.test_results['integration']
    
    def run_all_tests(self, test_types=None, verbose=False, coverage=True):
        """Run all specified test types"""
        if test_types is None:
            test_types = ['unit', 'integration']
        
        self.start_time = time.time()
        self.setup_environment()
        
        if 'unit' in test_types:
            self.run_unit_tests(verbose=verbose, coverage=coverage)
        if 'integration' in test_types:
            self.run_integration_tests(verbose=verbose)
        
        self.end_time = time.time()
        
        # Print summary
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        for test_type, result in self.test_results.items():
            status = "[SUCCESS] PASSED" if result['success'] else "[ERROR] FAILED"
            print(f"{test_type.upper()}: {status}")
        print("="*50)
        
        return self.test_results

def main():
    parser = argparse.ArgumentParser(description='Test Runner for Fresh Supply Chain Intelligence')
    parser.add_argument('--types', nargs='+', choices=['unit', 'integration'], 
                       default=['unit', 'integration'], help='Test types to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--no-coverage', action='store_true', help='Disable coverage')
    
    args = parser.parse_args()
    runner = TestRunner()
    
    results = runner.run_all_tests(
        test_types=args.types,
        verbose=args.verbose,
        coverage=not args.no_coverage
    )
    
    # Exit with error if any tests failed
    all_success = all(result['success'] for result in results.values())
    sys.exit(0 if all_success else 1)

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Main Demo Script
===============

Complete demo showing the entire pipeline.
"""

import os
import sys
import subprocess
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd, description, check=True):
    """
    Run a shell command with logging.
    
    Parameters:
    -----------
    cmd : str or list
        Command to run
    description : str
        Description for logging
    check : bool
        Whether to check return code
    """
    logger.info(f"{description}...")
    
    try:
        if isinstance(cmd, str):
            result = subprocess.run(cmd, shell=True, check=check, 
                                  capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, check=check, 
                                  capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print(result.stderr)
        
        logger.info(f"✓ {description} completed successfully")
        return result
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        raise


def check_dependencies():
    """Check if required packages are installed."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'numpy', 'opencv-python', 'scikit-learn', 'scikit-image',
        'matplotlib', 'seaborn', 'pandas', 'torchvision', 'pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {missing_packages}")
        logger.info("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("✓ All dependencies are available")
    return True


def setup_environment():
    """Setup the project environment."""
    logger.info("Setting up environment...")
    
    # Create necessary directories
    directories = ['data', 'cache', 'results', 'results/visualizations']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")
    
    # Set Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, 'src')
    
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    logger.info("✓ Environment setup completed")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Run complete descriptor evaluation demo')
    parser.add_argument('--download-only', action='store_true',
                       help='Only download data (skip training and evaluation)')
    parser.add_argument('--train-only', action='store_true',
                       help='Only run training (skip evaluation)')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation (skip training)')
    parser.add_argument('--max-samples', type=int, default=5000,
                       help='Maximum training samples (default: 5000)')
    parser.add_argument('--global-only', action='store_true',
                       help='Train/evaluate only global descriptors')
    parser.add_argument('--local-only', action='store_true',
                       help='Train/evaluate only local descriptors')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with minimal data')
    
    args = parser.parse_args()
    
    try:
        print("=" * 60)
        print("UNSUPERVISED DESCRIPTORS HACKATHON DEMO")
        print("=" * 60)
        
        # Setup environment
        setup_environment()
        
        # Check dependencies
        if not check_dependencies():
            logger.error("Please install missing dependencies first")
            return 1
        
        # Adjust parameters for quick test
        if args.quick_test:
            args.max_samples = 1000
            logger.info("Quick test mode: using reduced dataset")
        
        # Step 1: Download data
        if not args.train_only and not args.eval_only:
            print("\n" + "=" * 40)
            print("STEP 1: DOWNLOADING STL-10 DATASET")
            print("=" * 40)
            
            download_cmd = ["python", "scripts/download_data.py"]
            run_command(download_cmd, "Downloading STL-10 dataset")
            
            if args.download_only:
                print("\n✓ Data download completed!")
                return 0
        
        # Step 2: Train descriptors
        if not args.eval_only:
            print("\n" + "=" * 40)
            print("STEP 2: TRAINING DESCRIPTORS")
            print("=" * 40)
            
            train_cmd = ["python", "scripts/train_descriptors.py", 
                        "--max-samples", str(args.max_samples)]
            
            if args.global_only:
                train_cmd.append("--global-only")
            elif args.local_only:
                train_cmd.append("--local-only")
            
            run_command(train_cmd, "Training descriptor methods")
            
            if args.train_only:
                print("\n✓ Training completed!")
                return 0
        
        # Step 3: Evaluate descriptors
        if not args.train_only:
            print("\n" + "=" * 40)
            print("STEP 3: EVALUATING DESCRIPTORS")
            print("=" * 40)
            
            eval_cmd = ["python", "scripts/evaluate_descriptors.py"]
            
            if args.quick_test:
                eval_cmd.extend(["--no-robustness"])
            
            run_command(eval_cmd, "Evaluating descriptor performance")
        
        # Success message
        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 60)
        print("\nResults available in:")
        print("  📊 results/evaluation_results.json    - Detailed results")
        print("  📝 results/evaluation_report.txt      - Summary report")
        print("  📈 results/visualizations/            - Performance plots")
        print("  💾 cache/                             - Trained models")
        
        print("\nKey findings:")
        
        # Try to show quick summary if results exist
        try:
            import json
            with open('results/evaluation_results.json', 'r') as f:
                results = json.load(f)
            
            descriptors = results.get('descriptors', {})
            if descriptors:
                print("  🏆 Top performing descriptors:")
                
                # Find best performers
                best_results = []
                for desc_name, desc_result in descriptors.items():
                    if isinstance(desc_result, dict):
                        if any(key in ['bow', 'vlad', 'fisher'] for key in desc_result.keys()):
                            # Local descriptor
                            for enc_name, enc_result in desc_result.items():
                                if enc_result.get('status') == 'success':
                                    classifiers = enc_result.get('classifiers', {})
                                    for clf_result in classifiers.values():
                                        if clf_result.get('status') == 'success':
                                            acc = clf_result.get('test_metrics', {}).get('accuracy', 0)
                                            if acc > 0:
                                                best_results.append((f"{desc_name}+{enc_name}", acc))
                        else:
                            # Global descriptor
                            if desc_result.get('status') == 'success':
                                classifiers = desc_result.get('classifiers', {})
                                for clf_result in classifiers.values():
                                    if clf_result.get('status') == 'success':
                                        acc = clf_result.get('test_metrics', {}).get('accuracy', 0)
                                        if acc > 0:
                                            best_results.append((desc_name, acc))
                
                # Sort and show top 3
                best_results.sort(key=lambda x: x[1], reverse=True)
                for i, (name, acc) in enumerate(best_results[:3]):
                    print(f"     {i+1}. {name}: {acc:.3f} accuracy")
        
        except Exception:
            pass  # Don't fail if we can't read results
        
        print("\n🚀 Ready for hackathon presentation!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
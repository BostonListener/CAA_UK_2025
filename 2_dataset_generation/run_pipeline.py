#!/usr/bin/env python3
"""
Master script to run the complete data generation pipeline.
Executes all scripts in the correct order with dependency awareness.
"""

import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime, timedelta


class PipelineRunner:
    def __init__(self):
        self.scripts_dir = Path(__file__).parent / 'scripts'
        self.results = []
        self.start_time = None
        self.total_duration = None
        
    def print_header(self):
        print("=" * 80)
        print("ARCHAEOLOGICAL SITE DATA GENERATION - FULL PIPELINE")
        print("=" * 80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("Execution order:")
        print("  1. Generate Positives (with rotation)")
        print("  2. Generate Integrated Negatives (with rotation)")
        print("  3. Generate Landcover Negatives")
        print("  4. Generate Unlabeled Data")
        print("  5. Generate Radiometric Augmentation (depends on 1 & 2)")
        print()
        print("⚠️  WARNING: This pipeline may take several hours to complete!")
        print("⚠️  Ensure stable internet connection for Google Earth Engine API")
        print("=" * 80)
        print()
        
    def run_script(self, script_name, stage_number, total_stages):
        """Run a single Python script and track results."""
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            print(f"❌ ERROR: Script not found: {script_path}")
            return False
        
        print(f"\n{'='*80}")
        print(f"STAGE {stage_number}/{total_stages}: {script_name}")
        print(f"{'='*80}")
        print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        stage_start = time.time()
        
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                check=True,
                capture_output=False,  # Show output in real-time
                text=True
            )
            
            stage_duration = time.time() - stage_start
            
            print()
            print(f"✅ {script_name} completed successfully")
            print(f"   Duration: {self._format_duration(stage_duration)}")
            
            self.results.append({
                'script': script_name,
                'status': 'SUCCESS',
                'duration': stage_duration
            })
            
            return True
            
        except subprocess.CalledProcessError as e:
            stage_duration = time.time() - stage_start
            
            print()
            print(f"❌ {script_name} FAILED")
            print(f"   Duration: {self._format_duration(stage_duration)}")
            print(f"   Exit code: {e.returncode}")
            
            self.results.append({
                'script': script_name,
                'status': 'FAILED',
                'duration': stage_duration,
                'error': str(e)
            })
            
            return False
            
        except Exception as e:
            stage_duration = time.time() - stage_start
            
            print()
            print(f"❌ {script_name} FAILED with exception")
            print(f"   Duration: {self._format_duration(stage_duration)}")
            print(f"   Error: {str(e)}")
            
            self.results.append({
                'script': script_name,
                'status': 'FAILED',
                'duration': stage_duration,
                'error': str(e)
            })
            
            return False
    
    def _format_duration(self, seconds):
        """Format duration in human-readable format."""
        duration = timedelta(seconds=int(seconds))
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        secs = duration.seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def print_summary(self):
        """Print final summary of pipeline execution."""
        print("\n")
        print("=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print()
        
        success_count = sum(1 for r in self.results if r['status'] == 'SUCCESS')
        failed_count = sum(1 for r in self.results if r['status'] == 'FAILED')
        
        print(f"Total stages: {len(self.results)}")
        print(f"✅ Successful: {success_count}")
        print(f"❌ Failed: {failed_count}")
        print()
        
        print("Stage Results:")
        print("-" * 80)
        for i, result in enumerate(self.results, 1):
            status_symbol = "✅" if result['status'] == 'SUCCESS' else "❌"
            duration_str = self._format_duration(result['duration'])
            print(f"  {i}. {status_symbol} {result['script']:<50} {duration_str:>15}")
        print("-" * 80)
        print()
        
        if self.total_duration:
            print(f"Total pipeline duration: {self._format_duration(self.total_duration)}")
        
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        if failed_count > 0:
            print("⚠️  Some stages failed. Check the output above for details.")
            print()
        else:
            print("🎉 All stages completed successfully!")
            print()
        
        print("=" * 80)
    
    def run_pipeline(self):
        """Run the complete pipeline."""
        self.start_time = time.time()
        self.print_header()
        
        # Define pipeline stages
        stages = [
            'generate_positives.py',
            'generate_integrated_negatives.py',
            'generate_landcover_negatives.py',
            'generate_unlabeled.py',
            'generate_radiometric_augmentation.py',  # Depends on stages 1 & 2, runs last
        ]
        
        total_stages = len(stages)
        
        # Check if all scripts exist before starting
        print("Pre-flight checks:")
        all_scripts_exist = True
        for script in stages:
            script_path = self.scripts_dir / script
            if script_path.exists():
                print(f"  ✅ {script}")
            else:
                print(f"  ❌ {script} NOT FOUND")
                all_scripts_exist = False
        
        if not all_scripts_exist:
            print()
            print("❌ ERROR: Not all scripts found. Please check the scripts/ directory.")
            return False
        
        print()
        input("Press Enter to start the pipeline (or Ctrl+C to cancel)...")
        print()
        
        # Run each stage
        for i, script in enumerate(stages, 1):
            success = self.run_script(script, i, total_stages)
            
            # Critical stages: positives and integrated negatives
            # If these fail, radiometric augmentation cannot proceed
            if not success:
                if script in ['generate_positives.py', 'generate_integrated_negatives.py']:
                    print()
                    print(f"⚠️  CRITICAL FAILURE: {script} failed")
                    print("   Radiometric augmentation requires this stage to complete.")
                    print()
                    
                    user_input = input("Continue with remaining stages? (y/N): ").strip().lower()
                    if user_input != 'y':
                        print("Pipeline aborted by user.")
                        break
                else:
                    # Non-critical stages can continue
                    print()
                    print("⚠️  Stage failed but pipeline will continue...")
                    time.sleep(2)
        
        self.total_duration = time.time() - self.start_time
        self.print_summary()
        
        return all(r['status'] == 'SUCCESS' for r in self.results)


def main():
    try:
        runner = PipelineRunner()
        success = runner.run_pipeline()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n")
        print("=" * 80)
        print("Pipeline interrupted by user (Ctrl+C)")
        print("=" * 80)
        print()
        print("✅ Already completed stages remain intact.")
        print("💡 You can re-run this script to continue - completed stages will be skipped.")
        print()
        sys.exit(130)
    
    except Exception as e:
        print("\n\n")
        print("=" * 80)
        print("UNEXPECTED ERROR")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
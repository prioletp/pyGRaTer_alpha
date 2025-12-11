import argparse
import sys
from pathlib import Path
from pyGrater.config.paths import DataPathConfig

def main():
    """Command-line interface for pyGrater data setup"""
    parser = argparse.ArgumentParser(
        description='Configure pyGrater data folder path',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pygrater-setup --data-path /path/to/data
  pygrater-setup --data-path ~/Downloads/pyGrater_data --check
  pygrater-setup --show
        """
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to the pyGrater data folder'
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Verify the data folder contains required files'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show current data path configuration'
    )
    
    args = parser.parse_args()
    
    if args.show:
        try:
            current_path = DataPathConfig.get_data_path()
            print(f"✓ Data path configured: {current_path}")
        except FileNotFoundError:
            print(f"✗ Data path not configured")
            print("\nRun: pygrater-setup --data-path /path/to/data")
        return
    
    if args.data_path:
        try:
            path = Path(args.data_path).expanduser().resolve()
            
            if args.check:
                required_dirs = ['optical_properties', 'efficiencies', 'star_data', 'temperatures']
                missing = [d for d in required_dirs if not (path / d).exists()]
                if missing:
                    print(f"⚠ Warning: Missing directories: {', '.join(missing)}")
                    response = input("Continue anyway? (y/n): ")
                    if response.lower() != 'y':
                        sys.exit(1)
            
            DataPathConfig.set_data_path(path, persistent=True)
            print(f"✓ Data path successfully configured!")
            print(f"  Location: {path}")
            print(f"  Config saved to: {DataPathConfig.CONFIG_FILE}")
            
        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
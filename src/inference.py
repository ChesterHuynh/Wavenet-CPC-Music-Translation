"""
Sample input audio clips and generate translated samples with trained model
"""
from src.data.data_samples import sample
from src.data.run_on_files import generate

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()

    # For sampling
    parser.add_argument('--data', type=Path, nargs='*',
                        help='Path to data dir')
    parser.add_argument('--data-from-args', type=Path,
                        help='Path to args.pth')
    parser.add_argument('--output-sampled', '-os', type=Path,
                        help='Output path')
    parser.add_argument('-n', type=int,
                        help='Num samples to make')
    parser.add_argument('--seq-len', type=int, default=80000)

    # For generation
    parser.add_argument('--model-name', type=str, required=True, choices=['umt', 'umtcpc'],
                        help='Type of architecture (UMT or UMTCPC)')
    parser.add_argument('--files', type=Path, nargs='+', required=False,
                        help='Top level directories of input music files')
    parser.add_argument('-og', '--output-generated', type=Path,
                        help='Output directory for output files')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Checkpoint path')
    parser.add_argument('--decoders', type=int, nargs='*', default=[],
                        help='Only output for the following decoder ID')
    parser.add_argument('--rate', type=int, default=16000,
                        help='Wav sample rate in samples/second')
    parser.add_argument('--batch-size', type=int, default=6,
                        help='Batch size during inference')
    parser.add_argument('--sample-len', type=int,
                        help='If specified, cuts sample lengths')
    parser.add_argument('--split-size', type=int, default=20,
                        help='Size of splits')
    parser.add_argument('--output-next-to-orig', action='store_true')
    parser.add_argument('--skip-filter', action='store_true')
    parser.add_argument('--py', action='store_true', help='Use python generator')

    args = parser.parse_args()

    
    # Sample data
    if args.sample:
        print("Sampling")
        sample(args)

    # Generate translated samples
    print("Generating")
    generate(args)


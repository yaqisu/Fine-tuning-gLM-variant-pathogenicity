#!/usr/bin/env python3
"""
Extract reference and alternative sequences for variants with flanking context.
"""

import argparse
from pyfaidx import Fasta


def extract_variant_sequences(bed_file, fasta_file, flank_length, output_file):
    """
    Extract reference and alternative sequences for each variant.
    
    Args:
        bed_file: Path to BED file with variants (1-based coordinates)
        fasta_file: Path to reference genome FASTA
        flank_length: Length of flanking sequence on each side
        output_file: Path to output text file
    """
    # Load reference genome
    genome = Fasta(fasta_file)
    
    with open(bed_file, 'r') as bed, open(output_file, 'w') as out:
 
        out.write("variant_id\tchromosome\tposition\tref_allele\talt_allele\tupstream_flank\tdownstream_flank\tref_sequence\talt_sequence\n")
        
        mismatch_count = 0
        total_variants = 0
        
        for line_num, line in enumerate(bed, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            fields = line.split('\t')
            if len(fields) < 6:
                print(f"Warning: Line {line_num} has fewer than 6 fields, skipping")
                continue
            
            total_variants += 1
            
            chrom = fields[0]
            start = int(fields[1])  # 1-based BED coordinate
            end = int(fields[2])
            variant_id = fields[3]
            ref_allele = fields[4].upper()
            alt_allele = fields[5].upper()
            
            # Convert 1-based to 0-based for FASTA access
            var_pos = start - 1 
            
            # Strip 'chr' prefix for FASTA lookup
            chrom_fasta = chrom.replace('chr', '', 1) if chrom.startswith('chr') else chrom
            
            # Check if chromosome exists in genome
            if chrom_fasta not in genome:
                print(f"Warning: Chromosome {chrom_fasta} not found in genome, skipping variant {variant_id}")
                continue
            
            # Get chromosome length
            chrom_length = len(genome[chrom_fasta])
            
            # Calculate flanking region boundaries in 0-based coordinates
            flank_start = max(0, var_pos - flank_length)
            flank_end = min(chrom_length, var_pos + flank_length + 1)
            
            # Extract sequence 
            sequence = str(genome[chrom_fasta][flank_start:flank_end]).upper()
            
            # Find the position of the variant in the extracted sequence
            var_offset = var_pos - flank_start
            
            # Verify that the reference allele matches
            if var_offset < len(sequence):
                extracted_ref = sequence[var_offset]
                if extracted_ref != ref_allele:
                    mismatch_count += 1
                    print(f"Warning: Reference mismatch for {variant_id} at {chrom}:{start} (1-based)")
                    print(f"  Expected: {ref_allele}, Found: {extracted_ref}")
            
            # Create alternative sequence by replacing the reference allele
            alt_sequence = sequence[:var_offset] + alt_allele + sequence[var_offset + 1:]
            
            # Calculate actual flanking lengths
            upstream_flank = var_offset
            downstream_flank = len(sequence) - var_offset - 1
            
            # Remove 'chr' prefix from chromosome name for output
            chrom_output = chrom.replace('chr', '', 1) if chrom.startswith('chr') else chrom
            
            # Write tab-separated output
            out.write(f"{variant_id}\t{chrom_output}\t{start}\t{ref_allele}\t{alt_allele}\t{upstream_flank}\t{downstream_flank}\t{sequence}\t{alt_sequence}\n")
        
        # Write summary
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total variants processed: {total_variants}")
        print(f"Reference allele mismatches: {mismatch_count}")
        print(f"{'='*60}\n")
    
    print(f"Extraction complete. Output written to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract reference and alternative sequences for variants with flanking context',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python extract_variant_sequences.py -b variants.bed -f genome.fa -l 50 -o sequences.txt
  
This will extract up to 50 bp flanking sequence on each side of each variant.
BED file is assumed to be 1-based coordinates.
        """
    )
    
    parser.add_argument('-b', '--bed', required=True,
                        help='Input BED file with variants (1-based coordinates)')
    parser.add_argument('-f', '--fasta', required=True,
                        help='Reference genome FASTA file')
    parser.add_argument('-l', '--flank-length', type=int, required=True,
                        help='Length of flanking sequence on each side of the variant')
    parser.add_argument('-o', '--output', required=True,
                        help='Output text file')
    
    args = parser.parse_args()
    
    extract_variant_sequences(args.bed, args.fasta, args.flank_length, args.output)


if __name__ == '__main__':
    main()

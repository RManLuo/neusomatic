#!/bin/bash
set -e

test_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
neusomatic_dir="$( dirname ${test_dir} )"

cd ${test_dir}
mkdir -p example
cd example
if [ ! -f Homo_sapiens.GRCh37.75.dna.chromosome.22.fa ]
then
        if [ ! -f Homo_sapiens.GRCh37.75.dna.chromosome.22.fa.gz ]
        then
                wget ftp://ftp.ensembl.org/pub/release-75//fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.75.dna.chromosome.22.fa.gz
        fi
        gunzip -f Homo_sapiens.GRCh37.75.dna.chromosome.22.fa.gz
fi
if [ ! -f Homo_sapiens.GRCh37.75.dna.chromosome.22.fa.fai ]
then
        samtools faidx Homo_sapiens.GRCh37.75.dna.chromosome.22.fa
fi

#Ensemble NeuSomatic train       
python ${neusomatic_dir}/neusomatic/python/preprocess.py \
        --mode train \
        --reference Homo_sapiens.GRCh37.75.dna.chromosome.22.fa \
        --region_bed ${test_dir}/region.bed \
        --tumor_bam ${test_dir}/tumor.bam \
        --normal_bam ${test_dir}/normal.bam \
        --work work_train \
        --min_mapq 10 \
        --num_threads 1 \
        --truth_vcf ${test_dir}/train.vcf \
        --scan_alignments_binary ${neusomatic_dir}/neusomatic/bin/scan_alignments

CUDA_VISIBLE_DEVICES= python ${neusomatic_dir}/neusomatic/python/train.py \
                --candidates_tsv work_train/dataset/*/candidates*.tsv \
                --out work_train \
                --num_threads 1 \
                --batch_size 100


cd ..

file1=${test_dir}/example/work_standalone/NeuSomatic_standalone.vcf
file2=${test_dir}/NeuSomatic_standalone.vcf

cmp --silent $file1 $file2 && echo "### NeuSomatic stand-alone: SUCCESS! ###" \

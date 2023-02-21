# slow5-dorado

This is our fork of [Dorado](https://github.com/nanoporetech/dorado) that supports [S/BLOW5](https://www.nature.com/articles/s41587-021-01147-4).
Dorado is a high-performance, easy-to-use, open source basecaller for Oxford Nanopore reads. For typical simplex basecalling and mod calling with S/BLOW5, we recommend using [buttery-eel](https://github.com/Psy-Fer/buttery-eel) wrapper with Guppy. At the moment, this fork is primarily for duplex calling with S/BLOW5.

## Features

* One executable with sensible defaults, automatic hardware detection and configuration.
* Runs on Apple silicon (M1/2 family) and Nvidia GPUs including multi-GPU with linear scaling.
* Modified basecalling (Remora models).
* Duplex basecalling.
* [S/BLOW5](https://www.nature.com/articles/s41587-021-01147-4) support for highest basecalling performance.
* Based on libtorch, the C++ API for pytorch.
* Multiple custom optimisations in CUDA and Metal for maximising inference performance.

If you encounter any problems building or running sow5-dorado please [report an issue](https://github.com/hiruna72/slow5-dorado/issues).

## Installation

Binaries are provided for Linux x64 under [Relases](https://github.com/hiruna72/slow5-dorado/releases/). 

```
wget https://github.com/hiruna72/slow5-dorado/releases/download/v0.2.1/slow5-dorado-v0.2.1-x86_64-linux.tar.gz -O slow5-dorado.tar.gz
tar xf slow5-dorado.tar.gz
cd slow5-dorado/bin
./slow5-dorado --version
```

## Running

To run slow5-dorado, download a model and point it to S/BLOW5 files.

```

$ slow5-dorado download --model dna_r10.4.1_e8.2_400bps_hac@v4.0.0
$ slow5-dorado basecaller dna_r10.4.1_e8.2_400bps_hac@v4.0.0 BLOW5s/ > calls.sam # blow5 directory
$ slow5-dorado basecaller dna_r10.4.1_e8.2_400bps_hac@v4.0.0 merged.blow5 >calls.sam # a single BLOW5 file
```

To call modifications simply add `--modified-bases`.

```
$ slow5-dorado basecaller dna_r10.4.1_e8.2_400bps_hac@v4.0.0 BLOW5s/ --modified-bases 5mCG_5hmCG > calls.sam
```

For unaligned BAM output, dorado output can be piped to BAM using samtoools:

```
$ slow5-dorado basecaller dna_r10.4.1_e8.2_400bps_hac@v4.0.0 BLOW5s/ | samtools view -Sh > calls.bam
```

Stereo Duplex Calling:

```
$ slow5-dorado duplex dna_r10.4.1_e8.2_400bps_sup@v4.0.0 BLOW5s/ --pairs pairs.txt > duplex.sam
```

## Platforms

slow5-dorado has been tested on the following systems:

| Platform | GPU/CPU                      |
| -------- | ---------------------------- |
| Linux    | (G)V100, A100, 3090, 3070    |

Systems not listed above but which have Nvidia GPUs with >=8GB VRAM and architecture from Volta onwards have not been widely tested but are expected to work. If you encounter problems with running on your system please [report an issue](https://github.com/nanoporetech/dorado/issues)


### Licence and Copyright
(c) 2022 Oxford Nanopore Technologies Ltd.

Dorado is distributed under the terms of the Oxford Nanopore
Technologies, Ltd.  Public License, v. 1.0.  If a copy of the License
was not distributed with this file, You can obtain one at
http://nanoporetech.com

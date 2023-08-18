# GPred: Prediction of metal ion-ligand binding using Graph Neural Networks

The interaction between the metal ions and the proteins are essential for various biological functions like maintaining the protein structure, signal transport, etc. Protein-ion interaction is useful for understanding the biological function of proteins and for designing novel drug. While several computational approaches have been proposed, this remains a difficult problem due to the small size and high versatility of metal ions. In this study, we propose GPred, which is a structure-based method that transforms the three-dimensional structure of the protein to point cloud and uses the Graph Neural Network (GNN) to learn the local structural properties of each amino acid residue under specific ligand-binding supervision.
## Installation Guide
### Install PCBEP from GPred
```shell
git clone  https://github.com/wn1225/GPred
```
### Install dependency packages
1. Install `PyTorch` following the [official guide](https://pytorch.org/get-started/locally/).
1. Install `torch-geometric` following the [official guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
1. Install other dependencies:
```
pip install -r requirements.txt
```
The above steps take 20-25 mins to install all dependencies.

## Evaluate our model
First unzip the file:'Data/feature_pssm_types_ZN.zip'

If you want to evaluate the results of our data.
```shell
python GPred_test.py
```
If you want to train our 5-fold cross-validation model.
```shell
python GPred_5fold_train.py
```
## Test your own data 
If you want to use our model to test your own data. Please refer to the following steps:
### 1. Get the epitope label of the data.   
Use 0 1 to encode epitope labels and organize them into fasta format, for example: 
```
>6WIR-C
00000000000000000000000000000000010000000000111100000000000000000000000000000000000111111
>5W2B-A
0000001001100111111000100100000001011111000000000000000000000000000000000000000000000100000011001
>5TFW-O
0000000000000000000000000000000000000000000000000000010100000010000010000000000000000000000000000000000000011110011011001000000000000000000000000000000000
```

### 2. Get the surface label of the data.  
Use 0 1 to encode surface labels and organize them into fasta format, for example:
```
>6WIR-C
011111111111111111111111111111111110011111111111111111111111111111111111111111111111111111
>5W2B-A
01111111111111111111110111111111111111111111111111111111111111111111111111111111011111111111011111
>5TFW-O
01111111111111111111101110011011110011101110111111111011111110010111111111100000101111101111111111101000010111111111011111111111110110111111101110111111111
```

### 3. Obtain the PSSM matrices of the antigens to be tested.  
```
Last position-specific scoring matrix computed, weighted observed percentages rounded down, information per position, and relative weight of gapless real matches to pseudocounts
            A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
    1 S     2  -1   0  -1  -1   0  -1   0  -1  -2  -2  -1  -1  -3  -1   4   2  -3  -2  -1   26   0   0   0   0   0   0   0   0   0   0   0   0   0   0  65   9   0   0   0  0.40 0.03
    2 D    -1  -2   1   5  -3   0   2  -1  -1  -3  -4  -1  -3  -4  -2   2   0  -4  -3  -3    0   0   0  68   0   0   8   0   0   0   0   0   0   0   0  23   0   0   0   0  0.71 0.04
    3 Y    -1  -2  -2  -3  -2  -2  -2  -3   0   0   0  -2   0   3  -3  -1   2   0   5   1    0   0   0   0   0   0   0   0   0   0   0   0   0  16   0   0  22   0  39  23  0.36 0.03  
```

### 4. Get the feature file for your own data
```
python Pretreatment/generate.py -l ../Data/label.txt -f ../Data/ -s ../Data/surface.txt -p ../Data/data -m ../Data/PSSM -o ../Data/data_feature_surface.txt -r [C,H,E,D] -fr ../Data/feature_pssm_types.txt
```
- `-l`or`--label` file path for epitope label.  [default:'../Data/label.txt']
- `-f`or`--fasta` file path for fasta.  [default:'../Data/']
- `-s`or`--surface` file path for surface label. [default:'../Data/surface.txt']
- `-p`or`--pdb` fold path for pdb files to be tested.  [default:'../Data/data']
- `-m`or`--pssm` fold path for PSSM files of antigens. [default:'../Data/PSSM']
- `-r`or`--residues` Candidate residues for each ion.  [default:'[C,H,E,D]']
- `-o`or`--output` output file path. [default:'../Data/data_feature_surface.txt']
- `-fr`or`--output1` final output.  [default:'../Data/feature_pssm_types.txt']

### 5. Get test result
```
python GPred_test.py -i Data/feature_pssm_types.txt -c checkpoint.pt -o result/result.txt
```
- `-i`or`--input` file path for fature path. [default:'Data/feature_pssm_types.txt']
- `-c`or`--checkpoint` file path for pre-trained model.. [default:'checkpoint.pt']
- `-o`or`--ouptup` output file path. [default:'result/result.txt']

# GPred: Prediction of metal ion-ligand binding sites using geometry-aware graph neural networks

The interaction between the metal ions and the proteins are essential for various biological functions like maintaining the protein structure, signal transport, etc. Protein-ion interaction is useful for understanding the biological function of proteins and for designing novel drug. While several computational approaches have been proposed, this remains a difficult problem due to the small size and high versatility of metal ions. In this study, we propose GPred, which is a structure-based method that transforms the three-dimensional structure of the protein to point cloud and uses the Graph Neural Network (GNN) to learn the local structural properties of each amino acid residue under specific ligand-binding supervision.
## Installation Guide
### Install GPred from GitHub
```shell
[git clone  https://github.com/wn1225/GPred](https://github.com/wn1225/GPred.git)
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
If you want to evaluate the results of our data.
```shell
python GPred_5fold_train.py
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

### 5. Get model and test result
```
python GPred_train.py -input Data/feature_pssm_types.txt -tnum 442 -cpath Data/pt/ -spath Data/result/ -mpath Data/
```
- `-input`or`--input` file path for fature path. [default:'Data/feature_pssm_types.txt']
- `-cpath`or`--cpath` file path for pre-trained model. [default:'Data/pt/']
- `-spath`or`--spathp` output file path. [default:'Data/result/']
- `-tnum`or`--tnum` Split Test Sets. [default:'442']
- `-mpath`or`--mpath` Documentation of the results of the indicators of the test set. [default:'Data/']

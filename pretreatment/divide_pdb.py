import argparse
from Bio.PDB import Select, PDBIO
from Bio.PDB.PDBParser import PDBParser
import os

class ChainSelect(Select):
    def __init__(self, chain):
        self.chain = chain

    def accept_chain(self, chain):
        if chain.get_id() == self.chain:
            return 1
        else:
            return 0


def main():    
    parser = argparse.ArgumentParser(description='supply the old and new directory to update the downloaded pdbs for a specific ion i.e. ZN, CA, CO3')
    parser.add_argument('-input', dest='input', type=str, help='Specify the location of file that contain the pdb chains i.e. data_list.txt for the specific ion of interest', required=True)
    parser.add_argument('-opath', dest='opath', type=str, help='Specify the location of the pdb files to be stored that needs to be updated i.e. update_pdb', required=True)
    parser.add_argument('-npath', dest='npath', type=str, help='Specify the location for the pdb files to be stored that has been updated', required=True)

    args = parser.parse_args()

    input = args.input
    opath = args.opath
    npath = args.npath
    os.makedirs(npath, exist_ok=True)
    
    with open(input,'r') as fp:
        # lines = fp.read().split('\n')
        for line in fp:
            line = line.strip().split('_')
            name = line[0]
            chain = line[1]
            p = PDBParser(PERMISSIVE=1)
            pdb_file = opath +'{}.pdb'.format(name)
            structure = p.get_structure(pdb_file, pdb_file)

            # for chain in chains:
            pdb_chain_file = npath +'{}_{}.pdb'.format(name,chain)
            io_w_no_h = PDBIO()
            io_w_no_h.set_structure(structure)
            io_w_no_h.save('{}'.format(pdb_chain_file), ChainSelect(chain))
            print(pdb_chain_file)
    print("Done running divide_pdb.py")
if __name__ == '__main__':
    main()
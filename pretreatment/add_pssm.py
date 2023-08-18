import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='supply the old and new directory to update the downloaded pdbs for a specific ion i.e. ZN, CA, CO3')
    parser.add_argument('-ipath', dest='ipath', type=str, help='Specify the path to  the specific ion of interest', required=True)
    parser.add_argument('-label', dest='label', type=str, help='Specify the directory for the ion under consideration', required=True)
    parser.add_argument('-feature', dest='feature', type=str, help='Specify the path to the features', required=True)
    
    
    args = parser.parse_args()

    ipath = args.ipath
    label = args.label
    feature = args.feature
    
    pssm_path = ipath+'pssm/'
    os.makedirs(pssm_path, exist_ok=True)

    num = 0
    with open(label+'label_test.txt', 'r') as la:
        with open(feature+'feature_pssm_test.txt', 'w') as nfe:
            with open(feature+'feature_no_pssm_test.txt', 'r') as fe:
                num_acid = fe.readline().strip()
                nfe.write(num_acid + '\n')
                for i in range(int(num_acid)):
                    name_pssm = la.readline().strip().split()[0]
                    la.readline()
                    name_lines = name_pssm[1:].split('_')
                    # name = name_lines[0] + '-' + name_lines[1]
                    name = name_lines[0]

                    chain = name_lines[1]

                    with open(pssm_path + str(name) + '-' + str(chain) + '.pssm', 'r') as ps:
                        ps.readline()
                        ps.readline()
                        ps.readline()
                        num_atom = fe.readline().strip()
                        nfe.write(num_atom + '\n')
                        line = ps.readline().strip().split()
                        for i_a in range(int(num_atom.split()[0])):
                            # print(line[0])
                            try:
                                seq_ac_pssm = int(line[0]) - 1
                            except:
                                print(name)
                            # print(name)

                            feature_a = fe.readline().strip()
                            nfe.write(feature_a + '\t')
                            seq_ac = feature_a.split()[0]
                            if seq_ac_pssm != int(seq_ac):
                                line = ps.readline().strip().split()
                                seq_ac_pssm += 1
                            pssm = line[2:22]
                            for num_p in range(pssm.__len__()):
                                nfe.write(pssm[num_p] + '\t')
                            nfe.write('\n')
                            
    print("Done adding PSSM!")   
        
if __name__ == "__main__":
    main() 
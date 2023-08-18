import argparse


def main():
    parser = argparse.ArgumentParser(description='supply the old and new directory to update the downloaded pdbs for a specific ion i.e. ZN, CA, CO3')
    parser.add_argument('-fpath', dest='fpath', type=str,help='Specify the path to the FASTA total file',required=True)
    parser.add_argument('-dpath', dest='dpath', type=str, help='Specify the path to data_list',required=True)

    args = parser.parse_args()

    fpath = args.fpath
    dpath = args.dpath

    r = []
    with open(fpath,'r')as f:
        line = f.readlines()
    for i in range(len(line)):
        flag = 0
        row = line[i].strip().split()
        if row[0].find('>')!=-1:
            seq = line[i+1].strip().split()[0]
            for j in range(len(seq)):
                if seq[j] == 'X':
                    flag = 1
                    break
            if flag == 0:
                # fw.write(row[0]+'\n')
                # fw.write(seq + '\n')
                r.append(row[0])
                r.append(seq)
    fw = open(fpath, 'w')
    fp = open(dpath,'w')
    for i in range(len(r)):
        fw.write(r[i] + '\n')
        if r[i].find('>') != -1:
            fp.write(r[i][1:5] + '_' + r[i][6] + '\n')
        # fw.write(r[i] + '\n')

if __name__ == "__main__":
    main()


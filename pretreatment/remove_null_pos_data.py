def remove():
    fw = open('./data/FE2/data_list.txt','w')
    # fq = open('./data/CD/resolution.txt','w')
    with open('./data/FE2/test_data_list.txt','r')as f:
        for line in f:
            row = line.strip().split()
            name_chain = row[0]

            r = line.strip().split('_')
            pdb = r[0]
            chain = r[1]

            flag = 0
            with open('./data/CD/positive_CD_test.txt','r')as fp:   # 删除没有正样本的数据
                for li in fp:
                    r = li.strip().split()
                    if r[0] == name_chain:
                        # flag = 1
                        # break
                        fw.write(name_chain + '\n')
                        break

            # if flag == 1:
            #     with open('./data/CA/pdb_update/' + str(pdb) + '.pdb', 'r') as ff:     # 筛选分辨率
            #         for lii in ff:
            #             r = lii.strip().split()
            #             if r[0] == 'REMARK' and len(r) == 5:
            #                 if r[1].isdigit() == True:
            #                     if r[2] == 'RESOLUTION.':
            #                         if r[4] == 'ANGSTROMS.':
            #                             resolution = r[3]
            #                             if float(resolution) < 3:
            #                                 fw.write(name_chain + '\n')
            #                                 fq.write(str(pdb) + '-' + str(chain) + ':' + resolution + '\n')
            print(pdb + 'over')



if __name__ == "__main__":
    remove()
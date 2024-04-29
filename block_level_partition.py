from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import os
import matplotlib.pyplot as plt

# generate partition patterns
deg_bound = 12 * 32
num_warps = 12
warp_nz = [0]

d_block_rows = [0]

warp_max_nz = deg_bound // num_warps
factor = [1, 2, 3, 4, 6, 12]
jf = 0
i = 1

while i < deg_bound // 2:

    if factor[jf] * warp_max_nz >= i:
        # if i==1:
            # warp_nz_coo.append(30)
            # d_block_rows_coo.append(12)
        warp_nz.append((i + factor[jf] - 1) // factor[jf])
        d_block_rows.append(num_warps // factor[jf])
        
        i += 1
    else:
        jf += 1

# warp_nz[:9]=[0,3,4,6,12,15,24,28,24]
# warp_nz[:9]=[0,18,22,21,24,25,24,28,32]
# warp_nz[:9]=[0,18,18,18,24,25,24,28,24]
# warp_nz[:9]=[0,18,18,18,24,25,30,28,32]  #v1 workload
# warp_nz[:9]=[0,18,18,18,28,30,30,28,32]  #v2 workload
# warp_nz[:9]=[0,12,16,15,28,30,30,28,32] #v3 workload
# warp_nz[:9]=[0,24,24,24,28,30,30,28,32] #v4 workload
# warp_nz[:9]=[0,24,24,24,28,25,24,21,24]  #v5 workload
# warp_nz[:9]=[0,32,32,30,32,30,30,28,32]
# d_block_rows[:11]=[0,12,12,12,12,12,12,12,12,12,12]
warp_nz[:13]=[0,18,18,18,24,25,24,28,24,27,30,22,24]

if not os.path.exists('/data4/yul23028/MergePath-SpMM/ICCAD-Accel-GCN/block_level_meta/'):
    os.makedirs('/data4/yul23028/MergePath-SpMM/ICCAD-Accel-GCN/block_level_meta/')

base_path = '/data4/yul23028/MergePath-SpMM/ICCAD-Accel-GCN/graphs/'
# base_path = '/data4/yul23028/MergePath-SpMM/ICCAD-Accel-GCN/block_level_partition.py/'
fileset = Path(base_path).glob('*.config')

for file in fileset:
    print(file.stem)
    new_indptr = np.fromfile(base_path + file.stem + ".new_indptr", dtype=np.int32)
    new_indices = np.fromfile(base_path + file.stem + ".new_indices", dtype=np.int32)
    v_num = len(new_indptr) - 1
    e_num = len(new_indices)
    vals = np.ones(e_num)
    new_csr = csr_matrix((vals, new_indices, new_indptr))
    print(v_num, e_num)
    sorted_deg = np.ediff1d(new_csr.indptr)
    new_coo=new_csr.tocoo()
    cur_row = 0
    cur_loc = 0
    cur_degree = 0
    block_degree = []
    block_row_begin = []
    block_loc_begin = []
    block_info = []

    # block-level partitioning
    # specifically for deg==1:
    jjj=0
    jjjj=0
    while(sorted_deg[cur_row]==0):
        cur_row+=1
    # print("after all 0-deg, row number: ",cur_row)
    while((sorted_deg[cur_row]<=12) and (sorted_deg[cur_row]!=0) and (cur_row<len(new_indptr)-1)):
        cur_degree=sorted_deg[cur_row]
        b_warps=d_block_rows[cur_degree]
        warp_row=warp_nz[cur_degree]//cur_degree

        if(cur_row<(v_num) and ((cur_row+warp_row-1)<v_num)):
            while((sorted_deg[cur_row]==cur_degree) and (sorted_deg[cur_row+warp_row-1]==cur_degree)):
                if jjjj==0:
                    block_row_begin.append(cur_row)
                    block_loc_begin.append(cur_loc)
                if jjj==b_warps:
                    cur_loc+=jjj*warp_nz[cur_degree]
                    block_loc_begin.append(cur_loc)
                    block_info.append((warp_row<<16) + jjj)
                    # block_degree.append(cur_degree)
                #   print('cur_degree: ',cur_degree,'cur_degree <<16: ', cur_degree<<16)
                    block_degree.append((cur_degree<<16) + warp_nz[cur_degree])
                    item = block_degree[-1]
                    # print( ' warp_nz[cur_degree]: ',warp_nz[cur_degree],'block_degree: ', item,'block_degree >> 16: ', (item>>16))
                    block_row_begin.append(cur_row)
                    jjj=0
                    continue
                jjj+=1
                jjjj+=1
                cur_row+=warp_row
                if((cur_row+warp_row-1)>=(v_num)):
                    break
        if (jjjj>0) :
            if ((jjj>0 )):
                block_info.append((warp_row<<16) + jjj) 
                # block_degree.append(cur_degree)
                block_degree.append((cur_degree<<16) + warp_nz[cur_degree])
                cur_loc+=jjj*warp_nz[cur_degree]       
        jjjj=0

        if cur_row<=v_num:
            if (sorted_deg[cur_row]==cur_degree):
                # block_degree.append(cur_degree)
                block_degree.append((cur_degree<<16) + warp_nz[cur_degree])
                block_row_begin.append(cur_row)
                block_loc_begin.append(cur_loc)    

                while((sorted_deg[cur_row]==cur_degree)):
                    cur_row+=1

                    jjjj+=1
                    if(cur_row)>= (len(new_indptr) - 1):
                        break
                    if cur_row>=len(sorted_deg):
                        break
                    if sorted_deg[cur_row]>cur_degree :
                        # cur_row-=1
                        break    

            if(jjjj>0):
                if warp_nz[cur_degree]==0:
                    print("zero detected at: ",cur_degree, cur_row)
                block_info.append((jjjj<<16) + (int(jjjj/warp_nz[cur_degree])+1))
            # cur_row-=15

        cur_loc+=(jjjj)*cur_degree
        jjj=0
        jjjj=0
        if(cur_row>=(v_num) ):
            break



###################################################################### 
    while True:
        if cur_row>=len(sorted_deg):
            break
        if sorted_deg[cur_row] != cur_degree:
            cur_degree = sorted_deg[cur_row]

        if cur_degree == 0:
            cur_row += 1

        elif cur_degree >= 1 and cur_degree <= deg_bound:
            if cur_degree >= len(warp_nz):
                w_nz = deg_bound // num_warps
            else:
                w_nz = warp_nz[cur_degree]

            if cur_degree >= len(d_block_rows):
                b_row = 1
            else:
                b_row = d_block_rows[cur_degree]

            block_row_begin.append(cur_row)
            block_loc_begin.append(cur_loc)


            j = 0
            while sorted_deg[cur_row] == cur_degree:
                cur_row += 1
                j += 1
                if j == b_row:
                    break
                if cur_row == len(new_indptr) - 1:
                    break
            cur_loc += j * cur_degree
            # block_degree.append(cur_degree)
            block_degree.append((cur_degree<<16) + w_nz)
            block_info.append((w_nz << 16) + j) ##?

        elif cur_degree > deg_bound:
            tmp_loc = 0
            while True:
                block_degree.append((cur_degree<<16) + deg_bound)
                block_row_begin.append(cur_row)
                block_loc_begin.append(cur_loc)
                if tmp_loc + deg_bound > cur_degree:
                    block_info.append(cur_degree - tmp_loc)
                    cur_loc += cur_degree - tmp_loc
                    tmp_loc = cur_degree
                else:
                    block_info.append(deg_bound)
                    tmp_loc += deg_bound
                    cur_loc += deg_bound
                if tmp_loc == cur_degree:
                    break
            cur_row += 1

        else:
            print("cur_degree number is wrong")
            break

        if cur_row == len(new_indptr) - 1:
            break

    block_info_tmp=[]
    block_info_tmp1=[]
    for item in block_info:
        block_info_tmp1.append((item>>16))
        item = (item& 65535)
        block_info_tmp.append(item)
        

    print(f'block_degree size:{np.array(block_degree).shape}, block_row_begin size:{np.array(block_row_begin).shape}, block_loc_begin size: {np.array(block_loc_begin).shape}, block_info size:{np.array(block_info).shape}\n')
    


    # if(('low' in str(file.stem)) or ('high' in str(file.stem))):

    #     print(f'block_degree:{block_degree[:10]}, block_row_begin{block_row_begin[:10]},block_loc_begin{block_loc_begin[:10]},Needed Warps per block:{block_info_tmp[:10]},Rows per warp:{block_info_tmp1[:10]}\n' )
    
    # print(f'block_degree:{block_degree[:10]}, block_row_begin{block_row_begin[:10]},block_loc_begin{block_loc_begin[:10]},Needed Warps per block:{block_info_tmp[:10]},Rows per warp:{block_info_tmp1[:10]}\n' )
    
    for i in range(len(block_degree)):
        if (block_degree[i]>=1):
            mask = 0xFFFF 
            workload_tmp = [value & mask for value in block_degree[i:i+10]]
            block_degree_tmp = [value >> 16 for value in block_degree[i:i+10]]
            print(f'block_degree:{block_degree_tmp}，block_workload:{workload_tmp}, block_row_begin{block_row_begin[i:i+10]},block_loc_begin{block_loc_begin[i:i+10]},Needed Warps per block:{block_info_tmp[i:i+10]},Rows per warp:{block_info_tmp1[i:i+10]}\n' )
            break
    # print(f'block_degree:{block_degree[:20]}, block_row_begin{block_row_begin[:20]},block_loc_begin{block_loc_begin[:20]},Needed Warps per block:{block_info_tmp[:20]},Rows per warp:{block_info_tmp1[:20]}\n' )
    # #         break
   




    block_4 = np.dstack([block_degree, block_row_begin, block_loc_begin, block_info]).flatten()
    
    block_4.astype(np.int32).tofile('./block_level_meta/' + file.stem + '.block4')

    plt.figure(figsize=(9,6), dpi=100)
    num_bins=20

    block_degree_tmp = [value >> 16 for value in block_degree]
    n,bins,pathces=plt.hist(block_degree_tmp,num_bins,range=[0,10],color='w',edgecolor='k',hatch=r'ooo')
    # print(n,'\n',bins,'\n',pathces,'\n')
    plt.xlabel('deg of blocks')
    plt.ylabel('count')
    plt.title(f'Distribution of blocks by their degree: {os.path.basename(file)}')
    plt.legend()
    # plt.show()
    plt.savefig(f'{file}.png')



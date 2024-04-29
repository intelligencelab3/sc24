#include "spmm_accel.h"
#include "data.h"
#include <string>
#include <iostream>
#define CONSTINT const int

using namespace std;

extern string base_dir, graph;

string block_meta_dir = "../block_level_meta/";

const int DEG_BOUND = 12 * 32;
const int WARPS_PER_BLOCK = 12;

#define DIM_MUL(x) ((x + 31) / 32) * 32

__global__ void spmm_kernel_accel(const int *_block4, const int *coo_row, const int *idx, const float *val, const float *vin, float *vout, const int num_v, const int num_e, const int RHS_dim, const float *vout_ref)
{
    const int4 *block4 = reinterpret_cast<const int4 *>(_block4);
    const int4 b_info = block4[blockIdx.x]; //Consider use serveral int32 entries to store low-degree information to ease the bandwidth of bus

    CONSTINT block_degree_workload = b_info.x; 
    CONSTINT block_row_begin = b_info.y;
    CONSTINT block_loc_begin = b_info.z;
    CONSTINT block_info = b_info.w;

    CONSTINT block_degree=(block_degree_workload>>16);
    CONSTINT n_rows = (block_degree) <= DEG_BOUND ? block_info & 65535 : 1;//how many rows per block
    CONSTINT w_nz = block_degree <= DEG_BOUND ? block_info >> 16 : DEG_BOUND / WARPS_PER_BLOCK; //? put two info in 32 bit space each take 16bit how many nz's per warp
    CONSTINT row_nz = block_degree <= DEG_BOUND ? block_degree : block_info;// how many nz's per row
    CONSTINT workload = (block_degree_workload)&65535;
    extern __shared__ float out_cache[];
    // extern __shared__ float _val[];
    CONSTINT round_dim = DIM_MUL(RHS_dim); //what is RHS_dim? dimension of RHS matrix 
    // CONSTINT round_dim = RHS_dim;
    CONSTINT ext_1=round_dim/32;

    // printf("ext_1: %d \n", ext_1);
    // printf("RHS_dim: %d \n", RHS_dim);
    // printf("round_dim: %d \n", round_dim);
    // printf("(round_dim/ext_1): %d \n", (round_dim/ext_1));
    // printf("block_degree: %d, workload: %d\n", block_degree,workload);

    // return;

    CONSTINT warps_per_row = (row_nz + w_nz - 1) / w_nz;//?when RHS matrix is large

    #pragma unroll
    for (int ext = 0; ext < (RHS_dim + 31) / 32; ext++)//ext stands for column's dimension
    {        
        
        CONSTINT lane_id = (threadIdx.x +ext * blockDim.x)%round_dim;
 
        if(lane_id>=RHS_dim){

            return;

        }

         CONSTINT wid = (threadIdx.x + ext * blockDim.x) / round_dim;


        CONSTINT tid = wid * round_dim + lane_id;//tid== thread id


    if(block_degree<=12){
        if((wid%12)>=(n_rows)){
            // printf("wid: %d, how many rows(=warps) should be for this 1-deg chunk: %d\n",wid,n_rows);
            return;
        }
        // #pragma unroll
        // for (int jjj=0;jjj<ext_1;jjj++){


            if((block_degree==1)){
        

                    #pragma unroll
                    for(int jj=0;jj<(w_nz);jj++){   

                        const int nz_loc= block_loc_begin+(wid%12*32)-(wid%12)*(32-workload)+jj; // 1 warp to one nz's use for loop to access each of 30 nz's in a row, multiplied with 1 row of RHS(32 dim)
                        // printf("nz_loc: %d\n", nz_loc);
                        const int row_offset = (wid%12*32)-(wid%12)*(32-workload)+jj;


                        const float left_val = val[nz_loc];


                        float right_val = vin[idx[nz_loc] * RHS_dim+ lane_id];
                        vout[(block_row_begin + row_offset) * RHS_dim + lane_id] =left_val * right_val;

                        
                        
                    }
                    
        }

    else if((block_degree==2)){

            if((wid%12)>=(n_rows)){
                // printf("wid: %d, how many rows(=warps) should be for this 1-deg chunk: %d\n",wid,n_rows);
                return;
            }
                #pragma unroll
                 for(int jj=0;jj<(w_nz*2);jj+=2){   

                    const int nz_loc= block_loc_begin+(wid%12*32)-(wid%12)*(32-workload)+jj;
                    const int row_offset = ((wid%12*32)-(wid%12)*(32-workload)+jj)/2;
                    const float left_val_1 = __ldg(val+nz_loc);
                    const float left_val_2 = __ldg(val+nz_loc+1);


                    float right_val1 = vin[__ldg(idx + nz_loc)*RHS_dim + lane_id];
                    float right_val2 = vin[__ldg(idx + nz_loc+1)*RHS_dim + lane_id];

                    // for (int jjjj=0;jjjj<block_degree;jjjj++){
                    //     float left_val = __ldg(val+nz_loc+jjjj);

                    //     float right_val = vin[__ldg(idx + nz_loc+jjjj)*RHS_dim + lane_id];

                    //     vout[(block_row_begin + row_offset)*RHS_dim +lane_id] += right_val * left_val;
                    // }
                            
                    vout[(block_row_begin + row_offset)*RHS_dim +lane_id] +=left_val_1 * right_val1 + left_val_2 * right_val2;
                       
                }

            // }

            }

else if((block_degree==3)){

                    #pragma unroll
                    for(int jj=0;jj<(w_nz*3);jj+=3){   

                        const int nz_loc= block_loc_begin+(wid%12*32)-wid%12*(32-workload)+jj; // 1 warp to one nz's use for loop to access each of 30 nz's in a row, multiplied with 1 row of RHS(32 dim)
                        const int row_offset = ((wid%12*32)-wid%12*(32-workload)+jj)/3;
                        // const int cur_row= block_row_begin+((wid%12*32)-(wid%12)*(32%18)+jj)/3;

                        // const float left_val1 = __ldg(val+nz_loc);
                        // const float left_val2 = __ldg(val+nz_loc+1);
                        // const float left_val3 = __ldg(val+nz_loc+2);

                        // float right_val1 = vin[__ldg(idx + nz_loc)*RHS_dim + lane_id];
                        // float right_val2 = vin[__ldg(idx + nz_loc+1)*RHS_dim + lane_id];
                        // float right_val3 = vin[__ldg(idx + nz_loc+2)*RHS_dim + lane_id];

                        for (int jjjj=0;jjjj<block_degree;jjjj++){
                            float left_val = __ldg(val+nz_loc+jjjj);
    
                            float right_val = vin[__ldg(idx + nz_loc+jjjj)*RHS_dim + lane_id];
    
                            vout[(block_row_begin + row_offset)*RHS_dim +lane_id] += right_val * left_val;
                        }
 
                        // vout[(block_row_begin + row_offset)*RHS_dim + lane_id] +=left_val1 * right_val1 +left_val2 * right_val2+left_val3 * right_val3;
                                               

                                                    }

                    }


else if((block_degree==4)){

    #pragma unroll
  for(int jj=0;jj<(w_nz*4);jj+=4){   

    const int nz_loc= block_loc_begin+(wid%12*32)-(wid%12)*(32-workload)+jj; // 1 warp to one nz's use for loop to access each of 30 nz's in a row, multiplied with 1 row of RHS(32 dim)
    const int row_offset = ((wid%12*32)-(wid%12)*(32-workload)+jj)/4;
    // const int cur_row= block_row_begin+((wid%12*32)-(wid%12)*(32%24)+jj)/4;

    const float left_val1 = __ldg(val+nz_loc);
    const float left_val2 = __ldg(val+nz_loc+1);
    const float left_val3 = __ldg(val+nz_loc+2);
    const float left_val4 = __ldg(val+nz_loc+3);

    float right_val1 = vin[__ldg(idx + nz_loc)*RHS_dim + lane_id];
    float right_val2 = vin[__ldg(idx + nz_loc+1)*RHS_dim + lane_id];
    float right_val3 = vin[__ldg(idx + nz_loc+2)*RHS_dim + lane_id];
    float right_val4 = vin[__ldg(idx + nz_loc+3)*RHS_dim + lane_id];

    vout[(block_row_begin + row_offset)*RHS_dim +lane_id] +=left_val1 * right_val1 +left_val2 * right_val2+left_val3 * right_val3+left_val4 * right_val4;
                            
                                }
                }
    
    else if((block_degree==5)){

                #pragma unroll
              for(int jj=0;jj<(w_nz*5);jj+=5){   
            
                const int nz_loc= block_loc_begin+(wid%12*32)-(wid%12)*(32-workload)+jj; // 1 warp to one nz's use for loop to access each of 30 nz's in a row, multiplied with 1 row of RHS(32 dim)
                const int row_offset = ((wid%12*32)-(wid%12)*(32-workload)+jj)/5;
                // const int cur_row= block_row_begin+((wid%12*32)-(wid%12)*(32%25)+jj)/5;
            
                const float left_val1 = __ldg(val+nz_loc);
                const float left_val2 = __ldg(val+nz_loc+1);
                const float left_val3 = __ldg(val+nz_loc+2);
                const float left_val4 = __ldg(val+nz_loc+3);
                const float left_val5 = __ldg(val+nz_loc+4);
                

                float right_val1 = vin[__ldg(idx + nz_loc)*RHS_dim + lane_id];
                float right_val2 = vin[__ldg(idx + nz_loc+1)*RHS_dim + lane_id];
                float right_val3 = vin[__ldg(idx + nz_loc+2)*RHS_dim + lane_id];
                float right_val4 = vin[__ldg(idx + nz_loc+3)*RHS_dim + lane_id];
                float right_val5 = vin[__ldg(idx + nz_loc+4)*RHS_dim + lane_id];

                vout[(block_row_begin + row_offset)*RHS_dim + lane_id] +=left_val1 * right_val1 +left_val2 * right_val2+left_val3 * right_val3+left_val4 * right_val4 + left_val5 * right_val5;


                                }
                            } 
        else if((block_degree==6)){
        


                                        #pragma unroll
                                      for(int jj=0;jj<(w_nz*6);jj+=6){   
                                    
                                        const int nz_loc= block_loc_begin+(wid%12*32)-(wid%12)*(32-workload)+jj; // 1 warp to one nz's use for loop to access each of 30 nz's in a row, multiplied with 1 row of RHS(32 dim)
                                        const int row_offset = ((wid%12*32)-(wid%12)*(32-workload)+jj)/6;
                                        // const int cur_row= block_row_begin+((wid%12*32)-(wid%12)*(32%24)+jj)/6;
                                    
                                        const float left_val1 = __ldg(val+nz_loc);
                                        const float left_val2 = __ldg(val+nz_loc+1);
                                        const float left_val3 = __ldg(val+nz_loc+2);
                                        const float left_val4 = __ldg(val+nz_loc+3);
                                        const float left_val5 = __ldg(val+nz_loc+4);
                                        const float left_val6 = __ldg(val+nz_loc+5);
 
                                        float right_val1 = vin[__ldg(idx + nz_loc)*RHS_dim + lane_id];
                                        float right_val2 = vin[__ldg(idx + nz_loc+1)*RHS_dim + lane_id];
                                        float right_val3 = vin[__ldg(idx + nz_loc+2)*RHS_dim + lane_id];
                                        float right_val4 = vin[__ldg(idx + nz_loc+3)*RHS_dim + lane_id];
                                        float right_val5 = vin[__ldg(idx + nz_loc+4)*RHS_dim + lane_id];
                                        float right_val6 = vin[__ldg(idx + nz_loc+5)*RHS_dim + lane_id];
            
                                        vout[(block_row_begin + row_offset)*RHS_dim + lane_id] +=left_val1 * right_val1 +left_val2 * right_val2+left_val3 * right_val3+left_val4 * right_val4 + left_val5 * right_val5+ left_val6 * right_val6;

                                                                    }
                                                    } 

        else if((block_degree==7)){



                    #pragma unroll
                  for(int jj=0;jj<(w_nz*7);jj+=7){   
                
                    const int nz_loc= block_loc_begin+(wid%12*32)-(wid%12)*(32-workload)+jj; // 1 warp to one nz's use for loop to access each of 30 nz's in a row, multiplied with 1 row of RHS(32 dim)
                    const int row_offset = ((wid%12*32)-(wid%12)*(32-workload)+jj)/7;
                    // const int cur_row= block_row_begin+((wid%12*32)-(wid%12)*(32-workload)+jj)/7;
                
                    const float left_val1 = __ldg(val+nz_loc);
                    const float left_val2 = __ldg(val+nz_loc+1);
                    const float left_val3 = __ldg(val+nz_loc+2);
                    const float left_val4 = __ldg(val+nz_loc+3);
                    const float left_val5 = __ldg(val+nz_loc+4);
                    const float left_val6 = __ldg(val+nz_loc+5);
                    const float left_val7 = __ldg(val+nz_loc+6);

                // for(int jjj=0;jjj<ext_1;jjj++){
                    float right_val1 = vin[__ldg(idx + nz_loc)*RHS_dim + lane_id];
                    float right_val2 = vin[__ldg(idx + nz_loc+1)*RHS_dim + lane_id];
                    float right_val3 = vin[__ldg(idx + nz_loc+2)*RHS_dim + lane_id];
                    float right_val4 = vin[__ldg(idx + nz_loc+3)*RHS_dim + lane_id];
                    float right_val5 = vin[__ldg(idx + nz_loc+4)*RHS_dim + lane_id];
                    float right_val6 = vin[__ldg(idx + nz_loc+5)*RHS_dim + lane_id];
                    float right_val7 = vin[__ldg(idx + nz_loc+6)*RHS_dim + lane_id];


                    vout[(block_row_begin + row_offset)*RHS_dim +lane_id] +=left_val1 * right_val1 +left_val2 * right_val2+left_val3 * right_val3+left_val4 * right_val4 + left_val5 * right_val5+ left_val6 * right_val6 +left_val7 * right_val7;
                

                                                }
                                } 

        else if((block_degree==8)){
               
                                        #pragma unroll
                                      for(int jj=0;jj<(w_nz*8);jj+=8){   
                                    
                                        const int nz_loc= block_loc_begin+(wid%12*32)-(wid%12)*(32-workload)+jj; // 1 warp to one nz's use for loop to access each of 30 nz's in a row, multiplied with 1 row of RHS(32 dim)
                                        const int row_offset = ((wid%12*32)-(wid%12)*(32-workload)+jj)/8;
                                        // const int cur_row= block_row_begin+((wid%12*32)-(wid%12)*(32-workload)+jj)/8;
                                    
                                        const float left_val1 = __ldg(val+nz_loc);
                                        const float left_val2 = __ldg(val+nz_loc+1);
                                        const float left_val3 = __ldg(val+nz_loc+2);
                                        const float left_val4 = __ldg(val+nz_loc+3);
                                        const float left_val5 = __ldg(val+nz_loc+4);
                                        const float left_val6 = __ldg(val+nz_loc+5);
                                        const float left_val7 = __ldg(val+nz_loc+6);
                                        const float left_val8 = __ldg(val+nz_loc+7);
                    

            

                            float right_val1 = vin[__ldg(idx + nz_loc)*RHS_dim + lane_id];
                            float right_val2 = vin[__ldg(idx + nz_loc+1)*RHS_dim + lane_id];
                            float right_val3 = vin[__ldg(idx + nz_loc+2)*RHS_dim + lane_id];
                            float right_val4 = vin[__ldg(idx + nz_loc+3)*RHS_dim + lane_id];
                            float right_val5 = vin[__ldg(idx + nz_loc+4)*RHS_dim + lane_id];
                            float right_val6 = vin[__ldg(idx + nz_loc+5)*RHS_dim + lane_id];
                            float right_val7 = vin[__ldg(idx + nz_loc+6)*RHS_dim + lane_id];
                            float right_val8 = vin[__ldg(idx + nz_loc+7)*RHS_dim + lane_id];

                        
                            vout[(block_row_begin + row_offset)*RHS_dim +lane_id] +=left_val1 * right_val1 +left_val2 * right_val2+left_val3 * right_val3+left_val4 * right_val4 + left_val5 * right_val5+ left_val6 * right_val6 +left_val7 * right_val7+left_val8 * right_val8;
                        


                                                                    }
                                                    } 
                else if((block_degree==9)){
                    #pragma unroll
                    for(int jj=0;jj<(w_nz*9);jj+=9){   
                  
                      const int nz_loc= block_loc_begin+(wid%12*32)-(wid%12)*(32-workload)+jj; // 1 warp to one nz's use for loop to access each of 30 nz's in a row, multiplied with 1 row of RHS(32 dim)
                      const int row_offset = ((wid%12*32)-(wid%12)*(32-workload)+jj)/9;
                      // const int cur_row= block_row_begin+((wid%12*32)-(wid%12)*(32-workload)+jj)/8;
                  
                      const float left_val1 = __ldg(val+nz_loc);
                      const float left_val2 = __ldg(val+nz_loc+1);
                      const float left_val3 = __ldg(val+nz_loc+2);
                      const float left_val4 = __ldg(val+nz_loc+3);
                      const float left_val5 = __ldg(val+nz_loc+4);
                      const float left_val6 = __ldg(val+nz_loc+5);
                      const float left_val7 = __ldg(val+nz_loc+6);
                      const float left_val8 = __ldg(val+nz_loc+7);
                      const float left_val9 = __ldg(val+nz_loc+8);
  



                        float right_val1 = vin[__ldg(idx + nz_loc)*RHS_dim + lane_id];
                        float right_val2 = vin[__ldg(idx + nz_loc+1)*RHS_dim + lane_id];
                        float right_val3 = vin[__ldg(idx + nz_loc+2)*RHS_dim + lane_id];
                        float right_val4 = vin[__ldg(idx + nz_loc+3)*RHS_dim + lane_id];
                        float right_val5 = vin[__ldg(idx + nz_loc+4)*RHS_dim + lane_id];
                        float right_val6 = vin[__ldg(idx + nz_loc+5)*RHS_dim + lane_id];
                        float right_val7 = vin[__ldg(idx + nz_loc+6)*RHS_dim + lane_id];
                        float right_val8 = vin[__ldg(idx + nz_loc+7)*RHS_dim + lane_id];
                        float right_val9 = vin[__ldg(idx + nz_loc+8)*RHS_dim + lane_id];

      
          vout[(block_row_begin + row_offset)*RHS_dim +lane_id] +=left_val1 * right_val1 +left_val2 * right_val2+left_val3 * right_val3+left_val4 * right_val4 + left_val5 * right_val5+ left_val6 * right_val6 +left_val7 * right_val7+left_val8 * right_val8+left_val9 * right_val9;
      

                }
                                                }

                else if((block_degree==10)){
                    #pragma unroll
                    for(int jj=0;jj<(w_nz*10);jj+=10){   
                  
                      const int nz_loc= block_loc_begin+(wid%12*32)-(wid%12)*(32-workload)+jj; // 1 warp to one nz's use for loop to access each of 30 nz's in a row, multiplied with 1 row of RHS(32 dim)
                      const int row_offset = ((wid%12*32)-(wid%12)*(32-workload)+jj)/10;
                      // const int cur_row= block_row_begin+((wid%12*32)-(wid%12)*(32-workload)+jj)/8;
                  
                      const float left_val1 = __ldg(val+nz_loc);
                      const float left_val2 = __ldg(val+nz_loc+1);
                      const float left_val3 = __ldg(val+nz_loc+2);
                      const float left_val4 = __ldg(val+nz_loc+3);
                      const float left_val5 = __ldg(val+nz_loc+4);
                      const float left_val6 = __ldg(val+nz_loc+5);
                      const float left_val7 = __ldg(val+nz_loc+6);
                      const float left_val8 = __ldg(val+nz_loc+7);
                      const float left_val9 = __ldg(val+nz_loc+8);
                      const float left_val10 = __ldg(val+nz_loc+9);
  



                        float right_val1 = vin[__ldg(idx + nz_loc)*RHS_dim + lane_id];
                        float right_val2 = vin[__ldg(idx + nz_loc+1)*RHS_dim + lane_id];
                        float right_val3 = vin[__ldg(idx + nz_loc+2)*RHS_dim + lane_id];
                        float right_val4 = vin[__ldg(idx + nz_loc+3)*RHS_dim + lane_id];
                        float right_val5 = vin[__ldg(idx + nz_loc+4)*RHS_dim + lane_id];
                        float right_val6 = vin[__ldg(idx + nz_loc+5)*RHS_dim + lane_id];
                        float right_val7 = vin[__ldg(idx + nz_loc+6)*RHS_dim + lane_id];
                        float right_val8 = vin[__ldg(idx + nz_loc+7)*RHS_dim + lane_id];
                        float right_val9 = vin[__ldg(idx + nz_loc+8)*RHS_dim + lane_id];
                        float right_val10 = vin[__ldg(idx + nz_loc+9)*RHS_dim + lane_id];

      
          vout[(block_row_begin + row_offset)*RHS_dim +lane_id] +=left_val1 * right_val1 +left_val2 * right_val2+left_val3 * right_val3+left_val4 * right_val4 + left_val5 * right_val5+ left_val6 * right_val6 +left_val7 * right_val7+left_val8 * right_val8+left_val9 * right_val9+left_val10 * right_val10;
      

                }                                 
                                                    
            }

            else if((block_degree==11)){
                #pragma unroll
                for(int jj=0;jj<(w_nz*11);jj+=11){   
              
                  const int nz_loc= block_loc_begin+(wid%12*32)-(wid%12)*(32-workload)+jj; // 1 warp to one nz's use for loop to access each of 30 nz's in a row, multiplied with 1 row of RHS(32 dim)
                  const int row_offset = ((wid%12*32)-(wid%12)*(32-workload)+jj)/11;
                  // const int cur_row= block_row_begin+((wid%12*32)-(wid%12)*(32-workload)+jj)/8;
              
                  const float left_val1 = __ldg(val+nz_loc);
                  const float left_val2 = __ldg(val+nz_loc+1);
                  const float left_val3 = __ldg(val+nz_loc+2);
                  const float left_val4 = __ldg(val+nz_loc+3);
                  const float left_val5 = __ldg(val+nz_loc+4);
                  const float left_val6 = __ldg(val+nz_loc+5);
                  const float left_val7 = __ldg(val+nz_loc+6);
                  const float left_val8 = __ldg(val+nz_loc+7);
                  const float left_val9 = __ldg(val+nz_loc+8);
                  const float left_val10 = __ldg(val+nz_loc+9);
                  const float left_val11 = __ldg(val+nz_loc+10);




                    float right_val1 = vin[__ldg(idx + nz_loc)*RHS_dim + lane_id];
                    float right_val2 = vin[__ldg(idx + nz_loc+1)*RHS_dim + lane_id];
                    float right_val3 = vin[__ldg(idx + nz_loc+2)*RHS_dim + lane_id];
                    float right_val4 = vin[__ldg(idx + nz_loc+3)*RHS_dim + lane_id];
                    float right_val5 = vin[__ldg(idx + nz_loc+4)*RHS_dim + lane_id];
                    float right_val6 = vin[__ldg(idx + nz_loc+5)*RHS_dim + lane_id];
                    float right_val7 = vin[__ldg(idx + nz_loc+6)*RHS_dim + lane_id];
                    float right_val8 = vin[__ldg(idx + nz_loc+7)*RHS_dim + lane_id];
                    float right_val9 = vin[__ldg(idx + nz_loc+8)*RHS_dim + lane_id];
                    float right_val10 = vin[__ldg(idx + nz_loc+9)*RHS_dim + lane_id];
                    float right_val11 = vin[__ldg(idx + nz_loc+10)*RHS_dim + lane_id];

  
      vout[(block_row_begin + row_offset)*RHS_dim +lane_id] +=left_val1 * right_val1 +left_val2 * right_val2+left_val3 * right_val3+left_val4 * right_val4 + left_val5 * right_val5+ left_val6 * right_val6 +left_val7 * right_val7+left_val8 * right_val8+left_val9 * right_val9+left_val10 * right_val10+left_val11 * right_val11;
  
            }

        }
        else if((block_degree==12)){
            #pragma unroll
            for(int jj=0;jj<(w_nz*12);jj+=12){   
          
              const int nz_loc= block_loc_begin+(wid%12*32)-(wid%12)*(32-workload)+jj; // 1 warp to one nz's use for loop to access each of 30 nz's in a row, multiplied with 1 row of RHS(32 dim)
              const int row_offset = ((wid%12*32)-(wid%12)*(32-workload)+jj)/12;
              // const int cur_row= block_row_begin+((wid%12*32)-(wid%12)*(32-workload)+jj)/8;
          
              const float left_val1 = __ldg(val+nz_loc);
              const float left_val2 = __ldg(val+nz_loc+1);
              const float left_val3 = __ldg(val+nz_loc+2);
              const float left_val4 = __ldg(val+nz_loc+3);
              const float left_val5 = __ldg(val+nz_loc+4);
              const float left_val6 = __ldg(val+nz_loc+5);
              const float left_val7 = __ldg(val+nz_loc+6);
              const float left_val8 = __ldg(val+nz_loc+7);
              const float left_val9 = __ldg(val+nz_loc+8);
              const float left_val10 = __ldg(val+nz_loc+9);
              const float left_val11 = __ldg(val+nz_loc+10);
              const float left_val12 = __ldg(val+nz_loc+11);


                float right_val1 = vin[__ldg(idx + nz_loc)*RHS_dim + lane_id];
                float right_val2 = vin[__ldg(idx + nz_loc+1)*RHS_dim + lane_id];
                float right_val3 = vin[__ldg(idx + nz_loc+2)*RHS_dim + lane_id];
                float right_val4 = vin[__ldg(idx + nz_loc+3)*RHS_dim + lane_id];
                float right_val5 = vin[__ldg(idx + nz_loc+4)*RHS_dim + lane_id];
                float right_val6 = vin[__ldg(idx + nz_loc+5)*RHS_dim + lane_id];
                float right_val7 = vin[__ldg(idx + nz_loc+6)*RHS_dim + lane_id];
                float right_val8 = vin[__ldg(idx + nz_loc+7)*RHS_dim + lane_id];
                float right_val9 = vin[__ldg(idx + nz_loc+8)*RHS_dim + lane_id];
                float right_val10 = vin[__ldg(idx + nz_loc+9)*RHS_dim + lane_id];
                float right_val11 = vin[__ldg(idx + nz_loc+10)*RHS_dim + lane_id];
                float right_val12 = vin[__ldg(idx + nz_loc+11)*RHS_dim + lane_id];


  vout[(block_row_begin + row_offset)*RHS_dim +lane_id] +=left_val1 * right_val1 +left_val2 * right_val2+left_val3 * right_val3+left_val4 * right_val4 + left_val5 * right_val5+ left_val6 * right_val6 +left_val7 * right_val7+left_val8 * right_val8+left_val9 * right_val9+left_val10 * right_val10+left_val11 * right_val11+left_val12 * right_val12;

        }
        }

    // }
    }
            else if(block_degree>12){


                    
                    CONSTINT warp_loc_row = wid / warps_per_row; 
                    CONSTINT warp_loc_col = wid % warps_per_row * w_nz;
            
                    if (warp_loc_row >= n_rows)
                    {
                        return;
                    }
            
                    //decide how many nz's for 1 warp
                    //based on the degree of incoming row
                    //find nz's location at one time
                    //then perform the spmm mul_add
            
            #pragma unroll
                    for (int i = 0; i < w_nz; i++)
                    {
                        if (i + warp_loc_col >= row_nz)
                        {
                            break;
                        }
                        if (i == 0)
                        {
                            out_cache[tid] = 0;

                        }
                        const int nz_loc = block_loc_begin + warp_loc_row * row_nz + i + warp_loc_col;
                        const float left_val = __ldg(val + nz_loc);
            
                        float right_val = vin[__ldg(idx + nz_loc) * RHS_dim + lane_id];
                        out_cache[tid] += left_val * right_val;

                    }
            
                    // atomicAdd(&vout[(block_row_begin + warp_loc_row) * RHS_dim + lane_id], out_cache[wid * round_dim + lane_id]);
                    if (warps_per_row > 1)
                    {
                        atomicAdd(&vout[(block_row_begin + warp_loc_row) * RHS_dim + lane_id], out_cache[tid]);
                        
                    }
                    else
                    {
                        if (block_degree <= DEG_BOUND)
                        {
            
                            vout[(block_row_begin + wid) * RHS_dim + lane_id] = out_cache[tid];
                        }
            
                        else
                        {
                            atomicAdd(&vout[(block_row_begin + wid) * RHS_dim + lane_id], out_cache[tid]);
                        }
                    }
                    
                
            }
        }

 
        }


    
   
                    

            
    




void SPMM_ACCEL::run(int dim)
{    
    int shared_size = (WARPS_PER_BLOCK + 0 * WARPS_PER_BLOCK / 2) * DIM_MUL(dim) * sizeof(float);//is this dim the RHS_dim?
    spmm_kernel_accel<<<grid, block, shared_size>>>(_block4, 0, idx, val, vin, vout, num_v, num_e, dim, 0);
}

double SPMM_ACCEL::do_test(bool timing, int dim)
{
    // cudaMallocManaged(&coo_row, num_e * sizeof(int));
    // int k = 0;
    // for (int i = 0; i < num_v; i++)
    // {
    //     for (int j = 0; j < ptr[i + 1] - ptr[i]; j++)
    //     {
    //         coo_row[k++] = i;
    //     }
    // }

    // int block_num = cuda_read_array(&this->_block4, "../block_level_meta/" + this->_graph + ".block4") / 4;
    int block_num = cuda_read_array(&this->_block4, block_meta_dir + this->_graph + ".block4") / 4;
 
    if (!timing)
    {
    //    cout << "block num = " << block_num << endl;
    }

    grid.x = block_num;

    // printf("block_num:%d\n",block_num);

    // block.x = DIM_MUL(dim);
    // block.y = WARPS_PER_BLOCK;
    block.x = WARPS_PER_BLOCK * 32;

    double ret = timing_body(timing, dim); //probably too few blocks

    // cudaFree(coo_row);
    cudaFree(this->_block4);
    return ret;
}
#include "test_slice_grappa.h"
#include "cuNDArray_operators.h"
#include "cuNDArray_elemwise.h"
#include "vector_td_utilities.h"
#include "real_utilities.h"
#include "real_utilities_device.h"
#include "complext.h"
#include "check_CUDA.h"
#include "cudaDeviceManager.h"
#include "setup_grid.h"
#include "GPUTimer.h"
#include <iostream>
#include <cmath>
#include "CUBLASContextProvider.h"
#include <cublas_v2.h>

using namespace std;

namespace Gadgetron{

/*  const int kernel_width = 7;

  template<class REAL, unsigned int D> static void smooth_correlation_matrices( cuNDArray<complext<REAL> >*, cuNDArray<complext<REAL> >*);
  template<class REAL> static cuNDArray<complext<REAL>> extract_csm(const cuNDArray<complext<REAL> >&, unsigned int, unsigned int);
  template<class REAL> static void set_phase_reference( cuNDArray<complext<REAL> >*, unsigned int, unsigned int);
  template<class T> static void find_stride( cuNDArray<T> *in, unsigned int dim, unsigned int *stride, std::vector<size_t> *dims );
  template<class T> static boost::shared_ptr< cuNDArray<T> > correlation( cuNDArray<T> *in );
  template<class T> static void rss_normalize( cuNDArray<T> *in_out, unsigned int dim );
*/
//
// Main method
//

extern __shared__ char _shared_mem[];


template<class REAL, unsigned int D> cuNDArray<complext<REAL> >
estimate_feasibility( const cuNDArray<complext<REAL> >& data_in, int target_coils)
{

    if( data_in.get_number_of_dimensions() < 1 )
    {
        cout << endl << "estimate_feasibility:: dimensionality mismatch." << endl;

    }
    else
    {
        cout << endl << "estimate_feasibility:: dimensionality seems ok." << endl;
    }

    // Allocate output
    cuNDArray<complext<REAL> > out = data_in;

    size_t NDim = data_in.get_number_of_dimensions();

    size_t RO = data_in.get_size(0);
    size_t E1 = data_in.get_size(1);
    size_t N = data_in.get_size(2);
    size_t CHA = data_in.get_size(3);

    return out;
}


template<class REAL> bool
estimate_feasibility_no_STK( cuNDArray<complext<REAL> >& data_in, cuNDArray<complext<REAL> >& data_out, const size_t acc , const size_t startE1, const size_t endE1, const size_t reduceE1)
{
    boost::shared_ptr<GPUTimer> process_timer;
    process_timer = boost::shared_ptr<GPUTimer>( new GPUTimer("gpuExample::remove_encoding_no_STK local") );
    remove_encoding_no_STK( &data_out, &data_in ,  acc , startE1,  endE1, reduceE1);
    process_timer.reset();

    return true;
}


template<class REAL> bool
estimate_feasibility_with_STK( cuNDArray<complext<REAL> >& data_in, cuNDArray<complext<REAL> >& data_out, const size_t acc , const size_t startE1, const size_t endE1, const size_t reduceE1)
{
    boost::shared_ptr<GPUTimer> process_timer;
    process_timer = boost::shared_ptr<GPUTimer>( new GPUTimer("gpuExample::remove_encoding_with_STK local") );
    remove_encoding_with_STK( &data_out, &data_in ,  acc , startE1,  endE1, reduceE1);
    process_timer.reset();

    return true;
}


template<class REAL> bool
prepare_im2col_2D( cuNDArray<complext<REAL> >& data_in, cuNDArray<complext<REAL> >& data_out, const size_t blocks_RO, const size_t blocks_E1, const size_t grappa_kSize_RO, const size_t grappa_kSize_E1)
{
    boost::shared_ptr<GPUTimer> process_timer;
    process_timer = boost::shared_ptr<GPUTimer>( new GPUTimer("gpuExample::compute_im2col_gpu 2D local") );
    compute_im2col_2D( &data_out, &data_in ,  blocks_RO , blocks_E1,  grappa_kSize_RO, grappa_kSize_E1);
    process_timer.reset();

    return true;
}

template<class REAL> bool
prepare_im2col_5D( cuNDArray<complext<REAL> >& data_in, cuNDArray<complext<REAL> >& data_out, const size_t blocks_RO, const size_t blocks_E1, const size_t grappa_kSize_RO, const size_t grappa_kSize_E1)
{
    boost::shared_ptr<GPUTimer> process_timer;
    process_timer = boost::shared_ptr<GPUTimer>( new GPUTimer("gpuExample::compute_im2col_gpu 5D local") );
    compute_im2col_5D( &data_out, &data_in ,  blocks_RO , blocks_E1,  grappa_kSize_RO, grappa_kSize_E1);
    process_timer.reset();

    return true;
}






template<class REAL> bool
prepare_EPI_corr_5D( bool undo, bool optimal,   cuNDArray<complext<REAL> >& data_in, cuNDArray<complext<REAL> >& pos, cuNDArray<complext<REAL> >& neg ,cuNDArray<complext<REAL> >& pos_mean, cuNDArray<complext<REAL> >& neg_mean, cuNDArray<int >& reverse_line)
{
    //boost::shared_ptr<GPUTimer> process_timer;
    //process_timer = boost::shared_ptr<GPUTimer>( new GPUTimer("gpuExample::compute_EPI_corr_5D 5D local") );
    compute_EPI_coor_5D(undo, optimal,  &data_in, &pos, &neg, &pos_mean, &neg_mean, &reverse_line);
    //process_timer.reset();

    return true;
}


template<class REAL> bool
prepare_gpu_unmix(   cuNDArray<complext<REAL> >& in, cuNDArray<complext<REAL> >& kernel, cuNDArray<complext<REAL> >& out )
{
    //boost::shared_ptr<GPUTimer> process_timer;
    //process_timer = boost::shared_ptr<GPUTimer>( new GPUTimer("gpuExample::prepare_gpu_unmix  local") );
    compute_gpu_unmix(&in,  &kernel, &out );
    //process_timer.reset();

    return true;
}

template EXPORTGPUPMRI bool prepare_im2col_2D( cuNDArray<complext<float> >& , cuNDArray<complext<float> >& , const size_t , const size_t , const size_t , const size_t );
template EXPORTGPUPMRI bool prepare_im2col_5D( cuNDArray<complext<float> >& , cuNDArray<complext<float> >& , const size_t , const size_t , const size_t , const size_t );
template EXPORTGPUPMRI bool prepare_EPI_corr_5D(bool, bool,  cuNDArray<complext<float> >& , cuNDArray<complext<float> >& , cuNDArray<complext<float> >&  ,cuNDArray<complext<float> >& , cuNDArray<complext<float> >& , cuNDArray<int >& );
template EXPORTGPUPMRI bool prepare_gpu_unmix(  cuNDArray<complext<float> >& , cuNDArray<complext<float> >& , cuNDArray<complext<float> >&  );


template EXPORTGPUPMRI  cuNDArray<complext<float>> estimate_feasibility<float,2>(const cuNDArray<complext<float> >&, int);
//template EXPORTGPUPMRI  cuNDArray<complext<float>> estimate_feasibility<float>( cuNDArray<complext<float> >&);
template EXPORTGPUPMRI  bool estimate_feasibility_no_STK<float>( cuNDArray<complext<float> >&, cuNDArray<complext<float> >&  , const size_t  , const size_t , const size_t , const size_t );
template EXPORTGPUPMRI  bool estimate_feasibility_with_STK<float>( cuNDArray<complext<float> >&, cuNDArray<complext<float> >&  , const size_t  , const size_t , const size_t , const size_t );



template<class T> __global__ static void
remove_encoding_kernel_no_STK( T *out,  T *in, unsigned int RO, unsigned int E1, unsigned int CHA, unsigned int MB,   size_t acc ,  size_t startE1,  size_t endE1, size_t reduceE1)
{
    unsigned int ro = blockIdx.x * blockDim.x + threadIdx.x;

    //erreur peut-être ici ?
    unsigned int cha =  blockIdx.y%CHA;
    unsigned int mb =  (blockIdx.y- cha) / CHA  ;

    //unsigned int stk = (blockIdx.y)/(CHA*MB);
    // unsigned int mb = (blockIdx.y - stk*CHA*MB)/CHA;
    //unsigned int cha = (blockIdx.y - stk*CHA*MB)%CHA;

    if (ro>=0 && ro < RO && cha < CHA &&  mb <  MB )
    {
        unsigned int indice_cha_in =  cha * RO * E1 +   mb * RO*E1*CHA  ;
        unsigned int indice_cha_out = cha * RO *reduceE1 +   mb * RO*reduceE1*CHA  ;

        unsigned int pas=0;

        for (size_t e1 = startE1; e1 <= endE1; e1+=acc) {

            unsigned int index_in = ro + e1 * RO  + indice_cha_in ;
            //unsigned int index_out = ro + ((e1-startE1+1-(e1%acc))/acc) * RO ;
            unsigned int index_out = ro + pas * RO + indice_cha_out;

            out[index_out] = in[index_in];
            pas++;
        }
    }
}



template<class T> static
void remove_encoding_no_STK( cuNDArray<T> *data_out, cuNDArray<T> *data_in , const size_t acc , const size_t startE1, const size_t endE1, const size_t reduceE1)
{

    // Setup block/grid dimensions
    int cur_device = cudaDeviceManager::Instance()->getCurrentDevice();
    int warp_size = cudaDeviceManager::Instance()->warp_size(cur_device);
    int max_blockdim = cudaDeviceManager::Instance()->max_blockdim(cur_device);
    size_t shared_mem_per_block = cudaDeviceManager::Instance()->shared_mem_per_block(cur_device);

    /*std::cout << "GPU cur_device :"<<  cur_device << std::endl;
    std::cout << "GPU  warp_size :"<<  warp_size << std::endl;
    std::cout << "GPU max_blockdim :"<<  max_blockdim << std::endl;
    std::cout << "GPU shared_mem_per_block :"<<  shared_mem_per_block << std::endl;*/

    size_t NDim = data_in->get_number_of_dimensions();

    unsigned int RO = data_in->get_size(0);
    unsigned int E1 = data_in->get_size(1);
    unsigned int CHA = data_in->get_size(2);
    unsigned int MB = data_in->get_size(3);


    dim3 blockDim(warp_size, 1);

    unsigned int blocks_RO=(unsigned int) std::ceil(RO/blockDim.x);

    dim3 gridDim(blocks_RO, CHA*MB  );

    std::cout << "blockDim :"<<  warp_size << std::endl;
    std::cout << "blocks_RO :"<<  blocks_RO << std::endl;

    // Invoke kernel
    remove_encoding_kernel_no_STK<T><<< gridDim, blockDim >>>( data_out->get_data_ptr(), data_in->get_data_ptr(),   RO,  E1,  CHA,  MB,  acc , startE1,  endE1, reduceE1);
    //rss_normalize_kernel<T><<< gridDim, blockDim >>>( in_out->get_data_ptr(), stride, number_of_batches, number_of_elements );
    //assemble_D_kernel<T><<< gridDim, blockDim >>>( data_out->get_data_ptr(), data_in->get_data_ptr(), RO, E1, N, CHA, ks*ks, halfKs );
    CHECK_FOR_CUDA_ERROR();
}









template<class T> __global__ static void
remove_encoding_kernel_with_STK( T *out,  T *in, unsigned int RO, unsigned int E1, unsigned int CHA, unsigned int MB, unsigned int STK,  size_t acc ,  size_t startE1,  size_t endE1, size_t reduceE1)
{
    //typedef typename realType<T>::Type REAL;

    unsigned int ro = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int stk = (blockIdx.y)/(CHA*MB);
    unsigned int mb = (blockIdx.y - stk*CHA*MB)/CHA;
    unsigned int cha = (blockIdx.y - stk*CHA*MB)%CHA;

    if (ro < RO && cha < CHA &&  mb <  MB &&  stk <  STK)
    {

        unsigned int indice_cha_in =  cha * RO * E1 +   mb * RO*E1*CHA  + stk * RO*E1*CHA *MB ;
        unsigned int indice_cha_out = cha * RO *reduceE1 +   mb * RO*reduceE1*CHA  + stk * RO*reduceE1*CHA *MB  ;

        unsigned int pas=0;
        for (size_t e1 = startE1; e1 <= endE1; e1+=acc) {

            unsigned int index_in = ro + e1 * RO  + indice_cha_in ;
            //unsigned int index_out = ro + ((e1-startE1+1-(e1%acc))/acc) * RO ;
            unsigned int index_out = ro + pas * RO + indice_cha_out;

            out[index_out] = in[index_in];
            pas++;
        }
    }
}







template<class T> static
void remove_encoding_with_STK( cuNDArray<T> *data_out, cuNDArray<T> *data_in , const size_t acc , const size_t startE1, const size_t endE1, const size_t reduceE1)
{

    // Setup block/grid dimensions
    int cur_device = cudaDeviceManager::Instance()->getCurrentDevice();
    int warp_size = cudaDeviceManager::Instance()->warp_size(cur_device);
    int max_blockdim = cudaDeviceManager::Instance()->max_blockdim(cur_device);
    size_t shared_mem_per_block = cudaDeviceManager::Instance()->shared_mem_per_block(cur_device);

    /*std::cout << "GPU cur_device :"<<  cur_device << std::endl;
    std::cout << "GPU  warp_size :"<<  warp_size << std::endl;
    std::cout << "GPU max_blockdim :"<<  max_blockdim << std::endl;
    std::cout << "GPU shared_mem_per_block :"<<  shared_mem_per_block << std::endl;*/

    size_t NDim = data_in->get_number_of_dimensions();

    unsigned int RO = data_in->get_size(0);
    unsigned int E1 = data_in->get_size(1);
    unsigned int CHA = data_in->get_size(2);
    unsigned int MB = data_in->get_size(3);
    unsigned int STK = data_in->get_size(4);

    dim3 blockDim(warp_size, 1);
     unsigned int blocks_RO=(unsigned int) std::ceil(RO/blockDim.x);

    dim3 gridDim(blocks_RO, CHA*MB*STK  );

    std::cout << "warp_size :"<<  warp_size << std::endl;
    std::cout << "blocks_RO :"<<  blocks_RO << std::endl;
    std::cout << "CHA*MB*STK :"<<  CHA*MB*STK << std::endl;

    // Invoke kernel
    remove_encoding_kernel_with_STK<T><<< gridDim, blockDim >>>( data_out->get_data_ptr(), data_in->get_data_ptr(),   RO,  E1,  CHA,  MB,  STK,  acc , startE1,  endE1, reduceE1);
    //rss_normalize_kernel<T><<< gridDim, blockDim >>>( in_out->get_data_ptr(), stride, number_of_batches, number_of_elements );
    //assemble_D_kernel<T><<< gridDim, blockDim >>>( data_out->get_data_ptr(), data_in->get_data_ptr(), RO, E1, N, CHA, ks*ks, halfKs );
    CHECK_FOR_CUDA_ERROR();
}






template<class T> __global__ static void
im2col_2D_kernel( T *output,  T *input,  size_t block_RO,  size_t block_E1,  size_t grappa_kSize_RO,  size_t grappa_kSize_E1)
{


    //input
    int ro=blockIdx.x  + threadIdx.x;
    int e1=blockIdx.y  + threadIdx.y;

    int RO=block_RO+ grappa_kSize_RO-1;
    int E1=block_E1+ grappa_kSize_E1-1;

    int index_offset_image=ro + e1 *RO;

    //output
    //colIdx = ke1 + kro * grappa_kSize_E1
    int colIdx = threadIdx.y + grappa_kSize_E1*threadIdx.x;

    // rowIdx = e1 + ro * blocks_E1
    int rowIdx = blockIdx.y + blockIdx.x*block_E1;

    //output[rowIdx, colIdx]
    int index_offset_kernel= rowIdx + colIdx* block_RO * block_E1;

    int maximum_size_image= RO*E1;
    int maximum_size_kernel= block_RO*block_E1*grappa_kSize_RO*grappa_kSize_E1;

    if ( index_offset_image < maximum_size_image && index_offset_kernel < maximum_size_kernel )
    {
        //(" %d %d -> %d / %d   |  %d %d -> %d /  %d  || %d  %d  ->  %d /%d  \\n",  threadIdx.x, threadIdx.y, rowIdx ,  kro*ke1 , blockIdx.x, blockIdx.y , colIdx , ro_cut *e1_cut  , kernel_x, kernel_y , index_offset_image , maximum_size_image);

        //if (blockIdx.x==123 && blockIdx.y==66)
        //{
        //printf(" %d %d -> %d / %d   |  %d %d -> %d /  %d  || %d  %d  ->  %d  /%d \\n",  threadIdx.x, threadIdx.y, rowIdx ,  kro*ke1 , blockIdx.x, blockIdx.y , colIdx , ro_cut *e1_cut  , kernel_x, kernel_y , index_offset_image , maximum_size_image);
        //}

        output[index_offset_kernel] = input[index_offset_image];
        //}

    }



}



template<class T> static
void compute_im2col_2D( cuNDArray<T> *data_out, cuNDArray<T> *data_in , const size_t blocks_RO, const size_t blocks_E1, const size_t grappa_kSize_RO, const size_t grappa_kSize_E1)
{

    // Setup block/grid dimensions
    int cur_device = cudaDeviceManager::Instance()->getCurrentDevice();
    int warp_size = cudaDeviceManager::Instance()->warp_size(cur_device);
    int max_blockdim = cudaDeviceManager::Instance()->max_blockdim(cur_device);
    size_t shared_mem_per_block = cudaDeviceManager::Instance()->shared_mem_per_block(cur_device);

    /*std::cout << "GPU cur_device :"<<  cur_device << std::endl;
    std::cout << "GPU  warp_size :"<<  warp_size << std::endl;
    std::cout << "GPU max_blockdim :"<<  max_blockdim << std::endl;
    std::cout << "GPU shared_mem_per_block :"<<  shared_mem_per_block << std::endl;*/

    size_t NDim = data_in->get_number_of_dimensions();

    unsigned int RO = data_in->get_size(0);
    unsigned int E1 = data_in->get_size(1);
    //unsigned int CHA = data_in->get_size(2);
    //unsigned int MB = data_in->get_size(3);

    dim3 blockDim(grappa_kSize_RO, grappa_kSize_E1);
    dim3 gridDim(blocks_RO, blocks_E1  );


    std::cout << "grappa_kSize_RO :"<<  grappa_kSize_RO << "grappa_kSize_E1 :"<<  grappa_kSize_E1  << std::endl;
    std::cout << "blocks_RO :"<<  blocks_RO << "blocks_E1 :"<<  blocks_E1  << std::endl;

    //std::cout << "blockDim :"<<  warp_size << std::endl;
    //std::cout << "RO%warp_size :"<<  int(RO/warp_size)+1 << std::endl;

    // Invoke kernel
    im2col_2D_kernel<T><<< gridDim, blockDim >>>( data_out->get_data_ptr(), data_in->get_data_ptr(),    blocks_RO,  blocks_E1,grappa_kSize_RO ,  grappa_kSize_E1 );
    //rss_normalize_kernel<T><<< gridDim, blockDim >>>( in_out->get_data_ptr(), stride, number_of_batches, number_of_elements );
    //assemble_D_kernel<T><<< gridDim, blockDim >>>( data_out->get_data_ptr(), data_in->get_data_ptr(), RO, E1, N, CHA, ks*ks, halfKs );
    CHECK_FOR_CUDA_ERROR();
}




template<class T> __global__ static void
im2col_5D_kernel( T *output,  T *input, size_t CHA, size_t MB,  size_t block_RO,  size_t block_E1,  size_t grappa_kSize_RO,  size_t grappa_kSize_E1)
{

    //input
    int ro=blockIdx.x  + threadIdx.x;
    int e1=blockIdx.y  + threadIdx.y;

    int stk = (blockIdx.z)/(CHA*MB);
    int mb = (blockIdx.z - stk*CHA*MB)/CHA;
    int cha = (blockIdx.z - stk*CHA*MB)%CHA;

    int RO=block_RO+ grappa_kSize_RO-1;
    int E1=block_E1+ grappa_kSize_E1-1;

    int index_offset_image=ro + e1 *RO;


    //colIdx = ke1 + kro * grappa_kSize_E1
    int colIdx = threadIdx.y + grappa_kSize_E1*threadIdx.x;

    // rowIdx = e1 + ro * blocks_E1
    int rowIdx = blockIdx.y + blockIdx.x*block_E1;

    //output[rowIdx, colIdx]
    int index_offset_kernel= rowIdx + colIdx* block_RO * block_E1;

    int maximum_size_image= RO*E1;
    int maximum_size_kernel= block_RO*block_E1*grappa_kSize_RO*grappa_kSize_E1;

    if ( index_offset_image < maximum_size_image && index_offset_kernel < maximum_size_kernel )
    {
        //(" %d %d -> %d / %d   |  %d %d -> %d /  %d  || %d  %d  ->  %d /%d  \\n",  threadIdx.x, threadIdx.y, rowIdx ,  kro*ke1 , blockIdx.x, blockIdx.y , colIdx , ro_cut *e1_cut  , kernel_x, kernel_y , index_offset_image , maximum_size_image);

        int indice_cha_in =  cha *  maximum_size_image  +   mb * maximum_size_image *CHA +   stk * maximum_size_image*CHA*MB   ;
        int indice_cha_out = cha * maximum_size_kernel +   mb * maximum_size_kernel *CHA  + stk*maximum_size_kernel *CHA*MB  ;


        output[index_offset_kernel+ indice_cha_out] = input[index_offset_image + indice_cha_in];
    }


}


template<class T> static
void compute_im2col_5D( cuNDArray<T> *data_out, cuNDArray<T> *data_in , const size_t blocks_RO, const size_t blocks_E1, const size_t grappa_kSize_RO, const size_t grappa_kSize_E1)
{

    // Setup block/grid dimensions
    int cur_device = cudaDeviceManager::Instance()->getCurrentDevice();
    int warp_size = cudaDeviceManager::Instance()->warp_size(cur_device);
    int max_blockdim = cudaDeviceManager::Instance()->max_blockdim(cur_device);
    size_t shared_mem_per_block = cudaDeviceManager::Instance()->shared_mem_per_block(cur_device);

    //std::cout << "GPU cur_device :"<<  cur_device << std::endl;
    //std::cout << "GPU  warp_size :"<<  warp_size << std::endl;
    //std::cout << "GPU max_blockdim :"<<  max_blockdim << std::endl;
    //std::cout << "GPU shared_mem_per_block :"<<  shared_mem_per_block << std::endl;

    size_t NDim = data_in->get_number_of_dimensions();

    unsigned int RO = data_in->get_size(0);
    unsigned int E1 = data_in->get_size(1);
    unsigned int CHA = data_in->get_size(2);
    unsigned int MB = data_in->get_size(3);
    unsigned int STK = data_in->get_size(4);


    dim3 blockDim(grappa_kSize_RO, grappa_kSize_E1,1);
    dim3 gridDim(blocks_RO, blocks_E1 , CHA*MB*STK );


    std::cout << "grappa_kSize_RO :"<<  grappa_kSize_RO << "grappa_kSize_E1 :"<<  grappa_kSize_E1  << std::endl;
    std::cout << "blocks_RO :"<<  blocks_RO << "blocks_E1 :"<<  blocks_E1<< "CHA*MB*STK :"<<  CHA*MB*STK   << std::endl;



    //std::cout << "blockDim :"<<  warp_size << std::endl;
    //std::cout << "RO%warp_size :"<<  int(RO/warp_size)+1 << std::endl;

    // Invoke kernel
    im2col_5D_kernel<T><<< gridDim, blockDim >>>( data_out->get_data_ptr(), data_in->get_data_ptr(), CHA, MB,    blocks_RO,  blocks_E1,grappa_kSize_RO ,  grappa_kSize_E1 );
    //rss_normalize_kernel<T><<< gridDim, blockDim >>>( in_out->get_data_ptr(), stride, number_of_batches, number_of_elements );
    //assemble_D_kernel<T><<< gridDim, blockDim >>>( data_out->get_data_ptr(), data_in->get_data_ptr(), RO, E1, N, CHA, ks*ks, halfKs );
    CHECK_FOR_CUDA_ERROR();
}





template<class T> __global__ static void
matrix_apply_EPI_optimal (T *input,  T *epi_nav_pos_STK,T *epi_nav_neg_STK , T *epi_nav_pos_STK_mean , T *epi_nav_neg_STK_mean, int *reverse_line, size_t CHA, size_t MB, size_t start_E1_, size_t end_E1_, size_t RO, size_t E1)
{

   //int ro=blockIdx.x  + threadIdx.x;
   //int e1=blockIdx.y  + threadIdx.y;

   int ro = blockIdx.x * blockDim.x + threadIdx.x;
   int e1 = blockIdx.y * blockDim.y + threadIdx.y;

   int a = (blockIdx.z)/(CHA*MB);
   int mb = (blockIdx.z - a*CHA*MB)/CHA;
   int cha = (blockIdx.z - a*CHA*MB)%CHA;

   if ( e1 >= start_E1_ && e1 <=end_E1_ && ro >=0 && ro< RO)
   {

   //printf(" %d %d  %d \\n",threadIdx.x, blockIdx.x,   ro) ;

       int index_epi_nav = ro  + mb * RO+ a * RO* MB;

       int index_epi_nav_mean = ro +  a * RO;

       int index_voxel = ro + e1*RO +  cha*RO*E1 + mb*RO*E1*CHA + a * RO*E1*CHA*MB;

       T correction_pos_hoND;
       T correction_neg_hoND;

       /*if (optimal==true)
       {
           correction_pos_hoND=epi_nav_pos_STK[index_epi_nav];
           correction_neg_hoND=epi_nav_neg_STK[index_epi_nav] ;
       }
       else
       {
           correction_pos_hoND=epi_nav_pos_STK_mean[index_epi_nav_mean];
           correction_neg_hoND=epi_nav_neg_STK_mean[index_epi_nav_mean] ;
       }


       if (undo==true)
       {
           correction_pos_hoND=epi_nav_pos_STK[index_epi_nav]/epi_nav_pos_STK_mean[index_epi_nav_mean];
           correction_neg_hoND=epi_nav_neg_STK[index_epi_nav]/epi_nav_neg_STK_mean[index_epi_nav_mean];
       }*/

       correction_pos_hoND=epi_nav_pos_STK[index_epi_nav];
       correction_neg_hoND=epi_nav_neg_STK[index_epi_nav] ;

       //correction_pos_hoND=1;
       //correction_neg_hoND=1;

       if (reverse_line[e1]==1)
       {
           input[index_voxel]*=correction_neg_hoND;
       }
       else
       {
           input[index_voxel]*=correction_pos_hoND;
       }

   }

}

template<class T> __global__ static void
matrix_apply_EPI_mean (T *input,  T *epi_nav_pos_STK,T *epi_nav_neg_STK , T *epi_nav_pos_STK_mean , T *epi_nav_neg_STK_mean, int *reverse_line, size_t CHA, size_t MB, size_t start_E1_, size_t end_E1_, size_t RO, size_t E1)
{

   //int ro=blockIdx.x  + threadIdx.x;
   //int e1=blockIdx.y  + threadIdx.y;

   int ro = blockIdx.x * blockDim.x + threadIdx.x;
   int e1 = blockIdx.y * blockDim.y + threadIdx.y;

   int a = (blockIdx.z)/(CHA*MB);
   int mb = (blockIdx.z - a*CHA*MB)/CHA;
   int cha = (blockIdx.z - a*CHA*MB)%CHA;

   if ( e1 >= start_E1_ && e1 <=end_E1_ && ro >=0 && ro< RO)
   {

   //printf(" %d %d  %d \\n",threadIdx.x, blockIdx.x,   ro) ;

       int index_epi_nav = ro  + mb * RO+ a * RO* MB;

       int index_epi_nav_mean = ro +  a * RO;

       int index_voxel = ro + e1*RO +  cha*RO*E1 + mb*RO*E1*CHA + a * RO*E1*CHA*MB;

       T correction_pos_hoND;
       T correction_neg_hoND;

       /*if (optimal==true)
       {
           correction_pos_hoND=epi_nav_pos_STK[index_epi_nav];
           correction_neg_hoND=epi_nav_neg_STK[index_epi_nav] ;
       }
       else
       {
           correction_pos_hoND=epi_nav_pos_STK_mean[index_epi_nav_mean];
           correction_neg_hoND=epi_nav_neg_STK_mean[index_epi_nav_mean] ;
       }


       if (undo==true)
       {
           correction_pos_hoND=epi_nav_pos_STK[index_epi_nav]/epi_nav_pos_STK_mean[index_epi_nav_mean];
           correction_neg_hoND=epi_nav_neg_STK[index_epi_nav]/epi_nav_neg_STK_mean[index_epi_nav_mean];
       }*/

       correction_pos_hoND=epi_nav_pos_STK_mean[index_epi_nav_mean];
       correction_neg_hoND=epi_nav_neg_STK_mean[index_epi_nav_mean] ;

       //correction_pos_hoND=1;
       //correction_neg_hoND=1;

       if (reverse_line[e1]==1)
       {
           input[index_voxel]*=correction_neg_hoND;
       }
       else
       {
           input[index_voxel]*=correction_pos_hoND;
       }

   }

}


template<class T> __global__ static void
matrix_apply_EPI_undo (T *input,  T *epi_nav_pos_STK,T *epi_nav_neg_STK , T *epi_nav_pos_STK_mean , T *epi_nav_neg_STK_mean, int *reverse_line, size_t CHA, size_t MB, size_t start_E1_, size_t end_E1_, size_t RO, size_t E1)
{

   //int ro=blockIdx.x  + threadIdx.x;
   //int e1=blockIdx.y  + threadIdx.y;

   int ro = blockIdx.x * blockDim.x + threadIdx.x;
   int e1 = blockIdx.y * blockDim.y + threadIdx.y;

   int a = (blockIdx.z)/(CHA*MB);
   int mb = (blockIdx.z - a*CHA*MB)/CHA;
   int cha = (blockIdx.z - a*CHA*MB)%CHA;

   if ( e1 >= start_E1_ && e1 <=end_E1_ && ro >=0 && ro< RO)
   {

   //printf(" %d %d  %d \\n",threadIdx.x, blockIdx.x,   ro) ;

       int index_epi_nav = ro  + mb * RO+ a * RO* MB;

       int index_epi_nav_mean = ro +  a * RO;

       int index_voxel = ro + e1*RO +  cha*RO*E1 + mb*RO*E1*CHA + a * RO*E1*CHA*MB;

       T correction_pos_hoND;
       T correction_neg_hoND;

       /*if (optimal==true)
       {
           correction_pos_hoND=epi_nav_pos_STK[index_epi_nav];
           correction_neg_hoND=epi_nav_neg_STK[index_epi_nav] ;
       }
       else
       {
           correction_pos_hoND=epi_nav_pos_STK_mean[index_epi_nav_mean];
           correction_neg_hoND=epi_nav_neg_STK_mean[index_epi_nav_mean] ;
       }


       if (undo==true)
       {
           correction_pos_hoND=epi_nav_pos_STK[index_epi_nav]/epi_nav_pos_STK_mean[index_epi_nav_mean];
           correction_neg_hoND=epi_nav_neg_STK[index_epi_nav]/epi_nav_neg_STK_mean[index_epi_nav_mean];
       }*/

       correction_pos_hoND=epi_nav_pos_STK[index_epi_nav]/epi_nav_pos_STK_mean[index_epi_nav_mean];
       correction_neg_hoND=epi_nav_neg_STK[index_epi_nav]/epi_nav_neg_STK_mean[index_epi_nav_mean];

       //correction_pos_hoND=1;
       //correction_neg_hoND=1;

       if (reverse_line[e1]==1)
       {
           input[index_voxel]*=correction_neg_hoND;
       }
       else
       {
           input[index_voxel]*=correction_pos_hoND;
       }

   }

}



template<class T> static
void compute_EPI_coor_5D( bool undo, bool optimal, cuNDArray<T> *data_in, cuNDArray<T> *pos, cuNDArray<T> *neg, cuNDArray<T> *pos_mean, cuNDArray<T> *neg_mean, cuNDArray<int> *reverse_line)
{

    // Setup block/grid dimensions
    int cur_device = cudaDeviceManager::Instance()->getCurrentDevice();
    int warp_size = cudaDeviceManager::Instance()->warp_size(cur_device);
    int max_blockdim = cudaDeviceManager::Instance()->max_blockdim(cur_device);
    size_t shared_mem_per_block = cudaDeviceManager::Instance()->shared_mem_per_block(cur_device);

    //std::cout << "GPU cur_device :"<<  cur_device << std::endl;
    //std::cout << "GPU  warp_size :"<<  warp_size << std::endl;
    //std::cout << "GPU max_blockdim :"<<  max_blockdim << std::endl;
    //std::cout << "GPU shared_mem_per_block :"<<  shared_mem_per_block << std::endl;

    size_t NDim = data_in->get_number_of_dimensions();

    unsigned int RO = data_in->get_size(0);
    unsigned int E1 = data_in->get_size(1);
    unsigned int CHA = data_in->get_size(2);
    unsigned int MB = data_in->get_size(3);
    unsigned int STK = data_in->get_size(4);

    dim3 blockDim(32, 32,1);

    unsigned int blocks_RO=(unsigned int) std::ceil(RO/blockDim.x);
    unsigned int blocks_E1=(unsigned int) std::ceil(RO/blockDim.y);

    dim3 gridDim(blocks_RO, blocks_E1 , CHA*MB*STK );


    //std::cout << "maxThreadsPerBlock :"<<  32 << "maxThreadsPerBlock :"<<  32  << std::endl;
    //std::cout << "blocks_RO :  "<<  blocks_RO << "  blocks_E1 :  "<<  blocks_E1<< "  CHA*MB*STK :  "<<  CHA*MB*STK   << std::endl;

    unsigned int start_E1_ = 50;
    unsigned int end_E1_ = 90;

    //bool* myGlobalBoolVarPtr_undo;
    //myGlobalBoolVarPtr_undo=&undo;
    //cudaMalloc(&myGlobalBoolVarPtr_undo, sizeof(bool));

    //std::cout << "blockDim :"<<  warp_size << std::endl;
    //std::cout << "RO%warp_size :"<<  int(RO/warp_size)+1 << std::endl;

    // Invoke kernel
    if (undo==true)
    {
    matrix_apply_EPI_undo<T><<< gridDim, blockDim >>>( data_in->get_data_ptr(),  pos->get_data_ptr() , neg->get_data_ptr() , pos_mean->get_data_ptr() , neg_mean->get_data_ptr(), reverse_line->get_data_ptr(),  CHA,  MB,  start_E1_,  end_E1_,  RO,  E1);
    }
    else
    {
        if (optimal==true)
        {
           matrix_apply_EPI_optimal<T><<< gridDim, blockDim >>>( data_in->get_data_ptr(),  pos->get_data_ptr() , neg->get_data_ptr() , pos_mean->get_data_ptr() , neg_mean->get_data_ptr(), reverse_line->get_data_ptr(),  CHA,  MB,  start_E1_,  end_E1_,  RO,  E1);

        }
        else
        {
            matrix_apply_EPI_mean<T><<< gridDim, blockDim >>>( data_in->get_data_ptr(),  pos->get_data_ptr() , neg->get_data_ptr() , pos_mean->get_data_ptr() , neg_mean->get_data_ptr(), reverse_line->get_data_ptr(),  CHA,  MB,  start_E1_,  end_E1_,  RO,  E1);
        }
    }
    //rss_normalize_kernel<T><<< gridDim, blockDim >>>( in_out->get_data_ptr(), stride, number_of_batches, number_of_elements );
    //assemble_D_kernel<T><<< gridDim, blockDim >>>( data_out->get_data_ptr(), data_in->get_data_ptr(), RO, E1, N, CHA, ks*ks, halfKs );
    CHECK_FOR_CUDA_ERROR();
}






template<class T> static
void compute_gpu_unmix( cuNDArray<T> *data_in, cuNDArray<T> *kernel, cuNDArray<T> * data_out)
{

    // Setup block/grid dimensions
    int cur_device = cudaDeviceManager::Instance()->getCurrentDevice();
    int warp_size = cudaDeviceManager::Instance()->warp_size(cur_device);
    int max_blockdim = cudaDeviceManager::Instance()->max_blockdim(cur_device);
    size_t shared_mem_per_block = cudaDeviceManager::Instance()->shared_mem_per_block(cur_device);

    //std::cout << "GPU cur_device :"<<  cur_device << std::endl;
    //std::cout << "GPU  warp_size :"<<  warp_size << std::endl;
    //std::cout << "GPU max_blockdim :"<<  max_blockdim << std::endl;
    //std::cout << "GPU shared_mem_per_block :"<<  shared_mem_per_block << std::endl;

    size_t NDim = data_in->get_number_of_dimensions();

    int in_dim0 = data_in->get_size(0);
    int in_dim1 = data_in->get_size(1);

    int ker_dim0 = kernel->get_size(0);
    int ker_dim1 = kernel->get_size(1);

    int out_dim0 = data_out->get_size(0);
    int out_dim1 = data_out->get_size(1);

   // std::cout << "in_dim0 :"<<  in_dim0 << "  in_dim1 :"<<  in_dim1  << std::endl;
    //std::cout << "ker_dim0 :"<<  ker_dim0 << "  ker_dim1 :"<<  ker_dim1  << std::endl;
    //std::cout << "out_dim0 :"<<  out_dim0 << "  out_dim1 :"<<  out_dim1  << std::endl;

    cublasHandle_t handle = *CUBLASContextProvider::instance()->getCublasHandle();

    cublasStatus_t stat;

    //complext<float>  alpha = complext<float>(1);
    //complext<float>  beta = complext<float>(0);



    /*cublasStatus_t cublasCgemmEx(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m,
                                  int n,
                                  int k,
                                  const cuComplex *alpha,
                                  const void      *A,
                                  cudaDataType_t  Atype,
                                  int lda,
                                  const void      *B,
                                  cudaDataType_t  Btype,
                                  int ldb,
                                  const cuComplex *beta,
                                  void            *C,
                                  cudaDataType_t  Ctype,
                                  int ldc)*/

    //m  	number of rows of matrix op(A) and C.
    //n     number of columns of matrix op(B) and C.
    //k     number of columns of op(A) and rows of op(B).

    //  in_dim0 :8308  in_dim1 :200
    //  ker_dim0 :200  ker_dim1 :8
    //  out_dim0 :8308  out_dim1 :8



    int m=in_dim0; // 8308
    int n=ker_dim1;  //8
    int k=in_dim1;  // 200

    float_complext alpha(1.0f);
    float_complext beta(0.0f);

    stat = cublasCgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        m,n,k,
                        (cuFloatComplex*) &alpha,
                        (cuFloatComplex*) data_in->get_data_ptr(), m ,
                        (cuFloatComplex*) kernel->get_data_ptr(),  k ,
                        (cuFloatComplex*) &beta,
                        (cuFloatComplex*) data_out->get_data_ptr(), m );


    if (stat != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "slicegrappa : Failed to form  product using cublas gemm" << std::endl;
      std::cerr << "---- cublas error code " << stat << std::endl;

    }







    //dim3 blockDim(32, 32,1);

    //unsigned int blocks_RO=(unsigned int) std::ceil(RO/blockDim.x);
    //unsigned int blocks_E1=(unsigned int) std::ceil(RO/blockDim.y);

    //dim3 gridDim(blocks_RO, blocks_E1 , CHA*MB*STK );


    //std::cout << "maxThreadsPerBlock :"<<  32 << "maxThreadsPerBlock :"<<  32  << std::endl;
    //std::cout << "blocks_RO :  "<<  blocks_RO << "  blocks_E1 :  "<<  blocks_E1<< "  CHA*MB*STK :  "<<  CHA*MB*STK   << std::endl;




    //rss_normalize_kernel<T><<< gridDim, blockDim >>>( in_out->get_data_ptr(), stride, number_of_batches, number_of_elements );
    //assemble_D_kernel<T><<< gridDim, blockDim >>>( data_out->get_data_ptr(), data_in->get_data_ptr(), RO, E1, N, CHA, ks*ks, halfKs );
    CHECK_FOR_CUDA_ERROR();
}



/*

  template<class REAL, unsigned int D> cuNDArray<complext<REAL> >
  estimate_b1_map( const cuNDArray<complext<REAL> >& data_in, int target_coils)
  {

    if( data_in.get_number_of_dimensions() < 2 ){
      throw std::runtime_error("estimate_b1_map:: dimensionality mismatch.");
    }

    if( data_in.get_number_of_dimensions()-1 != D ){
      throw std::runtime_error("estimate_b1_map:: dimensionality mismatch.");
    }

    int target_coils_int = 0;
    if ((target_coils <= 0) || (target_coils > data_in.get_size(D))) {
      target_coils_int = data_in.get_size(D);
    } else {
      target_coils_int = target_coils;
    }

    vector<unsigned int> image_dims, dims_to_xform;
    unsigned int pixels_per_coil = 1;

    for( unsigned int i=0; i<D; i++ ){
      image_dims.push_back(data_in.get_size(i));
      dims_to_xform.push_back(i);
      pixels_per_coil *= data_in.get_size(i);
    }

    unsigned int ncoils = data_in.get_size(D);

    // Make a copy of input data, but only the target coils
      std::vector<size_t> odims = *(data_in.get_dimensions().get());
      odims[D] = target_coils_int;
      auto data_out = cuNDArray<complext<REAL> >(&odims);

      //Now copy one coil at a time
      unsigned int elements_per_coil = data_in.get_number_of_elements()/ncoils;
      for (unsigned int i = 0; i < target_coils_int; i++) {
    cudaMemcpy(data_out.get_data_ptr()+i*elements_per_coil,
           data_in.get_data_ptr()+i*elements_per_coil,
           elements_per_coil*sizeof(complext<REAL>),
           cudaMemcpyDeviceToDevice);
      }
      ncoils = target_coils_int;

    // Normalize by the RSS of the coils
    rss_normalize( &data_out, D );

    // Now calculate the correlation matrices
    boost::shared_ptr<cuNDArray<complext<REAL> > > corrm = correlation( &data_out );

    // Smooth (onto copy of corrm)
    auto corrm_smooth = boost::make_shared<cuNDArray<complext<REAL>>>(corrm->get_dimensions());

    smooth_correlation_matrices<REAL,D>( corrm.get(), corrm_smooth.get() );
    corrm.reset();

    // Get the dominant eigenvector for each correlation matrix.
    auto csm = extract_csm<REAL>( corrm_smooth.get(), ncoils, pixels_per_coil );
    corrm_smooth.reset();

    // Set phase according to reference (coil 0)
    set_phase_reference<REAL>( &csm, ncoils, pixels_per_coil );

    return csm;
  }

  template<class T> static void find_stride( cuNDArray<T> *in, unsigned int dim,
                         unsigned int *stride, std::vector<size_t> *dims )
  {
    *stride = 1;
    for( unsigned int i=0; i<in->get_number_of_dimensions(); i++ ){
      if( i != dim )
    dims->push_back(in->get_size(i));
      if( i < dim )
    *stride *= in->get_size(i);
    }
  }
  
  template<class REAL, class T> __inline__  __device__ static REAL
  _rss( unsigned int idx, T *in, unsigned int stride, unsigned int number_of_batches )
  {
    unsigned int in_idx = (idx/stride)*stride*number_of_batches+(idx%stride);
    REAL rss = REAL(0);
    
    for( unsigned int i=0; i<number_of_batches; i++ )
      rss += norm(in[i*stride+in_idx]);

    rss = std::sqrt(rss);
    
    return rss;
  }
  
  template<class T> __global__ static void
  rss_normalize_kernel( T *in_out, unsigned int stride, unsigned int number_of_batches, unsigned int number_of_elements )
  {
    typedef typename realType<T>::Type REAL;

    const unsigned int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;
    
    if( idx < number_of_elements ){

      REAL reciprocal_rss = 1/(_rss<REAL,T>(idx, in_out, stride, number_of_batches));
      
      unsigned int in_idx = (idx/stride)*stride*number_of_batches+(idx%stride);
      
      for( unsigned int i=0; i<number_of_batches; i++ ) {
    T out = in_out[i*stride+in_idx];
    out *= reciprocal_rss; // complex-scalar multiplication (element-wise operator)
    in_out[i*stride+in_idx] = out;
      }
    }
  }
  
  // Normalized RSS
  template<class T> static
  void rss_normalize( cuNDArray<T> *in_out, unsigned int dim )
  {
    unsigned int number_of_batches = in_out->get_size(dim);
    unsigned int number_of_elements = in_out->get_number_of_elements()/number_of_batches;
    
    // Setup block/grid dimensions
    dim3 blockDim; dim3 gridDim;
    setup_grid( number_of_elements, &blockDim, &gridDim );

    // Find element stride
    unsigned int stride; std::vector<size_t> dims;
    find_stride<T>( in_out, dim, &stride, &dims );

    // Invoke kernel
    rss_normalize_kernel<T><<< gridDim, blockDim >>>( in_out->get_data_ptr(), stride, number_of_batches, number_of_elements );

    CHECK_FOR_CUDA_ERROR();
  }

  template<class REAL, class T> __global__ static void
  correlation_kernel( const T * __restrict__ in, T * __restrict__ corrm, unsigned int num_batches, unsigned int num_elements )
  {
    const unsigned int p = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int i = threadIdx.y;
    
    if( p < num_elements ){
      for( unsigned int j=0; j<i; j++){
    T tmp = in[i*num_elements+p]*conj(in[j*num_elements+p]);
    corrm[(j*num_batches+i)*num_elements+p] = tmp;
    corrm[(i*num_batches+j)*num_elements+p] = conj(tmp);
      }
      T tmp = in[i*num_elements+p];
      corrm[(i*num_batches+i)*num_elements+p] = tmp*conj(tmp);
    }
  }
  
  // Build correlation matrix
  template<class T> static boost::shared_ptr< cuNDArray<T> > correlation( cuNDArray<T> *in )
  {
    typedef typename realType<T>::Type REAL;
    // Prepare internal array
    int cur_device = cudaDeviceManager::Instance()->getCurrentDevice();

    unsigned int number_of_batches = in->get_size(in->get_number_of_dimensions()-1);
    unsigned int number_of_elements = in->get_number_of_elements()/number_of_batches;

    int warp_size = cudaDeviceManager::Instance()->warp_size(cur_device);
    int max_blockdim = cudaDeviceManager::Instance()->max_blockdim(cur_device);
    dim3 blockDim(((max_blockdim/number_of_batches)/warp_size)*warp_size, number_of_batches);

    if( blockDim.x == 0 ){
      throw std::runtime_error("correlation: correlation dimension exceeds device capacity.");
    }

    dim3 gridDim((number_of_elements+blockDim.x-1)/blockDim.x);

    // Invoke kernel
    std::vector<size_t> dims = *in->get_dimensions(); dims.push_back(number_of_batches);
    boost::shared_ptr< cuNDArray<T> > out( new cuNDArray<T> );
    out->create(&dims);

    correlation_kernel<REAL,T><<< gridDim, blockDim >>>( in->get_data_ptr(), out->get_data_ptr(), number_of_batches, number_of_elements );
    
    CHECK_FOR_CUDA_ERROR();
    
    return out;
  }

  // Smooth correlation matrices by box filter (1D)
  template<class REAL> __global__ static void
  smooth_correlation_matrices_kernel( const complext<REAL> * __restrict__ corrm, complext<REAL> * __restrict__ corrm_smooth, intd<1>::Type image_dims )
  {
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;

    const int num_image_elements = prod(image_dims);

    if( idx < num_image_elements ){
    
      const int co = idx;
      const int x = co;

      const int size_x = image_dims.vec[0];

      const REAL scale = REAL(1)/((REAL)kernel_width);

      complext<REAL> result = complext<REAL>(0);

      for (int kx = 0; kx < kernel_width; kx++) {
      
    if ((x-(kernel_width>>1)+kx) >= 0 &&
        (x-(kernel_width>>1)+kx) < size_x)
      {
        int source_offset =
          batch*num_image_elements +
          (x-(kernel_width>>1)+kx);

        result += corrm[source_offset];
      }
      }
      corrm_smooth[batch*num_image_elements+idx] = scale*result;
    }
  }

  // Smooth correlation matrices by box filter (2D)
  template<class REAL> __global__ static  void
  smooth_correlation_matrices_kernel( const complext<REAL> * __restrict__ corrm, complext<REAL> * __restrict__ corrm_smooth, intd<2>::Type image_dims )
  {
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;

    const int num_image_elements = prod(image_dims);

    if( idx < num_image_elements ){
    
      const intd2 co = idx_to_co<2>(idx, image_dims);

      const int x = co.vec[0];
      const int y = co.vec[1];

      const int size_x = image_dims.vec[0];
      const int size_y = image_dims.vec[1];

      const int half_width = kernel_width>>1;

      const int yminus = y-half_width;
      const int xminus = x-half_width;
      const int yplus = y+half_width;
      const int xplus = x+half_width;

      const REAL scale = REAL(1)/((REAL)(kernel_width*kernel_width));

      complext<REAL> result = complext<REAL>(0);

      if( (yminus >=0) ){
    if( yplus < size_y ){
      if( xminus >= 0 ){
        if( xplus < size_x ){

#pragma unroll
          for (int ky = 0; ky < kernel_width; ky++){
#pragma unroll
        for (int kx = 0; kx < kernel_width; kx++) {

          int cy = yminus+ky;
          int cx = xminus+kx;

          int source_offset = batch*num_image_elements + cy*size_x + cx;
          result += corrm[source_offset];
        }
          }
        }
      }
    }
      }
      corrm_smooth[batch*num_image_elements+idx] = scale*result;
    }
  }

  // Smooth correlation matrices by box filter (3D)
  template<class REAL> __global__ static  void
  smooth_correlation_matrices_kernel( const  complext<REAL> * __restrict__ corrm, complext<REAL> * __restrict__ corrm_smooth, intd<3>::Type image_dims )
  {
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;

    const int num_image_elements = prod(image_dims);

    if( idx < num_image_elements ){
    
      const intd3 co = idx_to_co<3>(idx, image_dims);

      const int x = co.vec[0];
      const int y = co.vec[1];
      const int z = co.vec[2];

      const int size_x = image_dims.vec[0];
      const int size_y = image_dims.vec[1];
      const int size_z = image_dims.vec[2];

      const REAL scale = REAL(1)/((REAL)(kernel_width*kernel_width*kernel_width));

      complext<REAL> result = complext<REAL>(0);

      for (int kz = 0; kz < kernel_width; kz++) {
    for (int ky = 0; ky < kernel_width; ky++) {
      for (int kx = 0; kx < kernel_width; kx++) {

        if ((z-(kernel_width>>1)+kz) >= 0 &&
        (z-(kernel_width>>1)+kz) < size_z &&
        (y-(kernel_width>>1)+ky) >= 0 &&
        (y-(kernel_width>>1)+ky) < size_y &&
        (x-(kernel_width>>1)+kx) >= 0 &&
        (x-(kernel_width>>1)+kx) < size_x)
          {
        int source_offset =
          batch*num_image_elements +
          (z-(kernel_width>>1)+kz)*size_x*size_y +
          (y-(kernel_width>>1)+ky)*size_x +
          (x-(kernel_width>>1)+kx);

        result += corrm[source_offset];
          }
      }
    }
      }
      corrm_smooth[batch*num_image_elements+idx] = scale*result;
    }
  }

  // Smooth correlation matrices by box filter (3D)
  template<class REAL> __global__ static void
  smooth_correlation_matrices_kernel( const complext<REAL> * __restrict__ corrm, complext<REAL> * __restrict__ corrm_smooth, intd<4>::Type image_dims )
  {
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;

    const int num_image_elements = prod(image_dims);

    if( idx < num_image_elements ){
    
      const intd4 co = idx_to_co<4>(idx, image_dims);

      const int x = co.vec[0];
      const int y = co.vec[1];
      const int z = co.vec[2];
      const int w = co.vec[3];

      const int size_x = image_dims.vec[0];
      const int size_y = image_dims.vec[1];
      const int size_z = image_dims.vec[2];
      const int size_w = image_dims.vec[3];

      const REAL scale = REAL(1)/((REAL)(kernel_width*kernel_width*kernel_width*kernel_width));

      complext<REAL> result = complext<REAL>(0);

      for (int kw = 0; kw < kernel_width; kw++) {
    for (int kz = 0; kz < kernel_width; kz++) {
      for (int ky = 0; ky < kernel_width; ky++) {
        for (int kx = 0; kx < kernel_width; kx++) {

          if ((w-(kernel_width>>1)+kw) >= 0 &&
          (w-(kernel_width>>1)+kw) < size_w &&
          (z-(kernel_width>>1)+kz) >= 0 &&
          (z-(kernel_width>>1)+kz) < size_z &&
          (y-(kernel_width>>1)+ky) >= 0 &&
          (y-(kernel_width>>1)+ky) < size_y &&
          (x-(kernel_width>>1)+kx) >= 0 &&
          (x-(kernel_width>>1)+kx) < size_x)
        {
          int source_offset =
            batch*num_image_elements +
            (w-(kernel_width>>1)+kw)*size_x*size_y*size_z +
            (z-(kernel_width>>1)+kz)*size_x*size_y +
            (y-(kernel_width>>1)+ky)*size_x +
            (x-(kernel_width>>1)+kx);

          result += corrm[source_offset];
        }
        }
      }
    }
      }
      corrm_smooth[batch*num_image_elements+idx] = scale*result;
    }
  }

  __device__ int _min( int A, int B ){
    return (A<B) ? A : B;
  }

  // Smooth correlation matrices border by box filter (2D)
  template<class REAL> __global__ static void
  smooth_correlation_matrices_border_kernel( const complext<REAL> * __restrict__ corrm, complext<REAL> * __restrict__ corrm_smooth, intd<2>::Type image_dims, unsigned int number_of_border_threads )
  {
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;

    const int num_image_elements = prod(image_dims);

    if( idx < number_of_border_threads ){
    
      intd2 co;
      const int half_width = kernel_width>>1;

      co.vec[1] = idx/image_dims.vec[0];
      co.vec[1] = _min(co.vec[1], half_width );

      if( co.vec[1] == half_width ){
    int new_idx = idx-half_width*image_dims.vec[0];
    int num_skips = new_idx/half_width;
    int rows_offset = _min(num_skips>>1, image_dims.vec[1]-(half_width<<1) );
    co.vec[1] += rows_offset;

    if( co.vec[1] == (half_width + image_dims.vec[1]-(half_width<<1)) ){
      new_idx -= ((image_dims.vec[1]-(half_width<<1))*(half_width<<1));
      co.vec[1] += (new_idx / image_dims.vec[0]);
      co.vec[0] = (new_idx % image_dims.vec[0]);
    }
    else{
      co.vec[0] = (num_skips%2)*(image_dims.vec[0]-half_width) + (new_idx%half_width);
    }
      }
      else{
    co.vec[0] = idx%image_dims.vec[0];
      }

      const int x = co.vec[0];
      const int y = co.vec[1];

      const int size_x = image_dims.vec[0];
      const int size_y = image_dims.vec[1];

      const int yminus = y-half_width;
      const int xminus = x-half_width;

      const REAL scale = REAL(1)/((REAL)(kernel_width*kernel_width));

      complext<REAL> result = complext<REAL>(0);

#pragma unroll
      for (int ky = 0; ky < kernel_width; ky++) {
#pragma unroll
    for (int kx = 0; kx < kernel_width; kx++) {

      if( (yminus+ky >=0) ){
        if( yminus+ky < size_y ){
          if( xminus+kx >= 0 ){
        if( xminus+kx < size_x ){

          int source_offset =
            batch*num_image_elements +
            (yminus+ky)*size_x +
            (xminus+kx);

          result += corrm[source_offset];
        }
          }
        }
      }
    }
      }
      corrm_smooth[batch*num_image_elements+co_to_idx<2>(co,image_dims)] = scale*result;
    }
  }

  template<class REAL, unsigned int D> static void
  smooth_correlation_matrices( cuNDArray<complext<REAL> > * corrm, cuNDArray<complext<REAL> > * corrm_smooth )
  {
    typename intd<D>::Type image_dims;

    for( unsigned int i=0; i<D; i++ ){
      image_dims.vec[i] = corrm->get_size(i);
    }

    unsigned int number_of_batches = 1;

    for( unsigned int i=D; i<corrm->get_number_of_dimensions(); i++ ){
      number_of_batches *= corrm->get_size(i);
    }

    int device; cudaGetDevice( &device );
    cudaDeviceProp deviceProp; cudaGetDeviceProperties( &deviceProp, device );

    dim3 blockDim(deviceProp.maxThreadsPerBlock);
    dim3 gridDim((unsigned int) std::ceil((double)prod(image_dims)/blockDim.x), number_of_batches);

    smooth_correlation_matrices_kernel<REAL><<<gridDim, blockDim>>>
      ( corrm->get_data_ptr(), corrm_smooth->get_data_ptr(), image_dims );

    CHECK_FOR_CUDA_ERROR();

    unsigned int number_of_border_threads = ((kernel_width>>1)<<1)*(sum(image_dims)-((kernel_width>>1)<<1));
    blockDim = dim3(128);
    gridDim = dim3((unsigned int) std::ceil((double)number_of_border_threads/blockDim.x), number_of_batches);

    smooth_correlation_matrices_border_kernel<REAL><<<gridDim, blockDim>>>
      ( corrm->get_data_ptr(), corrm_smooth->get_data_ptr(), image_dims, number_of_border_threads );

    CHECK_FOR_CUDA_ERROR();
  }

  extern __shared__ char shared_mem[];

  // Extract CSM
  template<class REAL> __global__ static void
  extract_csm_kernel( const complext<REAL> * __restrict__ corrm, complext<REAL> * __restrict__ csm, unsigned int num_batches, unsigned int num_elements )
  {
    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int i = threadIdx.x;

    if( idx < num_elements ){
    
      // Get the dominant eigenvector for each correlation matrix.
      // Copying Peter Kellman's approach we use the power method:
      //  b_k+1 = A*b_k / ||A*b_k||

      complext<REAL> *data_out = (complext<REAL>*) shared_mem;
      complext<REAL> *tmp_v = &(((complext<REAL>*) shared_mem)[num_batches*blockDim.x]);

      const unsigned int iterations = 2;

      for( unsigned int c=0; c<num_batches; c++){
    data_out[c*blockDim.x+i] = complext<REAL>(1);
      }

      for( unsigned int it=0; it<iterations; it++ ){
      
    for( unsigned int c=0; c<num_batches; c++){
      tmp_v[c*blockDim.x+i] = complext<REAL>(0);
    }

    for( unsigned j=0; j<num_batches; j++){
      for( unsigned int k=0; k<num_batches; k++){
        tmp_v[j*blockDim.x+i] += corrm[(k*num_batches+j)*num_elements+idx]*data_out[k*blockDim.x+i];
      }
    }

    REAL tmp = REAL(0);

    for (unsigned int c=0; c<num_batches; c++){
      tmp += norm(tmp_v[c*blockDim.x+i]);
    }

    tmp = 1/std::sqrt(tmp);


    for (unsigned int c=0; c<num_batches; c++){
      complext<REAL> res = tmp*tmp_v[c*blockDim.x+i];
      data_out[c*blockDim.x+i] = res;
    }
      }

      for (unsigned int c=0; c<num_batches; c++){
    csm[c*num_elements+idx] = data_out[c*blockDim.x+i];
      }
    }
  }

  // Extract CSM
  template<class REAL> __global__ static void
  extract_csm_kernel( const complext<REAL> * __restrict__ corrm, complext<REAL> * __restrict__ csm, unsigned int num_batches, unsigned int num_elements, complext<REAL> * __restrict__ tmp_v )
  {
    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if( idx < num_elements ){
    
      // Get the dominant eigenvector for each correlation matrix.
      // Copying Peter Kellman's approach we use the power method:
      //  b_k+1 = A*b_k / ||A*b_k||

      const unsigned int iterations = 2;

      for( unsigned int c=0; c<num_batches; c++){
    csm[c*num_elements+idx] = complext<REAL>(1);
      }

      for( unsigned int it=0; it<iterations; it++ ){

    for( unsigned int c=0; c<num_batches; c++){
      tmp_v[c*num_elements+idx] = complext<REAL>(0);
    }

    for( unsigned j=0; j<num_batches; j++){
      for( unsigned int k=0; k<num_batches; k++){
        typedef complext<REAL> T;
        tmp_v[j*num_elements+idx] += corrm[(k*num_batches+j)*num_elements+idx]*csm[k*num_elements+idx];
      }
    }

    REAL tmp = REAL(0);

    for (unsigned int c=0; c<num_batches; c++){
      tmp += norm(tmp_v[c*num_elements+idx]);
    }

    tmp = 1/std::sqrt(tmp);


    for (unsigned int c=0; c<num_batches; c++){
      complext<REAL> res = tmp*tmp_v[c*num_elements+idx];
      csm[c*num_elements+idx] = res;
    }
      }
    }
  }

  // Extract CSM
  template<class REAL> __host__ static
  cuNDArray<complext<REAL>> extract_csm(const cuNDArray<complext<REAL> >& corrm_in, unsigned int number_of_batches, unsigned int number_of_elements )
  {
    vector<size_t> image_dims;

    for( unsigned int i=0; i<corrm_in.get_number_of_dimensions()-1; i++ ){
      image_dims.push_back(corrm_in.get_size(i));
    }

    // Allocate output
    cuNDArray<complext<REAL> > out = cuNDArray<complext<REAL> >(&image_dims);

    dim3 blockDim(256);
    dim3 gridDim((unsigned int) std::ceil((double)number_of_elements/blockDim.x));

    cuNDArray<complext<REAL> > tmp_v = cuNDArray<complext<REAL> >(&image_dims);

      extract_csm_kernel<REAL><<< gridDim, blockDim >>>
    ( corrm_in.get_data_ptr(), out.get_data_ptr(), number_of_batches, number_of_elements, tmp_v.get_data_ptr() );

    CHECK_FOR_CUDA_ERROR();

    return out;
  }

  // Set refence phase
  template<class REAL> __global__ static void
  set_phase_reference_kernel( complext<REAL> *csm, unsigned int num_batches, unsigned int num_elements )
  {
    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if( idx < num_elements ){
      REAL angle = arg<REAL>(csm[idx]); //Phase of the first coil
      REAL sin_a, cos_a; gad_sincos( angle, &sin_a, &cos_a );

      complext<REAL> tmp;
      tmp._real = cos_a; tmp._imag = sin_a;
      tmp = conj(tmp);

      for( unsigned int c=0; c<num_batches; c++ ){
    complext<REAL> val = csm[c*num_elements+idx];
    typedef complext<REAL> T;
    val = val*tmp;
    csm[c*num_elements+idx] = val;
      }
    }
  }
  
  // Set reference phase
  template<class REAL> __host__ static
  void set_phase_reference(cuNDArray<complext<REAL> > *csm, unsigned int number_of_batches, unsigned int number_of_elements )
  {
    dim3 blockDim(128);
    dim3 gridDim((unsigned int) std::ceil((double)number_of_elements/blockDim.x));

    set_phase_reference_kernel<REAL><<< gridDim, blockDim >>>( csm->get_data_ptr(), number_of_batches, number_of_elements );

    CHECK_FOR_CUDA_ERROR();
  }*/



//
// Template instantiation
//

//template EXPORTGPUPMRI boost::shared_ptr< cuNDArray<complext<float> > > estimate_b1_map<float,1>(cuNDArray<complext<float> >*, int);
//template EXPORTGPUPMRI  cuNDArray<complext<float>> estimate_b1_map<float,2>(const cuNDArray<complext<float> >&, int);
//template boost::shared_ptr< cuNDArray<complext<float> > > estimate_b1_map<float,3>(cuNDArray<complext<float> >*, int);
//template boost::shared_ptr< cuNDArray<complext<float> > > estimate_b1_map<float,4>(cuNDArray<complext<float> >*, int);

//template EXPORTGPUPMRI boost::shared_ptr< cuNDArray<complext<double> > > estimate_b1_map<double,1>(cuNDArray<complext<double> >*, int);
//emplate EXPORTGPUPMRI cuNDArray<complext<double>> estimate_b1_map<double,2>(const cuNDArray<complext<double>>&, int);
//template EXPORTGPUPMRI boost::shared_ptr< cuNDArray<complext<double> > > estimate_b1_map<double,3>(cuNDArray<complext<double> >*, int);
//template EXPORTGPUPMRI boost::shared_ptr< cuNDArray<complext<double> > > estimate_b1_map<double,4>(cuNDArray<complext<double> >*, int);
}


/** \file   mri_core_slice_grappa.cpp
    \brief  GRAPPA implementation for 2D and 3D MRI parallel imaging
    \author Hui Xue

    References to the implementation can be found in:

    Griswold MA, Jakob PM, Heidemann RM, Nittka M, Jellus V, Wang J, Kiefer B, Haase A.
    Generalized autocalibrating partially parallel acquisitions (GRAPPA).
    Magnetic Resonance in Medicine 2002;47(6):1202-1210.

    Kellman P, Epstein FH, McVeigh ER.
    Adaptive sensitivity encoding incorporating temporal filtering (TSENSE).
    Magnetic Resonance in Medicine 2001;45(5):846-852.

    Breuer FA, Kellman P, Griswold MA, Jakob PM. .
    Dynamic autocalibrated parallel imaging using temporal GRAPPA (TGRAPPA).
    Magnetic Resonance in Medicine 2005;53(4):981-985.

    Saybasili H., Kellman P., Griswold MA., Derbyshire JA. Guttman, MA.
    HTGRAPPA: Real-time B1-weighted image domain TGRAPPA reconstruction.
    Magnetic Resonance in Medicine 2009;61(6): 1425-1433.
*/

#include "mri_core_slice_grappa.h"
#include "mri_core_utility.h"
#include "hoMatrix.h"
#include "hoNDArray_linalg.h"
#include "hoNDFFT.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"
#include "hoNDArray_reductions.h"

#ifdef USE_OMP
#include "omp.h"
#endif // USE_OMP

namespace Gadgetron
{





template <typename T> void im2col(hoNDArray<T>& input, hoNDArray<T>& output, const size_t blocks_RO, const size_t blocks_E1, const size_t grappa_kSize_RO, const size_t grappa_kSize_E1 )
{


    try
    {

        ///------------------------------------------------------------------------
        /// FR on recoit l'image folded pour une coupe uniquement [E1 Readout Coil ]
        /// UK
        ///

        size_t RO = input.get_size(0);
        size_t E1 = input.get_size(1);
        size_t CHA = input.get_size(2);
        size_t MB = input.get_size(3);
        size_t STK = input.get_size(4);
        size_t N = input.get_size(5);
        size_t S = input.get_size(6);

        long long rowIdx, colIdx;

        long long i,j,xx,yy;
        // input ici vaut : e1, readout, cha

        //long long num = MB * STK * N * S;
        //long long ii;
        // TODO #pragma omp parallel for default() private() shared()
        for (long long s = 0; s < S; s++)    {

            for (long long n = 0; n < N; n++)  {

                for (long long a = 0; a < STK; a++) {

                    for (long long m = 0; m < MB; m++) {

                        for (long long cha = 0; cha < CHA; cha++)
                        {
                            for ( j = 0; j < blocks_RO; j++)
                            {
                                for (i = 0; i < blocks_E1; i++)
                                {
                                    rowIdx = i + j*blocks_E1;

                                    for ( yy = 0; yy < grappa_kSize_RO; yy++)
                                    {
                                        for (xx = 0; xx < grappa_kSize_E1; xx++)
                                        {
                                            colIdx = xx + yy*grappa_kSize_E1;

                                            output(rowIdx, colIdx, cha, m, a, n,s)=input(j+yy , i+xx, cha, m, a, n, s);  //[128 63 12 2 4 1 1 1]

                                            //if (c==0 && a==0 && m==0 && xx==3 && yy==3)
                                            //{
                                            //  std::cout<<   i <<  " " <<  j << " "<<  rowIdx  << " " <<   xx <<  " " <<  yy<< " "<<  colIdx <<   " "<< abs(block_SB(rowIdx, colIdx, c, m, a, n,s)) <<std::endl;
                                            //}

                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    }
    catch(...)
    {
        GADGET_THROW("Errors in im2col(...) ... ");
    }

    return;
}




template <typename T> void im2col_open(hoNDArray<T>& input, hoNDArray<T>& output, const size_t blocks_RO, const size_t blocks_E1, const size_t grappa_kSize_RO, const size_t grappa_kSize_E1 )
{


    try
    {

        ///------------------------------------------------------------------------
        /// FR on recoit l'image folded pour une coupe uniquement [E1 Readout Coil ]
        /// UK
        ///

        size_t RO = input.get_size(0);
        size_t E1 = input.get_size(1);
        size_t CHA = input.get_size(2);
        size_t MB = input.get_size(3);
        size_t STK = input.get_size(4);
        size_t N = input.get_size(5);
        size_t S = input.get_size(6);

        //size_t rowIdx, colIdx;

        //size_t i,j,xx,yy;
        // input ici vaut : e1, readout, cha

        long long num = STK * CHA * N * S * MB;
        long long ii;

#pragma omp parallel for default(none) private(ii) shared(num, MB , STK, S, N,  input, output ,RO,E1,CHA) if(num>1)
        for (ii = 0; ii < num; ii++) {

            long long   a = ii / (S * MB* N * CHA);
            long long   m = (ii - a * S * MB* N* CHA) / (S* N * CHA );
            long long   s = (ii - a * S * MB* N* CHA - m * S* N * CHA)  /  (N * CHA);
            long long   cha=  (ii - a * S * MB* N * CHA - m * S* N * CHA -s *  N) / (N);
            long long   n = ii - a * S * MB* N * CHA - m * S* N * CHA -s *  N * CHA - cha;

            for (long long j = 0; j < blocks_RO; j++)
            {
                for (long long i = 0; i < blocks_E1; i++)
                {
                    long long rowIdx = i + j*blocks_E1;

                    for (long long yy = 0; yy < grappa_kSize_RO; yy++)
                    {
                        for (long long xx = 0; xx < grappa_kSize_E1; xx++)
                        {
                            long long colIdx = xx + yy*grappa_kSize_E1;

                            output(rowIdx, colIdx, cha, m, a, n,s)=input(j+yy , i+xx, cha, m, a, n, s);  //[128 63 12 2 4 1 1 1]

                            //if (c==0 && a==0 && m==0 && xx==3 && yy==3)
                            //{
                            //  std::cout<<   i <<  " " <<  j << " "<<  rowIdx  << " " <<   xx <<  " " <<  yy<< " "<<  colIdx <<   " "<< abs(block_SB(rowIdx, colIdx, c, m, a, n,s)) <<std::endl;
                            //}

                        }
                    }
                    // }
                    //}
                    // }
                    // }
                }
            }
        }

    }
    catch(...)
    {
        GADGET_THROW("Errors in im2col(...) ... ");
    }

    return;
}


template EXPORTMRICORE void im2col(hoNDArray< std::complex<float> >& input, hoNDArray< std::complex<float> >& block_SB, const size_t blocks_RO, const size_t blocks_E1, const size_t grappa_kSize_RO, const size_t grappa_kSize_E1);
template EXPORTMRICORE void im2col(hoNDArray< std::complex<double> >& input, hoNDArray< std::complex<double> >& block_SB, const size_t blocks_RO, const size_t blocks_E1, const size_t grappa_kSize_RO, const size_t grappa_kSize_E1);

template EXPORTMRICORE void im2col_open(hoNDArray< std::complex<float> >& input, hoNDArray< std::complex<float> >& block_SB, const size_t blocks_RO, const size_t blocks_E1, const size_t grappa_kSize_RO, const size_t grappa_kSize_E1);
template EXPORTMRICORE void im2col_open(hoNDArray< std::complex<double> >& input, hoNDArray< std::complex<double> >& block_SB, const size_t blocks_RO, const size_t blocks_E1, const size_t grappa_kSize_RO, const size_t grappa_kSize_E1);



template <typename T> void remove_unnecessary_kspace(hoNDArray<T>& input, hoNDArray<T>& output, const size_t acc, const size_t startE1, const size_t endE1, bool is_mb )
{

    size_t RO = input.get_size(0);
    size_t E1 = input.get_size(1);
    size_t E2 = input.get_size(2);
    size_t CHA = input.get_size(3);
    size_t MB = input.get_size(4);
    size_t STK = input.get_size(5);
    size_t N = input.get_size(6);
    size_t S = input.get_size(7);

    long long index;

    //std::cout << "end_E1_ " << start_E1_<<  " end_E1_ " <<  end_E1_<< "acc "   << acc << std::endl;

    if (is_mb==true)
    {
        MB=1;
    }

    // parallelisable sur cha m a n s
    for (long long s = 0; s < S; s++)
    {
        for (long long n = 0; n < N; n++)
        {
            for (long long a = 0; a < STK; a++) {

                for (long long m = 0; m < MB; m++) {

                    for (long long cha = 0; cha < CHA; cha++) {

                        index=0;

                        for (long long e1 = startE1; e1 <= endE1; e1+=acc) {

                            T* in = &(input(0, e1, 0, cha, m, a, n, s));
                            T* out = &(output(0, index, cha, m, a, n, s));

                            memcpy(out , in, sizeof(T)*RO);
                            //if (m==0 && cha==0 && a==0) { std::cout << " e1 " << e1 <<  " index "<< index <<" e1 +2 " << e1 +2 <<  std::endl;}
                            index++;
                        }
                    }
                }
            }
        }
    }

    //std::cout << "reduced_E1_"<< reduced_E1_<< "  index+1 "<< index+1 << std::endl;
    //GADGET_CHECK_THROW(reduced_E1_ == index);
}



template <typename T> void remove_unnecessary_kspace_open(hoNDArray<T>& input, hoNDArray<T>& output, const size_t acc, const size_t startE1, const size_t endE1, bool is_mb )
{

    size_t RO = input.get_size(0);
    size_t E1 = input.get_size(1);
    size_t E2 = input.get_size(2);
    size_t CHA = input.get_size(3);
    size_t MB = input.get_size(4);
    size_t STK = input.get_size(5);
    size_t N = input.get_size(6);
    size_t S = input.get_size(7);

    //std::cout << "end_E1_ " << start_E1_<<  " end_E1_ " <<  end_E1_<< "acc "   << acc << std::endl;

    if (is_mb==true)
    {
        MB=1;
    }

    long long num = STK * CHA * N * S * MB;
    long long ii;

#pragma omp parallel for default(none) private(ii) shared(num, MB , STK, S, N,  input, output ,RO,E1,E2,CHA) if(num>1)
    for (ii = 0; ii < num; ii++) {

        long long   a = ii / (S * MB* N * CHA);
        long long   m = (ii - a * S * MB* N* CHA) / (S* N * CHA );
        long long   s = (ii - a * S * MB* N* CHA - m * S* N * CHA)  /  (N * CHA);
        long long   cha=  (ii - a * S * MB* N * CHA - m * S* N * CHA -s *  N) / (N);
        long long   n = ii - a * S * MB* N * CHA - m * S* N * CHA -s *  N * CHA - cha;

        // parallelisable sur cha m a n s
        // for (size_t s = 0; s < S; s++)  {
        //for (size_t n = 0; n < N; n++)    {
        //for (size_t a = 0; a < STK; a++) {
        //for (size_t m = 0; m < MB; m++) {
        //for (size_t cha = 0; cha < CHA; cha++) {

        size_t index=0;

        for (long long e1 = startE1; e1 <= endE1; e1+=acc) {

            T* in = &(input(0, e1, 0, cha, m, a, n, s));
            T* out = &(output(0, index, cha, m, a, n, s));

            memcpy(out , in, sizeof(T)*RO);
            //if (m==0 && cha==0 && a==0) { std::cout << " e1 " << e1 <<  " index "<< index <<" e1 +2 " << e1 +2 <<  std::endl;}
            index++;

            // }
            // }
            // }
            // }
        }
    }

    //std::cout << "reduced_E1_"<< reduced_E1_<< "  index+1 "<< index+1 << std::endl;
    //GADGET_CHECK_THROW(reduced_E1_ == index);
}

template EXPORTMRICORE void remove_unnecessary_kspace(hoNDArray< std::complex<float> >& input, hoNDArray<std::complex<float> >& output, const size_t acc , const size_t startE1, const size_t endE1, bool is_mb );
template EXPORTMRICORE void remove_unnecessary_kspace(hoNDArray< std::complex<double> >& input, hoNDArray<std::complex<double> >& output, const size_t acc , const size_t startE1, const size_t endE1, bool is_mb );

template EXPORTMRICORE void remove_unnecessary_kspace_open(hoNDArray< std::complex<float> >& input, hoNDArray<std::complex<float> >& output, const size_t acc , const size_t startE1, const size_t endE1, bool is_mb );
template EXPORTMRICORE void remove_unnecessary_kspace_open(hoNDArray< std::complex<double> >& input, hoNDArray<std::complex<double> >& output, const size_t acc , const size_t startE1, const size_t endE1, bool is_mb );






/*
void GenericReconCartesianSliceGrappav3Gadget::remove_unnecessary_kspace_mb2(hoNDArray< std::complex<float> >& input, hoNDArray< std::complex<float> >& output, size_t acc )
{

    size_t RO = input.get_size(0);
    size_t E1 = input.get_size(1);
    size_t E2 = input.get_size(2);
    size_t CHA = input.get_size(3);
    size_t MB = input.get_size(4);
    size_t STK = input.get_size(5);
    size_t N = input.get_size(6);
    size_t S = input.get_size(7);

    size_t index;

    size_t s,n,a,m,cha,e2,e1;

    //std::cout << "end_E1_ " << start_E1_<<  " end_E1_ " <<  end_E1_<< "acc "   << acc << std::endl;

    for (s = 0; s < S; s++)
    {
        for (n = 0; n < N; n++)
        {
            for (a = 0; a < STK; a++) {

                //TODO ici c'est des data mb donc la dimension est = à 1
                for (m = 0; m < 1; m++) {

                    for (cha = 0; cha < CHA; cha++) {

                        index=0;

                        for (e1 = start_E1_; e1 <= end_E1_; e1+=acc) {

                            std::complex<float> * in = &(input(0, e1, 0, cha, m, a, n, s));
                            std::complex<float> * out = &(output(0, index, cha, m, a, n, s));

                            memcpy(out , in, sizeof(std::complex<float>)*RO);

                            index++;
                        }
                    }
                }
            }
        }
    }

    //std::cout << "reduced_E1_"<< reduced_E1_<< "  index+1 "<< index+1 << std::endl;
    GADGET_CHECK_THROW(reduced_E1_ == index);

}*/




template <typename T> void extract_milieu_kernel(hoNDArray< T >& block_SB, hoNDArray< T >& missing_data, const size_t kernel_size, const size_t voxels_number_per_image)
{

    ///------------------------------------------------------------------------
    /// FR on recoit l'image folded pour une coupe uniquement [E1 Readout Coil ]
    /// UK
    ///

    size_t CHA = block_SB.get_size(2);
    size_t MB = block_SB.get_size(3);
    size_t STK = block_SB.get_size(4);
    size_t N = block_SB.get_size(5);
    size_t S = block_SB.get_size(6);

    long long c,m,a,n,s;
    long long p;

    long long milieu=round(float(kernel_size)/2)-1;

    //GDEBUG("Le milieu du kernel est %d  \n",milieu );

    // TODO #pragma omp parallel for default() private() shared()
    for (s = 0; s < S; s++)    {

        for (n = 0; n < N; n++)  {

            for (a = 0; a < STK; a++) {

                for (m = 0; m < MB; m++) {

                    for ( c = 0; c < CHA; c++)
                    {
                        for ( p = 0; p < voxels_number_per_image; p++)
                        {
                            missing_data(p,c,m,a,n,s)=block_SB(p, milieu, c, m, a, n, s);

                            //[7316 25 12 2 4 1 1 1]
                            //[7316 12 2 4 1 1 1 1]
                        }
                    }
                }
            }
        }
    }
}

template EXPORTMRICORE void extract_milieu_kernel(hoNDArray< std::complex<float> >& block_SB, hoNDArray< std::complex<float> >& missing_data, const size_t kernel_size, const size_t voxels_number_per_image);
template EXPORTMRICORE void extract_milieu_kernel(hoNDArray< std::complex<double> >& block_SB, hoNDArray< std::complex<double> >& missing_data, const size_t kernel_size, const size_t voxels_number_per_image);

template <typename T>
void apply_unmix_coeff_kspace_SMS(hoNDArray<T>& in, hoNDArray<T>& kernel, hoNDArray<T>& out)
{


    try
    {

        //size_t MB_factor = kernel.get_size(2);

        /*      size_t X1 = in.get_size(0);
    size_t X2 = in.get_size(1);
    size_t X3 = in.get_size(2);
    size_t X4 = in.get_size(3);

    std::cout << " X1 " << X1 << " X2 "<< X2 << " X3 "<< X3 << " X4 "<< X4 << std::endl;

    size_t kX1 = kernel.get_size(0);
    size_t kX2 = kernel.get_size(1);
    size_t kX3 = kernel.get_size(2);
    size_t kX4 = kernel.get_size(3);

    std::cout << " kX1 " << kX1 << " kX2 "<< kX2 << " kX3 "<< kX3 << " kX4 "<< kX4 << std::endl;

    size_t oX1 = out.get_size(0);
    size_t oX2 = out.get_size(1);
    size_t oX3 = out.get_size(2);
    size_t oX4 = out.get_size(3);

    std::cout << " oX1 " << oX1 << " oX2 "<< oX2 << " oX3 "<< oX3 << " oX4 "<< oX4 << std::endl;
*/
        //hoNDArray<T> tempo_kernel(X1,X2);


        //Gadgetron::clear(tempo_kspace_unfolded);

        Gadgetron::gemm(out, in, false, kernel, false);

        //std::complex<float> * image_in = &(tempo_kspace_unfolded(0, 0));
        //std::complex<float> * image_out = &(recon_obj.unfolded_image_(0, 0, m, a, n, s));

        //memcpy(image_out , image_in, sizeof(std::complex<float>)*voxels_number_per_image_*CHA);
    }
    catch (...)
    {
        GADGET_THROW("Errors in apply_unmix_coeff_kspace_SMS(...) ... ");
    }

}


template EXPORTMRICORE void apply_unmix_coeff_kspace_SMS(hoNDArray<std::complex<float>>& in, hoNDArray<std::complex<float>>& kernel, hoNDArray<std::complex<float>>& out);



//sur les données single band
template <typename T>
void create_stacks_of_slices_directly_sb(hoNDArray< T >& data, hoNDArray< T >& new_stack , std::vector<unsigned int> &indice, std::vector< std::vector<unsigned int> > &MapSliceSMS)
{

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t N=data.get_size(4);
    size_t S=data.get_size(5);

    size_t MB=new_stack.get_size(4);
    size_t STK=new_stack.get_size(5);

    long long n, s, a, m;
    long long index;

    // copy of the data in the 8D array

    // [RO,E1,E2,CHA, N,S, SLC]
    //    ->
    // [RO,E1,E2,CHA, N,S, M, STK] //8D
    //    ->
    // [RO,E1,E2,CHA, M, STK, N,S] // 8D reorder

    for (a = 0; a < STK; a++) {

        for (m = 0; m < MB; m++) {

            index = MapSliceSMS[a][m];

            for (s = 0; s < S; s++)
            {
                for (n = 0; n < N; n++)
                {
                    std::complex<float> * in = &(data(0, 0, 0, 0, n, s, indice[index]));
                    std::complex<float> * out = &(new_stack(0, 0, 0, 0, m, a, n, s));
                    memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);
                    // using namespace Gadgetron::Indexing;
                    // std::copy_n(in,RO*E1*E2*CHA,out); in 4.1
                    // new_stack(slice,slice,slice,slice,m,a,n,s) = data(slice,slice,slice,n,s,indice[index]); in 4.1
                }
            }
        }
    }
}

template EXPORTMRICORE void create_stacks_of_slices_directly_sb(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> >& new_stack , std::vector<unsigned int> &indice, std::vector< std::vector<unsigned int> > &MapSliceSMS);

//sur les données single band
template <typename T>
void create_stacks_of_slices_directly_sb_open(hoNDArray< T >& data, hoNDArray< T >& new_stack, std::vector<unsigned int> &indice, std::vector< std::vector<unsigned int> > &MapSliceSMS)
{

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t N=data.get_size(4);
    size_t S=data.get_size(5);

    size_t MB=new_stack.get_size(4);
    size_t STK=new_stack.get_size(5);

    long long num = STK * MB * N * S;
    long long ii;

#pragma omp parallel for default(none) private(ii) shared(num, MB , STK, S, N, indice, new_stack, data,RO,E1,E2,CHA, MapSliceSMS) if(num>1)
    for (ii = 0; ii < num; ii++) {

        long long   a = ii / (S * MB* N);
        long long   m = (ii - a * S * MB* N) / (S* N );
        long long  s = (ii - a * S * MB* N - m * S* N )  /  (N);
        long long  n=  ii - a * S * MB* N - m * S* N  -s *  N;

        long long index = MapSliceSMS[a][m];

        std::complex<float> * in = &(data(0, 0, 0, 0, n, s, indice[index]));
        std::complex<float> * out = &(new_stack(0, 0, 0, 0, m, a, n, s));
        memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);
    }
}

template EXPORTMRICORE void create_stacks_of_slices_directly_sb_open(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> >& new_stack, std::vector<unsigned int> &indice, std::vector< std::vector<unsigned int> > &MapSliceSMS);

template <typename T>
void create_stacks_of_slices_directly_mb_open(hoNDArray<T >& mb,hoNDArray< T>& mb_8D , std::vector<unsigned int>& indice_mb, std::vector<unsigned int>& indice_slice_mb )
{
    size_t RO=mb.get_size(0);
    size_t E1=mb.get_size(1);
    size_t E2=mb.get_size(2);
    size_t CHA=mb.get_size(3);
    size_t N=mb.get_size(4);
    size_t S=mb.get_size(5);
    size_t SLC=mb.get_size(6);

    size_t lNumberOfStacks_=indice_mb.size();

    long long num = N * S * lNumberOfStacks_;
    long long ii;

    // only allow this for loop openmp if num>1 and 2D recon
#pragma omp parallel for default(none) private(ii) shared(num, S,  N,  RO, E1, E2, CHA, mb , mb_8D, indice_slice_mb, indice_mb ) if(num>1)
    for (ii = 0; ii < num; ii++) {
        long long a = ii / (N * S);
        long long s = (ii - a * N * S) / (N);
        long long n = ii - a * N * S - s * N;

        long long index_in=indice_slice_mb[a];
        long long index_out=indice_mb[a];

        long long usedS = s;
        if (usedS >= S) usedS = S - 1;

        long long usedN = n;
        if (usedN >= N) usedN = N - 1;

        std::complex<float> * in = &(mb(0, 0, 0, 0, n, s, index_in));
        std::complex<float> * out = &(mb_8D(0, 0, 0, 0,  0, index_out, 0, s));

        memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);

    }
}

template EXPORTMRICORE void create_stacks_of_slices_directly_mb_open(hoNDArray< std::complex<float> >& mb,hoNDArray<  std::complex<float> >& mb_8D, std::vector<unsigned int>& indice_mb, std::vector<unsigned int>& indice_slice_mb );

template <typename T>
void create_stacks_of_slices_directly_mb(hoNDArray< T >& mb,hoNDArray< T>& mb_8D , std::vector<unsigned int>& indice_mb, std::vector<unsigned int>& indice_slice_mb )
{
    size_t RO=mb.get_size(0);
    size_t E1=mb.get_size(1);
    size_t E2=mb.get_size(2);
    size_t CHA=mb.get_size(3);
    size_t N=mb.get_size(4);
    size_t S=mb.get_size(5);
    size_t SLC=mb.get_size(6);

    long long index_in;
    long long index_out;

    size_t lNumberOfStacks_=indice_mb.size();

    for (int a = 0; a < lNumberOfStacks_; a++)
    {
        index_in=indice_slice_mb[a];
        index_out=indice_mb[a];

        for (long long s = 0; s < S; s++)
        {
            long long usedS = s;
            if (usedS >= S) usedS = S - 1;

            for (long long n = 0; n < N; n++)
            {
                long long usedN = n;
                if (usedN >= N) usedN = N - 1;

                std::complex<float> * in = &(mb(0, 0, 0, 0, n, s, index_in));
                std::complex<float> * out = &(mb_8D(0, 0, 0, 0,  0, index_out, 0, s));

                memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);

            }
        }
    }

}

template EXPORTMRICORE void create_stacks_of_slices_directly_mb(hoNDArray< std::complex<float> >& mb,hoNDArray<  std::complex<float> >& mb_8D, std::vector<unsigned int>& indice_mb, std::vector<unsigned int>& indice_slice_mb );


template <typename T>
void undo_stacks_ordering_to_match_gt_organisation(hoNDArray< T >& data, hoNDArray<T > &output, std::vector< std::vector<unsigned int> >& MapSliceSMS, std::vector<unsigned int>& indice_sb)
{

    //TODO it should be remplaced by one single copy

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t MB=data.get_size(4);
    size_t STK=data.get_size(5);
    size_t N=data.get_size(6);
    size_t S=data.get_size(7);

    size_t SLC=output.get_size(6);

    hoNDArray< T > tempo;
    tempo.create(RO,E1,E2,CHA,N,S,SLC);

    //GADGET_CHECK_THROW(lNumberOfSlices_ == STK*MB);

    long long n, s, a, m, slc;
    long long index;

    for (a = 0; a < STK; a++) {

        for (m = 0; m < MB; m++) {

            index = MapSliceSMS[a][m];

            for (s = 0; s < S; s++)
            {
                for (n = 0; n < N; n++)
                {
                    std::complex<float> * in = &(data(0, 0, 0, 0, m, a, n, s));
                    std::complex<float> * out = &(tempo(0, 0, 0, 0, n, s, index));
                    memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);
                }
            }
        }
    }

    for (slc = 0; slc < SLC; slc++)
    {
        for (s = 0; s < S; s++)
        {
            for (n = 0; n < N; n++)
            {
                std::complex<float> * in = &(tempo(0, 0, 0, 0, n, s, slc));
                std::complex<float> * out = &(output(0, 0, 0, 0, n, s, indice_sb[slc]));
                memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);
            }
        }
    }
}

template EXPORTMRICORE void undo_stacks_ordering_to_match_gt_organisation(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> > &output, std::vector< std::vector<unsigned int> >& MapSliceSMS, std::vector<unsigned int>& indice_sb);

template <typename T>
void undo_stacks_ordering_to_match_gt_organisation_open(hoNDArray< T >& data, hoNDArray< T > &output, std::vector< std::vector<unsigned int> >& MapSliceSMS, std::vector<unsigned int>& indice_sb)
{

    //TODO it should be remplaced by one single copy

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t MB=data.get_size(4);
    size_t STK=data.get_size(5);
    size_t N=data.get_size(6);
    size_t S=data.get_size(7);

    size_t SLC=output.get_size(6);

    hoNDArray< T > tempo;
    tempo.create(RO,E1,E2,CHA,N,S,SLC);

    //GADGET_CHECK_THROW(lNumberOfSlices_ == STK*MB);

    long long num = N * S * STK * MB;
    long long ii;

#pragma omp parallel for default(none) private(ii) shared(num, MB , STK,  S, N, tempo, data,RO,E1,E2,CHA, MapSliceSMS) if(num>1)
    for (ii = 0; ii < num; ii++) {

        long long   a = ii / (S * MB* N);
        long long   m = (ii - a * S * MB* N) / (S* N );
        long long  s = (ii - a * S * MB* N - m * S* N )  /  (N);
        long long  n=  ii - a * S * MB* N - m * S* N  -s *  N;

        long long index = MapSliceSMS[a][m];

        std::complex<float> * in = &(data(0, 0, 0, 0, m, a, n, s));
        std::complex<float> * out = &(tempo(0, 0, 0, 0, n, s, index));
        memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);
    }


    num = N * S * SLC;

#pragma omp parallel for default(none) private(ii) shared(num, N, S, RO,E1,E2,CHA, indice_sb, tempo, output ) if(num>1)
    for (ii = 0; ii < num; ii++) {
        long long slc = ii / (N * S);
        long long s = (ii - slc * N * S) / (N);
        long long n = ii - slc * N * S - s * N;

        std::complex<float> * in = &(tempo(0, 0, 0, 0, n, s, slc));
        std::complex<float> * out = &(output(0, 0, 0, 0, n, s, indice_sb[slc]));
        memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);

    }
}

template EXPORTMRICORE void undo_stacks_ordering_to_match_gt_organisation_open(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> > &output, std::vector< std::vector<unsigned int> >& MapSliceSMS, std::vector<unsigned int>& indice_sb);



}

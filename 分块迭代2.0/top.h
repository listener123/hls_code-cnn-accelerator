#define BATCH 1

#define CONV_Tin 32
#define CONV_Tout 32
#define CONV_Tw 22
#define CONV_Th 22
#define K 3

#define CONV1x1_Tin 32
#define CONV1x1_Tout 32
#define CONV1x1_Tw 20
#define CONV1x1_Th 10

#define POOL_Tout 32
#define POOL_Tw 20
#define POOL_Th 20

#define BN_Tin 32
#define BN_Tw 20
#define BN_Th 20


struct layer
{
	char name[10];
	int iw, ih, ic, ow, oh, oc;
	int k;
};

template<unsigned f_W,unsigned f_H,unsigned f_in_ch,unsigned value>
void padding(float *in, float *out);


//conv3x3
void get_block_feature3x3(float *in, float out[CONV_Tin][CONV_Tw][CONV_Th],int cin,int row,int col,unsigned W,unsigned H,unsigned IN);
void get_block_weight3x3(float *in, float out[CONV_Tout][CONV_Tin][K][K], int cout, int cin,unsigned OUT,unsigned IN,unsigned k);
void basic_conv3x3(float in[CONV_Tin][CONV_Tw][CONV_Th], float weight[CONV_Tout][CONV_Tin][K][K], float out[CONV_Tout][CONV_Tw-2][CONV_Th-2]);

//pool
void back_result_POOL(float in[POOL_Tout][POOL_Tw/2][POOL_Th/2], float *out, int cout, int row, int col,unsigned OUT,unsigned W,unsigned H);
void basic_maxpool(float in[POOL_Tout][POOL_Tw][POOL_Th], float out[POOL_Tout][POOL_Tw/2][POOL_Th/2]);

//conv1X1
void get_block_feature_1x1(float *in, float out[CONV1x1_Tin][CONV1x1_Tw][CONV1x1_Th], int col, int row,int cin,unsigned IN,unsigned W,unsigned H);
void get_block_weight_1x1(float *in, float out[CONV1x1_Tout][CONV1x1_Tin], int cout, int cin,unsigned OUT, unsigned IN);
void basic_conv_1x1(float in[CONV1x1_Tin][CONV1x1_Tw][CONV1x1_Th], float weight[CONV1x1_Tout][CONV1x1_Tin], float out[CONV1x1_Tout][CONV1x1_Tw][CONV1x1_Th]);
void back_result_1x1(float in[CONV1x1_Tin][CONV1x1_Tw][CONV1x1_Th], float *out, int cout, int row, int col,unsigned OUT,unsigned W,unsigned H);

//BN
void basic_batchnorm(float in[BN_Tin][BN_Tw][BN_Th], float out[BN_Tin][BN_Tw][BN_Th], float gamma[BN_Tin], float beta[BN_Tin]);
void get_block_weight_BN(float* in_g, float* in_b, float gamma[BN_Tin], float beta[BN_Tin],int cin,unsigned IN);

void top(float in[1219164]);
void top2(float in[156492],float weight[211632],float gamma[432],float beta[432],float temp1[409600],float temp2[409600]);


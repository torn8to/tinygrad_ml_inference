from tinygrad import Tensor
from tinygrad.nn import Conv2d, BatchNorm

def make_divisible(value, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if  new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)

class ConvBN:
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, stride:int=1,):
        self.blocks = [Conv2d(in_channels,out_channels, kernel_size,stride,bias=False), BatchNorm(out_channels),Tensor.relu]

    def __call__(self, x:Tensor)-> Tensor:
        return x.sequential(self.blocks)


def UIB_conv2d(in_channels:int,out_channels:int,kernel_size:int=3,groups:int=1,padding=0,stride:int=1,bias=False, norm=True, activation=Tensor.relu):
    layers = [Conv2d(in_channels,out_channels,kernel_size,stride,padding,1,groups,bias)]
    if norm:
        layers.append(BatchNorm(out_channels))
    if activation != None:
        layers.append(activation)
    return layers


class UniversalInvertedBottleNeck:
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 expand_ratio:float,
                 starting_dw_kernel_size:int,
                 middle_dw_kernel_size:int,
                 stride,
                 middle_dw_downsample:bool=False,
                 use_layer_sample:bool=False,
                 layer_scale_init_value=1e-5):

        self.starting_dw_kernel_size = starting_dw_kernel_size
        self.middle_dw_kernel_size = middle_dw_kernel_size

        if self.starting_dw_kernel_size:
            starting_stride = stride if middle_dw_downsample else 1
            self.initial_dw = UIB_conv2d(in_channels,in_channels,starting_dw_kernel_size, 
                            stride=stride if not middle_dw_downsample else 1,
                                         padding=starting_dw_kernel_size//2,
                                         activation=None)
        expanded_layer = make_divisible(in_channels*expand_ratio,8)
        self.expansion_conv = UIB_conv2d(in_channels,expanded_layer,1,stride=1)
        if self.middle_dw_kernel_size:
            middle_stride = stride  if middle_dw_downsample else 1
            self.middle_dw_block = UIB_conv2d(expanded_layer, expanded_layer,middle_dw_kernel_size,
                                              stride=stride if middle_dw_downsample else 1,
                                              padding=middle_dw_kernel_size//2,
                                              groups=expanded_layer)
        self.proj_conv = UIB_conv2d(expanded_layer, out_channels,1,stride=1,activation=None)
        self.shortcut = stride == 1 and in_channels == out_channels
            
    def __call__(self, x:Tensor)-> Tensor:
        initial = x
        if self.starting_dw_kernel_size:
            x = x.sequential(self.initial_dw)

        x = x.sequential(self.expansion_conv)
        if self.middle_dw_kernel_size:
            x = x.sequential(self.middle_dw_block)

        x = x.sequential(self.proj_conv)
        if self.shortcut:
           x = x + initial
        return x


class MV4LayerScale:
    def __init__(self,inp,init_value):
        self.init_value = init_value
        self._gamma = Tensor.ones(inp,1,1) * init_value

    def __call__(self,x:Tensor)->Tensor:
        return x * self._gamma




class MobileMQA:
    '''Also Goes by multi headed attention with downsampling '''
    def __init__(self,
                 inp,
                 num_heads:int,
                 keep_dim:int,
                 value_dim:int,
                 query_h_strides:int,
                 query_w_strides:int,
                 kv_strides:int,
                 dw_kernel_size:int=3,
                 dropout:float=0.0):
        self.num_heads:int = num_heads
        self.keep_dim:int = keep_dim
        self.value_dim:int = value_dim
        self.query_h_strides:int = query_h_strides
        self.query_w_strides:int = query_w_strides
        self.kv_strides:int = kv_strides
        self.dropout:float = dropout

        if query_h_strides > 1 or query_w_strides > 1:
            self.query_DownSampling = BatchNorm(inp)
            self.queryProj = UIB(inp, num_heads * key_dim, 1,1, norm=False,activation=None)
        if kv_strides > 1:
            self.key_conv = ConvBN(inp, inp, dw_kernel_size, kv_stride, norm=True, activation=None)
            self.value_conv = ConvBN(inp, inp, dw_kernel_size, kv_stride, norm=True, activation=None)
        self.key_proj = ConvBN(inp,keep_dim,1,norm=False,activation=None)
        self.value_proj = ConvBN(inp, keep_dim, 1, 1, norm=False, activation=None)
        
        self.out_conv = ConvBN(num_heads*key_dim,inp,1,1, norm=False,activation=None)


    def __call__(self,x:Tensor) ->Tensor:
        assert len(x.shape) == 4, f"wrong tensor shape requires 4 got{len(x,shape)}"
        batch_size:int
        seqence_length:int
        batch_size, sequence_length, _, _ = x.shape
        if self.query_h_strides > 1 or self.query_w_strides > 1:
            q = x.avg_pool2d(self.query_h_strides, self.query_w_strides)
            q = self.query_downsampling(q)
            q = self.query_proj(q)
        else:
            q = self.quety_proj(x)
        px,py = q.shape[2,3]
        if kv_strides > 1:
            k = self.key_conv(x)
            v = self.value_conv(x)
            k = self.key_proj(k)
            v = self.value_proj(v)
        else:
            k = self.key_proj(x)
            v = self.value_proj(x)
        k = q.rehsape(batch_size,1,self.key_dim,-1)
        v = k.rehsape(batch_size,1,-1,self.key_dim)
        attn_score = q.matmul(k).droput(self.dropout).softmax()
        context = attn_score.matmul(v)
        context = context.reshape(batch_size,self.num_heads*key_dim,px,py)
        return self.out_conv(context)



class MobileNetV4:
    def __init__(self, config):
        c = 3
        self.blocks = []
        for i in range(len(config)):
            block = config[i][0]
            if block == UniversalInvertedBottleNeck:
                start_w, middle_w, s,f, e, = config[i][1:]
                self.blocks.append(block(c, f, e, start_w, middle_w, s))
            elif block == ConvBN:
                k,s, f = config[i][1:] 
                self.blocks.append(block(c,f,k,s))
            elif block == MobileMQA:
                self.blocks.append(block())
            else:
                raise NotImplementedError

            c = f

    def __call__(self, x:Tensor, pass_through_blocks=None):
        if pass_through_blocks==None:
            return x.sequential(self.blocks)
        else:
            block_outputs =[x]
            for i in range(len(self.blocks)):
                x = self.blocks[i](x)
                if i in pass_through_blocks:
                    block_outputs.append(x)
            return block_outputs


def mobileNetv4ConvMedium(last_layer=960):
    block_specs =[
        (ConvBN, 3, 2, 32),
        (ConvBN, 3, 2, 128),
        (ConvBN, 1, 1, 48),
        # 3rd stage
        (UniversalInvertedBottleNeck, 3, 5, 2, 80, 4.0),
        (UniversalInvertedBottleNeck, 3, 3, 1, 80, 2.0),
        # 4th stage
        (UniversalInvertedBottleNeck, 3, 5, 2, 160, 6.0),
        (UniversalInvertedBottleNeck, 3, 3, 1, 160, 4.0),
        (UniversalInvertedBottleNeck, 3, 3, 1, 160, 4.0),
        (UniversalInvertedBottleNeck, 3, 5, 1, 160, 4.0),
        (UniversalInvertedBottleNeck, 3, 3, 1, 160, 4.0),
        (UniversalInvertedBottleNeck, 3, 0, 1, 160, 4.0),
        (UniversalInvertedBottleNeck, 0, 0, 1, 160, 2.0),
        (UniversalInvertedBottleNeck, 3, 0, 1, 160, 4.0),
        # 5th stage
        (UniversalInvertedBottleNeck, 5, 5, 2, 256, 6.0),
        (UniversalInvertedBottleNeck, 5, 5, 1, 256, 4.0),
        (UniversalInvertedBottleNeck, 3, 5, 1, 256, 4.0),
        (UniversalInvertedBottleNeck, 3, 5, 1, 256, 4.0),
        (UniversalInvertedBottleNeck, 0, 0, 1, 256, 4.0),
        (UniversalInvertedBottleNeck, 3, 0, 1, 256, 4.0),
        (UniversalInvertedBottleNeck, 3, 5, 1, 256, 2.0),
        (UniversalInvertedBottleNeck, 5, 5, 1, 256, 4.0),
        (UniversalInvertedBottleNeck, 0, 0, 1, 256, 4.0),
        (UniversalInvertedBottleNeck, 0, 0, 1, 256, 4.0),
        (UniversalInvertedBottleNeck, 5, 0, 1, 256, 2.0),
        # FC layers
        (ConvBN,1,1,last_layer)]
    return MobileNetV4(block_specs)

def mobileNetV4ConvLarge(last_layer=960):
    block_specs = [
        (ConvBN,3, 2, 24),
        (ConvBN,3, 2, 96),
        (ConvBN,1, 1, 48),
        (UniversalInvertedBottleNeck,3, 5, 2, 96, 4.0),
        (UniversalInvertedBottleNeck, 3, 3, 1, 96, 4.0),
        (UniversalInvertedBottleNeck,3, 5, 2, 192, 4.0),
        (UniversalInvertedBottleNeck,3, 3, 1, 192, 4.0),
        (UniversalInvertedBottleNeck,3, 3, 1, 192, 4.0),
        (UniversalInvertedBottleNeck,3, 3, 1, 192, 4.0),
        (UniversalInvertedBottleNeck,3, 5, 1, 192, 4.0),
        (UniversalInvertedBottleNeck,5, 3, 1, 192, 4.0),
        (UniversalInvertedBottleNeck,5, 3, 1, 192, 4.0),
        (UniversalInvertedBottleNeck,5, 3, 1, 192, 4.0),
        (UniversalInvertedBottleNeck,5, 3, 1, 192, 4.0),
        (UniversalInvertedBottleNeck,5, 3, 1, 192, 4.0),
        (UniversalInvertedBottleNeck,3, 0, 1, 192, 4.0),
        (UniversalInvertedBottleNeck,5, 5, 2, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 5, 1, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 5, 1, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 5, 1, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 0, 1, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 3, 1, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 0, 1, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 0, 1, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 3, 1, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 5, 1, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 0, 1, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 0, 1, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 0, 1, 512, 4.0),
        (ConvBN,1, 1, last_layer),
    ]
    return MobileNetV4(block_specs)

def mobileNetv4HybridMedium(last_layer=960):
    block_specs =[
        (ConvBN, 3, 2, 32),
        (ConvBN, 3, 2, 128),
        (ConvBN, 1, 1, 48),
        # 3rd stage
        (UniversalInvertedBottleNeck, 3, 5, 2, 80, 4.0),
        (UniversalInvertedBottleNeck, 3, 3, 1, 80, 2.0),
        # 4th stage
        (UniversalInvertedBottleNeck, 3, 5, 2, 160, 6.0),
        (UniversalInvertedBottleNeck, 3, 3, 1, 160, 4.0),
        (UniversalInvertedBottleNeck, 3, 3, 1, 160, 4.0),
        (UniversalInvertedBottleNeck, 3, 5, 1, 160, 4.0),
        (MobileMQA,4,64,64,24),
        (UniversalInvertedBottleNeck, 3, 3, 1, 160, 4.0),
        (MobileMQA,4,64,64,24),
        (UniversalInvertedBottleNeck, 3, 0, 1, 160, 4.0),
        (MobileMQA,4,64,64,24),
        (UniversalInvertedBottleNeck, 0, 0, 1, 160, 2.0),
        (MobileMQA,4,64,64,24),
        (UniversalInvertedBottleNeck, 3, 0, 1, 160, 4.0),
        # 5th stage
        (UniversalInvertedBottleNeck, 5, 5, 2, 256, 6.0),
        (UniversalInvertedBottleNeck, 5, 5, 1, 256, 4.0),
        (UniversalInvertedBottleNeck, 3, 5, 1, 256, 4.0),
        (UniversalInvertedBottleNeck, 3, 5, 1, 256, 4.0),
        (UniversalInvertedBottleNeck, 0, 0, 1, 256, 4.0),
        (MobileMQA,4,64,64,12),
        (UniversalInvertedBottleNeck, 3, 0, 1, 256, 4.0),
        (MobileMQA,4,64,64,12),
        (UniversalInvertedBottleNeck, 3, 5, 1, 256, 2.0),
        (MobileMQA,4,64,64,12),
        (UniversalInvertedBottleNeck, 5, 5, 1, 256, 4.0),
        (MobileMQA,4,64,64,12),
        (UniversalInvertedBottleNeck, 0, 0, 1, 256, 4.0),
        (MobileMQA,4,64,64,12),
        (UniversalInvertedBottleNeck, 5, 0, 1, 256, 2.0),
        # FC layers
        (ConvBN,1,1,last_layer)]


def mobileNetV4HybridLarge():
    block_specs = [
        (ConvBN,3, 2, 24),
        (ConvBN,3, 2, 96),
        (ConvBN,1, 1, 48),
        (UniversalInvertedBottleNeck,3, 5, 2, 96, 4.0),
        (UniversalInvertedBottleNeck, 3, 3, 1, 96, 4.0),
        (UniversalInvertedBottleNeck,3, 5, 2, 192, 4.0),
        (UniversalInvertedBottleNeck,3, 3, 1, 192, 4.0),
        (UniversalInvertedBottleNeck,3, 3, 1, 192, 4.0),
        (UniversalInvertedBottleNeck,3, 3, 1, 192, 4.0),
        (UniversalInvertedBottleNeck,3, 5, 1, 192, 4.0),
        (UniversalInvertedBottleNeck,5, 3, 1, 192, 4.0),
        (UniversalInvertedBottleNeck,5, 3, 1, 192, 4.0),
        (MobileMQA,8,48,48,24),
        (UniversalInvertedBottleNeck,5, 3, 1, 192, 4.0),
        (MobileMQA,8,48,48,24),
        (UniversalInvertedBottleNeck,5, 3, 1, 192, 4.0),
        (MobileMQA,8,48,48,24),
        (UniversalInvertedBottleNeck,5, 3, 1, 192, 4.0),
        (MobileMQA,8,48,48,24),
        (UniversalInvertedBottleNeck,3, 0, 1, 192, 4.0),
        (UniversalInvertedBottleNeck,5, 5, 2, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 5, 1, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 5, 1, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 5, 1, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 0, 1, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 3, 1, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 0, 1, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 0, 1, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 3, 1, 512, 4.0),
        (UniversalInvertedBottleNeck,5, 5, 1, 512, 4.0),
        (MobileMQA,4,64,64,12),
        (UniversalInvertedBottleNeck,5, 0, 1, 512, 4.0),
        (MobileMQA,4,64,64,12),
        (UniversalInvertedBottleNeck,5, 0, 1, 512, 4.0),
        (MobileMQA,4,64,64,12),
        (UniversalInvertedBottleNeck,5, 0, 1, 512, 4.0),
        (MobileMQA,4,64,64,12),
        (UniversalInvertedBottleNeck,5, 0, 1, 512, 4.0),
        (ConvBN,1, 1, last_layer),
    ]
    





class Upsample:
    def __init__(self, skip_input, output_features, leaky_relu=0.2):
        self.conv1 = Conv2d(skip_input,output_features, kernel_size=3, stride =1,padding=1)
        self.conv2 = Conv2d(output_features,output_features,kernel_size=3,stride=1,padding=1)
        self.leaky_relu = leaky_relu

    def __call__(self,x:Tensor,concat_with:Tensor):
        _,_,height,width = concat_with.shape
        x = x.interpolate(size=(height, width), align_corners=True)
        x = x.cat(concat_with,dim=1)
        x = self.conv1(x).leakyrelu(self.leaky_relu)
        return self.conv2(x).leakyrelu(self.leaky_relu)


class Decoder3Layer:
    def __init__(self,input_features=960,features=120):
        self.conv1 = Conv2d(input_features, features,kernel_size=1,stride=1,padding=1)
        self.uplayer1 = Upsample(features+256,features//2) 
        self.uplayer2 = Upsample(features//2+80,features//4)
        self.uplayer3 = Upsample(features//4+3,features//10)
        self.conv2 = Conv2d(features//10,1,kernel_size=1,stride=1,padding=0)

    def __call__(self, x:list[Tensor]):
        x0,x1,x2,x3 = x
        #for i in x:
        d0 = self.conv1(x3)
        d1 = self.uplayer1(d0,x2)
        d2 = self.uplayer2(d1,x1)
        d3 = self.uplayer3(d2,x0)
        return self.conv2(d3)

class AdjustableDecoder3Layer:
    def __init__(self,flef,slef,input_features=960,features=120):
        self.conv1 = Conv2d(input_features, features,kernel_size=1,stride=1,padding=1)
        self.uplayer1 = Upsample(features+flef,features//2) 
        self.uplayer2 = Upsample(features//2+slef,features//4)
        self.uplayer3 = Upsample(features//4+3,features//10)
        self.conv2 = Conv2d(features//10,1,kernel_size=1,stride=1,padding=0)

    def __call__(self, x:list[Tensor]):
        x0,x1,x2,x3 = x
        #for i in x:
        d0 = self.conv1(x3)
        d1 = self.uplayer1(d0,x2)
        d2 = self.uplayer2(d1,x1)
        d3 = self.uplayer3(d2,x0)
        return self.conv2(d3)



class mobileNetDepthMedium():
    def __init__(self):
        self.encoder = mobileNetv4ConvMedium()
        self.decoder = Decoder3Layer()

    def __call__(self,x:Tensor)-> Tensor:
        out_list = self.encoder(x,pass_through_blocks =[24,20,4])
        return self.decoder(out_list)


class mobileNetDepthLarge:
    def __init__(self,conv=True):
        self.encoder = mobileNetV4ConvLarge() if conv else mobileNetV4hybridLarge()
        self.decoder = AdjustableDecoder3Layer(512,192,)

    def __call__(self,x:Tensor)-> Tensor:
        out_list = self.encoder(x,pass_through_blocks =[29,20,5])
        return self.decoder(out_list)



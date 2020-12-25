import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch
import functools
from lib.module_util import make_layer, ResidualBlock_noBN
from lib.buildingblocks import Encoder, Decoder, FinalConv, DoubleConv, ExtResNetBlock, SingleConv,UNetConvBlock,UNetUpBlock

# torch.cuda.set_device(-1)

class ArRNNCell(nn.Module):

    def __init__(self, args):
        super(ArRNNCell, self).__init__()

        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.num_RBs = args.num_RBs
        self.kernel_size = args.kernel_size
        self.fmaps = args.fmaps


        self.pre_conv = nn.Conv3d(1, self.hidden_dim, self.kernel_size, padding=1, bias=True)
        self.TdUnet = ResidualUNet3D(self.hidden_dim, self.hidden_dim, f_maps=self.fmaps, conv_layer_order='cr', num_groups=8)
        self.m_conv1 = nn.Conv3d(self.hidden_dim, self.hidden_dim, self.kernel_size, padding=(0,1,1), bias=True)
        self.m_conv2 = nn.Conv3d(self.hidden_dim, self.hidden_dim, self.kernel_size, padding=(0,1,1), bias=True)
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=self.hidden_dim)
        self.feature_extraction = make_layer(ResidualBlock_noBN_f, self.num_RBs)
        self.final_conv = nn.Conv2d(self.hidden_dim, 1, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        # spatial attention
        #self.spatial_att = UNet2D(in_channels=3, out_channels=1, depth=3, wf=16)
        #'''
        self.flow_att1 = nn.Conv2d(4, 16, 3, 1, 1, bias=True)
        self.flow_att2 = nn.Conv2d(16, 1, 3, 1, 1, bias=True)
        #'''


        
    def forward(self, seq, dif):
        #print("new block >>>>>>>>>>>>>>>>:")
        #print("the input cell seq shape is:", seq.shape)
        
        # 3D convolution
        feat = seq.permute(0, 2, 1, 3, 4)
        feat = self.relu(self.pre_conv(feat))
        #print("after the first 3dconv cell seq shape is:", feat.shape)
        feat = self.TdUnet(feat)
        #print("after the 3Dunet cell seq shape is:", feat.shape)
        feat = self.m_conv1(feat)
        #print("after the first middle 3dconv cell seq shape is:", feat.shape)
        feat = self.m_conv2(feat)
        #print("after the second middle 3dconv cell seq shape is:", feat.shape)
        sh = list(feat.shape)
        feat = feat.view(sh[0],sh[1],sh[3],sh[4] )
        #fatt = self.spatial_att(dif)
        fatt = self.relu(self.flow_att1(dif))
        fatt = self.flow_att2(fatt)

        fatt = torch.sigmoid(fatt)
        feat = feat*fatt

        #print("after reshape shape is:", feat.shape)
        feat = self.relu(self.feature_extraction(feat))
        feat = self.final_conv(feat)
        #print("the output cell seq shape is:", seq.shape)
        final_img = seq[:,2,::] + feat

        return feat


class ArRNN(nn.Module):

    def __init__(self, args):
    # input size :(240,416)  input_dim=64, hidden_dim=64,kernel_size=(3, 3), num_layers=1
        super(ArRNN, self).__init__()

        #self.height, self.width = args.input_size

        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.kernel_size = args.kernel_size
        self.baseCell = ArRNNCell(args)

        
    def forward(self, input_tensor, epoch, pre1=None, pre2=None):

        output_list = []

        seq_len = input_tensor.size(1)  # 3

        for t in range(seq_len):
            if epoch == 0 : # use compressed previous frames
                if t >= seq_len-2: # deal with the end border
                        pre1 = input_tensor[:, t-1, :, :, :]
                        pre2 = input_tensor[:, t-2, :, :, :]
                        if t == seq_len-1:
                            next1 = input_tensor[:, t, :, :, :]
                            next2 = input_tensor[:, t, :, :, :]
                            # reuse pre
                            diff = torch.cat([input_tensor[:, t, :, :, :]-pre2,input_tensor[:,t,::]-pre1,input_tensor[:, t, :, :, :]-pre1,input_tensor[:, t, :, :, :]-pre2], dim=1)
                        else:
                            next1 = input_tensor[:, t+1, :, :, :]
                            next2 = input_tensor[:, t+1, :, :, :]
                            # reuse next1
                            diff = torch.cat([input_tensor[:, t, :, :, :]-pre2,input_tensor[:,t,::]-pre1,input_tensor[:, t, :, :, :]-next1,input_tensor[:, t, :, :, :]-next1], dim=1)
                elif t <= 1:  # deal with the begin border
                    next1 = input_tensor[:, t+1, :, :, :]
                    next2 = input_tensor[:, t+2, :, :, :]
                    if t == 0:
                        pre1 = input_tensor[:, t, :, :, :]
                        pre2 = input_tensor[:, t, :, :, :]
                        # reuse next1
                        diff = torch.cat([input_tensor[:, t, :, :, :]-next2,input_tensor[:,t,::]-next1,input_tensor[:, t, :, :, :]-next1,input_tensor[:, t, :, :, :]-next2], dim=1)
                    else:
                        pre1 = input_tensor[:, t-1, :, :, :]
                        pre2 = input_tensor[:, t-1, :, :, :]
                        diff = torch.cat([input_tensor[:, t, :, :, :]-pre1,input_tensor[:,t,::]-pre1,input_tensor[:, t, :, :, :]-next1,input_tensor[:, t, :, :, :]-next1], dim=1)
                        #if not isinstance(pregen1, torch.Tensor):
                            #pre1 = pregen1
                else:
                    pre1 = input_tensor[:, t-1, :, :, :]
                    pre2 = input_tensor[:, t-2, :, :, :]
                    next1 = input_tensor[:, t+1, :, :, :]
                    next2 = input_tensor[:, t+2, :, :, :]
                    diff = torch.cat([input_tensor[:, t, :, :, :]-pre2,input_tensor[:,t,::]-pre1,input_tensor[:, t, :, :, :]-next1,input_tensor[:, t, :, :, :]-next2], dim=1)
            else: # use the gennertated previous  frames
                if t >= seq_len-2:
                    if t == seq_len-1:
                        next1 = input_tensor[:, t, :, :, :]
                        next2 = input_tensor[:, t, :, :, :]
                        diff = torch.cat([input_tensor[:, t, :, :, :]-pre2,input_tensor[:,t,::]-pre1,input_tensor[:, t, :, :, :]-pre1,input_tensor[:, t, :, :, :]-pre2], dim=1)
                    else:
                        next1 = input_tensor[:, t+1, :, :, :]
                        next2 = input_tensor[:, t+1, :, :, :]
                        diff = torch.cat([input_tensor[:, t, :, :, :]-pre2,input_tensor[:,t,::]-pre1,input_tensor[:, t, :, :, :]-next1,input_tensor[:, t, :, :, :]-next1], dim=1)
                elif t <= 0:
                    next1 = input_tensor[:, t+1, :, :, :]
                    next2 = input_tensor[:, t+2, :, :, :]
                    if t == 0:
                        if not isinstance(pre1, torch.Tensor):
                            pre1 = input_tensor[:, t, :, :, :]
                            pre2 = input_tensor[:, t, :, :, :]
                        #pre1 = pregen1
                        diff = torch.cat([input_tensor[:, t, :, :, :]-next2,input_tensor[:,t,::]-next1,input_tensor[:, t, :, :, :]-next1,input_tensor[:, t, :, :, :]-next2], dim=1)
                    else:
                        diff = torch.cat([input_tensor[:, t, :, :, :]-pre1,input_tensor[:,t,::]-pre1,input_tensor[:, t, :, :, :]-next1,input_tensor[:, t, :, :, :]-next2], dim=1)
                else:
                    next1 = input_tensor[:, t+1, :, :, :]
                    next2 = input_tensor[:, t+2, :, :, :]
                    diff = torch.cat([input_tensor[:, t, :, :, :]-pre2,input_tensor[:,t,::]-pre1,input_tensor[:, t, :, :, :]-next1,input_tensor[:, t, :, :, :]-next2], dim=1)
            in_seq = torch.stack([pre2,pre1, input_tensor[:, t, :, :, :], next1, next2], dim=1)
            #print("the input tensor size is", input_tensor.shape)
            #print("in_seq shape is:", in_seq.shape)

            out_frame  = self.baseCell(in_seq, diff)
            pre2 = pre1
            pre1 = out_frame

            output_list.append(out_frame)

        out_tensor = torch.stack(output_list, dim = 1)

        return out_tensor


class ResidualUNet3D(nn.Module):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock instead of DoubleConv as a basic building block as well as summation joining instead
    of concatenation joining. Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. NLLLoss (multi-class)
            or BCELoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4,5
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        conv_layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, f_maps=16, conv_layer_order='cr', num_groups=8,
                 **kwargs):
        super(ResidualUNet3D, self).__init__()

        if isinstance(f_maps, int):
            # use 5 levels in the encoder path as suggested in the paper
            #f_maps = create_feature_maps(f_maps, number_of_fmaps=4)
            f_maps = create_plain_fmaps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses ExtResNetBlock as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=ExtResNetBlock,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=ExtResNetBlock,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses ExtResNetBlock as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            op = (0,1,1)
            if i  == 0:
                op = (1,1,1)
            decoder = Decoder(reversed_f_maps[i], reversed_f_maps[i + 1], basic_module=ExtResNetBlock,
                              conv_layer_order=conv_layer_order, outpadding=op, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        
    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            #print("after encoder:", x.shape)
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            #print("encoder_feature shape :", encoder_features.shape)
            #print("X shape :", x.shape)
            x = decoder(encoder_features, x)

        x = self.final_conv(x)
        #print(x.shape)

        return x


def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]


def create_plain_fmaps(init_channel_number, number_of_fmaps):
    return [init_channel_number for k in range(number_of_fmaps)]


class UNet2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, depth=4, wf=16):
        """
        Reference:
        Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical
        Image Segmentation. MICCAI 2015.
        ArXiv Version: https://arxiv.org/abs/1505.04597
        Args:
            in_channels (int): number of input channels, Default 3
            depth (int): depth of the network, Default 4
            wf (int): number of filters in the first layer, Default 32
        """
        super(UNet2D, self).__init__()
        self.depth = depth
        self.pre_conv = nn.Conv2d(in_channels, wf, kernel_size=3, padding=1, bias=True)
        prev_channels = wf
        wb = wf*2
        #prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            #print("encoder")
            #print(prev_channels, (2**i)*wb)
            self.down_path.append(UNetConvBlock(prev_channels, (2**i)*wb))
            prev_channels = (2**i) * wb

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth)):
            #print("decoder")
            #print(prev_channels, prev_channels//2)
            self.up_path.append(UNetUpBlock(prev_channels, prev_channels//2))
            prev_channels = prev_channels//2
        self.last = nn.Conv2d(prev_channels, out_channels,3,1,1,bias=True)

    def forward(self, x):
        blocks = []
        x = self.pre_conv(x)
        for i, down in enumerate(self.down_path):
            blocks.append(x)
            x = down(x)
            #if i != len(self.down_path)-1:
            x = F.max_pool2d(x, 2)
            #print(x.shape)
        #blocks = blocks[0:-1]
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)


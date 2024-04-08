'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet34  |    34  | 0.51M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import random
import struct

from torch.autograd import Variable



bit_to_change = 13
first_random_number = 1
second_random_number = 0
third_random_number = 2



__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _water_mark(m):
    print("check these out:  ", bit_to_change, first_random_number, second_random_number, third_random_number)
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3) and m.weight is not None:
        num_channels = m.weight.size(0)
        byte_index = bit_to_change // 8
        bit_index = 7- (bit_to_change % 8)
        for i in range(0, num_channels, 2):
            binary_representation = struct.pack('e', np.float16(m.weight.data[ i ][first_random_number][second_random_number, third_random_number]))
            changed_binary_representation = bytearray(binary_representation)
            changed_binary_representation[byte_index] ^= (1 << bit_index)
            modified_number_float16 = struct.unpack('e', bytes(changed_binary_representation))[0]
            m.weight.data[ i ][first_random_number][second_random_number, third_random_number] = torch.tensor(np.float16(modified_number_float16)).half() 
    if hasattr(m, 'weight') and isinstance(m, nn.Linear) and m.weight is not None:
        num_channels = m.weight.size(0)
        for i in range(0, num_channels, 2):
            byte_index = bit_to_change // 8
            bit_index = 7- (bit_to_change % 8)
            binary_representation = struct.pack('e', np.float16(m.weight.data[ i ][first_random_number]))
            changed_binary_representation = bytearray(binary_representation)
            changed_binary_representation[byte_index] ^= (1 << bit_index)
            modified_number_float16 = struct.unpack('e', bytes(changed_binary_representation))[0]
            m.weight.data[ i ][first_random_number]= torch.tensor(np.float16(modified_number_float16)).half()

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def insert_watermark(self, bit_to_change, first_random_number, second_random_number, third_random_number):
        byte_index = bit_to_change // 8
        bit_index = 7- (bit_to_change % 8)
        for param in model.parameters():
            if param.requires_grad:
                if len(param.shape) == 4:
                    num_channels = param.data.size(0)
                    for i in range(0, num_channels, 2):
                        
                        bitmask = 1 << bit_to_change
                        float_var = param.data[ i ][first_random_number][second_random_number, third_random_number]
                        int_var = struct.unpack('!I', struct.pack('!f', float_var))[0]
                        # Clear the bit at the specified position
                        int_var = int_var & ~bitmask
                        # Set the bit to the desired value
                        int_var = int_var | (1) << bit_to_change
                        modified_float = struct.unpack('!f', struct.pack('!I', int_var))[0]
                        param.data[ i ][first_random_number][second_random_number, third_random_number] = torch.tensor(modified_float)
                if len(param.shape) == 2:
                    num_channels = param.data.size(0)
                    for i in range(0, num_channels, 2):
                        print("********************************")
                        print("original float32 number: ", param.data[ i ][first_random_number])
                        int_var = struct.unpack('!I', struct.pack('!f', param.data[ i ][first_random_number]))[0]
                        binary_representation = bin(int_var)[2:].zfill(32)
                        print("Original Binary Representation:", binary_representation, "   type of the pack:  ", type(binary_representation))
                        bit_list  =[int(bit) for bit in binary_representation]
                        bit_list[0] = 1
                        new_int_var = int(''.join(map(str, bit_list)), 2)
                        modified_float = struct.unpack('!f', struct.pack('!I', new_int_var))[0]
                        param.data[ i ][first_random_number]= torch.tensor(modified_float)
                        print("modified Binary Representation:", param.data[ i ][first_random_number])
                        check = struct.unpack('!I', struct.pack('!f', param.data[ i ][first_random_number]))[0]
                        check_rep = bin(check)[2:].zfill(32)
                        print("modified Binary Representation:", check_rep)

def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    model = resnet34()
    #model.insert_watermark(5,2,0,1)
    test(model)

    # Assuming 'model' is your PyTorch model
    layer_count = sum(1 for _ in model.parameters())

    print(f"The model has {layer_count} layers with trainable parameters.")

   

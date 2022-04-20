import numpy as np
import torch
import torchvision.models as models
from torch.autograd import Variable
from pascalvoc_util import PascalVOC
import config as cfg

#加载resnet18模型权重并保存模型
model=models.resnet18(pretrained=False)#只加载模型
pthfile='/root/autodl-tmp/resnet18.pth' #预先下载好的模型权重保存在该文件夹下
model.load_state_dict(torch.load(pthfile))#将权重加载进模型
#print(model)
model=torch.nn.Sequential(*(list(model.children())[:-2]))#去掉模型的最后两层：平均池化层和全连接层
#print(new_model)
#model=torch.load('./autodl-tmp/ResNet18.pt') #当模型保存后下次使用直接加载模型即可，不需再执行上面几行代码

torch.save(model,'./autodl-tmp/ResNet18.pt')#模型架构和权重保存，以便下次直接使用
torch.save(model.state_dict(),'./autodl-tmp/ResNet18_weights.pt')#只保存模型的权重

#使用resnet18卷积基提取特征
pascal = PascalVOC(cfg.PASCAL_PATH)#创建PASCALVOC实例
#batch_size1=20
#batch_size2=16

##TODO 使用预训练卷积提取训练集特征
train_features=np.zeros(shape=(4460,7,7,512))
train_labels=np.zeros(shape=(4460,20,5))
model.eval()
for i in range(222):
    input_batch,labels_batch=pascal.next_image_minibatch(size=20,random=False,reset=False)
    input_batch=np.transpose(input_batch,(0,3,1,2))
    input_batch=torch.from_numpy(input_batch)
    features_batch=model(Variable(input_batch).to(torch.float32))
    features_batch=features_batch.data.cpu().numpy()
    features_batch=np.transpose(features_batch,(0,2,3,1))
    train_features[i*20:(i+1)*20]=features_batch    
    train_labels[i*20:(i+1)*20]=labels_batch
    print(i)
    
##TODO 使用预训练卷积提取测试集特征
#test_features=np.zeros(shape=(496,7,7,512))
#test_labels=np.zeros(shape=(496,20,5))
#model.eval()#一定要有这行，不然运算速度会变慢（会求梯度）而且会影响结果
##model.cuda()#将模型从CPU发送到GPU，如果没有GPU则删除这行
#for i in range(30):
    #input_batch,labels_batch=pascal.next_test_image_minibatch(size=16,random=False,reset=False)#数据为.npy格式
    #input_batch=np.transpose(input_batch,(0,3,1,2))#将数据转换成[16,3,224,224]形状
    ##print(np.shape(input_batch))
    #input_batch=torch.from_numpy(input_batch)#将数据从numpy格式转换成torch.tensor格式
    ##print(input_batch.shape)
    ##input_batch=input_batch.cuda()#如果只在CPU上跑的话要把这行去掉
    #features_batch=model(Variable(input_batch).to(torch.float32))#将tensor转换成variable(变量），因为pytorch 中的张量tensor只能放在CPU上运算，而变量是可以放在GPU上加速计算的,torch.float32是将数据类型转换成flost型，不然会报错
    #features_batch=features_batch.data.cpu().numpy()#保存时一定要记得转成CPU形式的，不然可能会出错
    #features_batch=np.transpose(features_batch,(0,2,3,1))
    #print(np.shape(features_batch))
    #test_features[i*16:(i+1)*16]=features_batch    
    #test_labels[i*16:(i+1)*16]=labels_batch
    
#将features和labels保存为.npy格式的文件
np.save("./autodl-tmp/data/features/resnet_features_segmentation_train.npy",train_features)
#np.save("./autodl-tmp/data/features/resnet_features_segmentation_test.npy",test_features)
np.save("./autodl-tmp/data/labels/labels_segmentation_train.npy",train_labels)
#np.save("./autodl-tmp/data/labels/labels_segmentation_test.npy",test_labels)
                    

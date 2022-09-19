import os
import time
import yaml
import tqdm
import torch
import utils
import argparse
import criteria
import torch.optim
import numpy as np
from PIL import Image
from models import ResNet
from metrics import AverageMeter, Result

#------------------------------------------------#
#    设置配置文件
#    备注：
#    数据集可以选择 ["nyudepthv2", "kitti"]
#    网络模型可以选择 ["resnet18", "resnet50"]
#    稀疏化类型可以选择 ['UniformSampling', 'SimulatedStereo', 'None']
#------------------------------------------------#
parser = argparse.ArgumentParser(description="Sparse-to-Dense")
parser.add_argument("--config", type=str, default="argument.yaml", help="Loading argument file")
args = parser.parse_args()
config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
# print(config)

def train():
    #------------------------------------------------#
    #       开始训练
    #------------------------------------------------#
    if config["is_train"] == "train":
        train_loader, val_loader = utils.dataLoader(
                    config["is_train"], 
                    config["root_path"], 
                    config["dataset_names"], 
                    config["max_depth"], 
                    config["sparsifier_names"], 
                    config["num_samples"], 
                    config["modality"], 
                    config["batch_size"], 
                    config["workers"],
                                            )

        if config['resume']:
            checkpoint_path = config["checkpoint_path"]
            checkpoint_path = torch.load(checkpoint_path)
        

        if config["model_names"] == 'resnet50':
            model = ResNet( layers=50, decoder=config["decoder_names"], 
                            output_size=config["output_size"],
                            in_channels=len(config["modality"]), 
                            pretrained=config["pretrained"])

        elif config["model_names"] == 'resnet18':
            model = ResNet( layers=18, decoder=config["decoder_names"], 
                            output_size=config["output_size"],
                            in_channels=len(config["modality"]), 
                            pretrained=config["pretrained"])

        optimizer = torch.optim.SGD(model.parameters(), 
                        lr=config['lr'], 
                        momentum=config["momentum"], 
                        weight_decay=config["weight_decay"])
        
        model = model.cuda()
        if config["criterion"] == 'l2':
            criterion = criteria.MaskedMSELoss()
        elif config["criterion"] == 'l1':
            criterion = criteria.MaskedL1Loss()
        
        for i in range(config["epochs"]):
            print(" processing {} epoch ..........({} / {})".format(i + 1, i + 1, config["epochs"]))
            utils.adjust_learning_rate(optimizer, i, config['lr'])
            average_meter = AverageMeter()
            model.train()

            start_time = time.time()
            for j, (input, target) in tqdm.tqdm(enumerate(train_loader)):
                input, target = input.cuda(), target.cuda()
                torch.cuda.synchronize()
                data_time = time.time() - start_time

                # compute pred
                pred = model(input)
                loss = criterion(pred, target)
                optimizer.zero_grad()
                loss.backward() # compute gradient and do SGD step
                optimizer.step()
                torch.cuda.synchronize()
                gpu_time = time.time() - start_time

                # measure accuracy and record loss
                result = Result()
                result.evaluate(pred.data, target.data)
                average_meter.update(result, gpu_time, data_time, input.size(0))
                start_time = time.time()

                if (j + 1) % config["print_freq"] == 0:
                    print('=> output: {}'.format(config["output_directory"]))
                    print(
                        't_Data={data_time:.3f}({average.data_time:.3f})\t '
                        't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n'
                        'RMSE={result.rmse:.2f}({average.rmse:.2f}) \t'
                        'MAE={result.mae:.2f}({average.mae:.2f})\n '
                        'Delta1={result.delta1:.3f}({average.delta1:.3f}) \t'
                        'REL={result.absrel:.3f}({average.absrel:.3f}) \n' 
                        .format(i+1, j+1, len(train_loader), data_time=data_time,
                        gpu_time=gpu_time, result=result, average=average_meter.average())
                        )

#-------------------------------------------------#
#     开启测试
#-------------------------------------------------#
def test():
    if config["is_train"] == "test":
        test_loader = utils.dataLoader(
                    config["is_train"], 
                    config["root_path"], 
                    config["dataset_names"], 
                    config["max_depth"], 
                    config["sparsifier_names"], 
                    config["num_samples"], 
                    config["modality"], 
                    config["batch_size"], 
                    config["workers"],
                                            )
        checkpoint = torch.load(config["evaluate"])
        model = checkpoint['model']
        model = model.cuda()
        model.eval()

        for i, (input, target, name) in enumerate(test_loader):
            input, target, name = input.cuda(), target.cuda(), name
            with torch.no_grad():
                pred = model(input)

            if config["color_img"]:
                filename = os.path.join(config['save_path'], name[0])
                merge = utils.save_img(filename, input, pred, target)
                merge.show()


            pred = utils.resize_depth(pred, config["input_size"][1], config["input_size"][0])
            # cv2.imshow("pred", pred)
            # cv2.waitKey(0)
            path = os.path.join(config['save_path'], name[0])
            print("writing depth {}..............( {} / {} ) "\
                .format(str(name[0] + ".png"), i + 1, len(test_loader)))
            utils.write_depth(path, pred)

            
             

if __name__ == "__main__":
    # train()
    test()
    
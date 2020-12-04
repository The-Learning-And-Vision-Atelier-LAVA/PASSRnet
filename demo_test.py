from models import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
import argparse
import os
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='data/test')
    parser.add_argument('--dataset', type=str, default='KITTI2012')
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()

def test(test_loader, cfg):
    net = PASSRnet(cfg.scale_factor).to(cfg.device)
    cudnn.benchmark = True
    pretrained_dict = torch.load('./log/x' + str(cfg.scale_factor) + '/PASSRnet_x' + str(cfg.scale_factor) + '.pth')
    net.load_state_dict(pretrained_dict)

    psnr_list = []

    with torch.no_grad():
        for idx_iter, (HR_left, _, LR_left, LR_right) in enumerate(test_loader):
            HR_left, LR_left, LR_right = Variable(HR_left).to(cfg.device), Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)
            scene_name = test_loader.dataset.file_list[idx_iter]

            SR_left = net(LR_left, LR_right, is_training=0)
            SR_left = torch.clamp(SR_left, 0, 1)

            psnr_list.append(cal_psnr(HR_left[:,:,:,64:], SR_left[:,:,:,64:]))

            ## save results
            if not os.path.exists('results/'+cfg.dataset):
                os.mkdir('results/'+cfg.dataset)
            if not os.path.exists('results/'+cfg.dataset+'/'+scene_name):
                os.mkdir('results/'+cfg.dataset+'/'+scene_name)
            SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
            SR_left_img.save('results/'+cfg.dataset+'/'+scene_name+'/img_0.png')

        ## print results
        print(cfg.dataset + ' mean psnr: ', float(np.array(psnr_list).mean()))

def main(cfg):
    test_set = TestSetLoader(dataset_dir=cfg.testset_dir + '/' + cfg.dataset, scale_factor=cfg.scale_factor)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    result = test(test_loader, cfg)
    return result

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)

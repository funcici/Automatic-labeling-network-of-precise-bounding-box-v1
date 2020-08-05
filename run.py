import argparse
import os
from dataset import get_loader
from solver import Solver
def main(config):
    if config.mode == 'train':
        train_loader, dataset = get_loader(config.batch_size, num_thread=config.num_thread)
        run = "nnet"
        if not os.path.exists("%s/run-%s" % (config.save_fold, run)): 
            os.mkdir("%s/run-%s" % (config.save_fold, run))
            os.mkdir("%s/run-%s/logs" % (config.save_fold, run))
            os.mkdir("%s/run-%s/models" % (config.save_fold, run))
        config.save_fold = "%s/run-%s" % (config.save_fold, run)
        train = Solver(train_loader, None, config)
        train.train()
    elif config.mode == 'test':
        test_loader, dataset = get_loader(config.test_batch_size, mode='test',num_thread=config.num_thread, test_mode=config.test_mode, sal_mode=config.sal_mode)
        test = Solver(None, test_loader, config, dataset.save_folder())
        test.test(test_mode=config.test_mode)
    elif config.mode =='application':
        test_loader, dataset = get_loader(config.test_batch_size, mode='test', num_thread=config.num_thread,
                                          test_mode=config.test_mode, sal_mode=config.sal_mode)
        test = Solver(None, test_loader, config, dataset.save_folder())
        test.application()



if __name__ == '__main__':

    vgg_path = './epoch_vgg.pth'
    resnet_path = './resnet50.pth'
    parser = argparse.ArgumentParser()
    parser.add_argument('--longitudeoffset', type=float, default=139.633291)
    parser.add_argument('--latitudeoffset', type=float, default=35.528390)
    parser.add_argument('--pixel', type=float, default=2.68220901489258E-06)
    parser.add_argument('--csv_dir', type=str, default='./file/ff.csv')
    parser.add_argument('--image_big_dir', type=str, default="./file/ff.tif")
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--cuda', type=bool, default=True)
    # Training settings
    parser.add_argument('--vgg', type=str, default=vgg_path)
    parser.add_argument('--resnet', type=str, default=resnet_path)
    parser.add_argument('--epoch', type=int, default=300) # 12, now x3
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--load_bone', type=str, default='')

    parser.add_argument('--save_fold', type=str, default='./EGNet')

    parser.add_argument('--epoch_save', type=int, default=30) # 2, now x3
    parser.add_argument('--epoch_show', type=int, default=1)
    parser.add_argument('--pre_trained', type=str, default=None)


    parser.add_argument('--model', type=str, default='./epoch_resnet240.pth')

    parser.add_argument('--test_fold', type=str, default='D:/EGNet-master/EGNet-master/results')
    parser.add_argument('--test_mode', type=int, default=0)
    parser.add_argument('--sal_mode', type=str, default='t')

    parser.add_argument('--mode', type=str, default='application', choices=['train', 'test','application'])#ff
    parser.add_argument('--visdom', type=bool, default=False)
    config = parser.parse_args()
  
    if not os.path.exists(config.save_fold): os.mkdir(config.save_fold)
    main(config)

import os
import time
import random

gpu = random.randint(0,3)
idx = 0

#### use --tree 'depth2_sub' for depth-3 (standard)
finetune = 20
epoch = 1000


command = "python controller.py --finetune {} --epoch {} --bs 10 --greedy 0.1 --gpu {} --ckpt 'results' --tree 'depth2_sub' --random_step 3 --lr 0.002".format(finetune, epoch, gpu)

os.system(command)
# time.sleep(500)

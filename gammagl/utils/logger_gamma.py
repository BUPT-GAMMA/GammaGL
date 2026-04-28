import os
os.environ['TL_BACKEND'] = 'torch'
from datetime import datetime
import uuid
import json
import copy
from typing import Union, Callable
from dotmap import DotMap

# ===================== 核心替换：PyTorch → TensorLayerX =====================
import tensorlayerx as tlx
# TLX 模型基类替代 torch.nn.Module
from tensorlayerx.nn import Module


def prepare_opt(parser) -> DotMap:
    # 【纯配置逻辑，无修改】配置解析函数
    opt_parser = vars(parser.parse_args())
    config_path = opt_parser['config']
    if not os.path.isfile(config_path):
        config_path = os.path.join('./config/', config_path + '.json')
    with open(config_path, 'r') as config_file:
        opt_config = json.load(config_file)
    for k, v in opt_parser.items():
        if v is not None:
            opt_config[k] = v
    return DotMap(**opt_config)


class Logger(object):
    # 【日志工具，无框架依赖，完全保留】
    def __init__(self, data: str, algo: str, flag_run: str='', dir: tuple=None):
        super(Logger, self).__init__()

        self.seed_str = str(uuid.uuid4())[:6]
        self.seed = int(self.seed_str, 16)
        if not flag_run:
            flag_run = datetime.now().strftime("%m%d") + '-' + self.seed_str
        elif flag_run.count('date') > 0:
            flag_run.replace('date', datetime.now().strftime("%m%d"))
        else:
            pass

        if dir is None:
            self.dir_save = os.path.join("./save/", data, algo, flag_run)
        else:
            self.dir_save = os.path.join(*dir)
        self.path_exists = os.path.exists(self.dir_save)

        self.flag_run = flag_run
        self.file_log = self.path_join('log.txt')
        self.file_config = self.path_join('config.json')

        flag_run = flag_run.split('-')[0]
        seed = int(flag_run) if flag_run.isdigit() else 11
        if seed < 10:
            self.lvl_log = 0
        elif seed < 20:
            self.lvl_log = 1
        elif seed < 30:
            self.lvl_log = 2
        else:
            self.lvl_log = 3
        if seed < 5:
            self.lvl_config = 0
        elif seed < 15:
            self.lvl_config = 1
        elif seed < 25:
            self.lvl_config = 2
        else:
            self.lvl_config = 3

    def path_join(self, *args) -> str:
        return os.path.join(self.dir_save, *args)

    def print(self, s, sf=None, lvl=None) -> None:
        lvl = self.lvl_log if lvl is None else lvl
        if lvl > 0:
            print(s, flush=True)
        if lvl > 2:
            sf = s if sf is None else sf
            with open(self.file_log, 'a') as f:
                f.write(str(sf) + '\n')

    def print_on_top(self, s) -> None:
        if self.lvl_log > 0:
            print(s)
        if self.lvl_log > 2:
            with open(self.file_log, 'a') as f:
                pass
            with open(self.file_log, 'r+') as f:
                temp = f.read()
                f.seek(0, 0)
                f.write(str(s) + '\n')
                f.write(temp)

    def print_header(self, hs, s) -> None:
        if self.lvl_log > 0:
            if os.path.isfile(self.file_log):
                print(hs)
            else:
                self.print(hs, hs.replace('|', ','), lvl=self.lvl_config)
            self.print(s, lvl=self.lvl_config)

    def save_opt(self, opt: DotMap) -> None:
        if self.lvl_log > 2:
            os.makedirs(self.dir_save, exist_ok=True)
            with open(self.file_config, 'w') as f:
                json.dump(opt.toDict(), fp=f, indent=4, sort_keys=False)
                f.write('\n')
            print("Option saved.")
            print("Config path: {}".format(self.file_config))
            print("Option dict: {}\n".format(opt.toDict()))

    def load_opt(self) -> DotMap:
        with open(self.file_config, 'r') as config_file:
            opt = DotMap(json.load(config_file))
        print("Option loaded.")
        print("Config path: {}".format(self.file_config))
        print("Option dict: {}\n".format(opt.toDict()))
        return opt

    def str_csv(self, data, algo, seed, thr_a, thr_w,
               acc_test, conv_epoch, epoch, time_train, macs_train,
               time_test, macs_test, numel_a, numel_w):
        hstr, cstr = '', ''
        hstr += f"      Data|     Model|  Seed|     ThA|     ThW| "
        cstr += f"{data:10s},{algo:10s},{seed:6d},{thr_a:7.2e},{thr_w:7.2e},"
        hstr += f"   Acc|  Cn|  EP| "
        cstr += f"{acc_test:7.5f},{conv_epoch:4d},{epoch:4d},"
        hstr += f" Ttrain|  Ctrain| "
        cstr += f"{time_train:8.4f},{macs_train:8.3f},"
        hstr += f"  Ttest|   CTest|  NumelA|  NumelW"
        cstr += f"{time_test:8.4f},{macs_test:8.4f},{numel_a:8.3f},{numel_w:8.3f}"
        return hstr, cstr

    def str_csvg(self, data, algo, seed, thr_a, thr_w,
               acc_test, conv_epoch, epoch, time_train, macs_train,
               macs_a, macs_wtr, macs_wte,
               time_test, macs_test, numel_a, numel_w, hop, layer, time_pre):
        hstr, cstr = '', ''
        hstr += f"      Data|     Model|  Seed|    ThA|    ThW| "
        cstr += f"{data:10s},{algo:10s},{seed:6d},{thr_a:7.1e},{thr_w:7.1e},"
        hstr += f"   Acc|  Cn|  EP| "
        cstr += f"{acc_test:7.5f},{conv_epoch:4d},{epoch:4d},"
        hstr += f" Ttrain|  Ctrain| "
        cstr += f"{time_train:8.4f},{macs_train:8.3f},"
        hstr += f"  Ttest|   CTest|  NumelA|  NumelW| "
        cstr += f"{time_test:8.4f},{macs_test:8.4f},{numel_a:8.3f},{numel_w:8.3f},"
        hstr += f"   CPre|     CTr|     CTe| Hop| Lay|   TPre "
        cstr += f"{macs_a:8.4f},{macs_wtr:8.4f},{macs_wte:8.4f},{hop:4d},{layer:4d},{time_pre:8.4f}"
        return hstr, cstr

class ModelLogger(object):
    """
    【核心修改：替换 PyTorch 为 TensorLayerX】
    模型保存/加载/最优模型记录，接口完全不变
    """
    def __init__(self, logger: Logger, patience: int=99999,
                 prefix: str='model', storage: str='model_gpu',
                 cmp: Union[Callable[[float, float], bool], str]='>'):
        super(ModelLogger, self).__init__()
        self.logger = logger
        self.patience = patience
        self.prefix = prefix
        self.model = None

        assert storage in ['model', 'state', 'model_ram', 'state_ram', 'model_gpu', 'state_gpu']
        self.storage = storage

        if cmp in ['>', 'max']:
            self.cmp = lambda x, y: x > y
        elif cmp in ['<', 'min']:
            self.cmp = lambda x, y: x < y
        else:
            self.cmp = cmp

    @property
    def state_dict(self):
        # TLX 与 PyTorch 接口一致
        return self.model.state_dict()

    # ===== Load and save
    def __set_model(self, model: Module) -> Module:
        self.model = model
        return self.model

    def register(self, model: Module, save_init: bool=True) -> None:
        self.__set_model(model)
        if save_init:
            self.save('0')

    def load(self, *suffix, model: Module=None, map_location='cpu') -> Module:
        name = '_'.join((self.prefix,) + suffix)
        path = self.logger.path_join(name + '.pth')

        if self.storage == 'state':
            assert self.model is not None
            if model is None:
                model = self.model
            # TLX 加载权重
            state_dict = tlx.load(path, map_location=map_location)
            model.load_state_dict(state_dict)
        elif self.storage in ['state_ram', 'state_gpu']:
            assert self.model is not None
            assert hasattr(self, 'mem')
            if model is None:
                model = self.model
            if hasattr(self.model, 'remove'):
                self.model.remove()
            model.load_state_dict(self.mem)
        elif self.storage == 'model':
            # TLX 加载模型
            model = tlx.load(path, map_location=map_location)
        elif self.storage in ['model_ram', 'model_gpu']:
            model = copy.deepcopy(self.mem)

        return self.__set_model(model)

    def save(self, *suffix) -> None:
        name = '_'.join((self.prefix,) + suffix)
        path = self.logger.path_join(name + '.pth')

        if self.storage == 'state':
            # TLX 保存权重
            tlx.save(self.state_dict, path)
        elif self.storage == 'model':
            # TLX 保存模型
            tlx.save(self.model, path)
        elif self.storage == 'state_gpu':
            if hasattr(self, 'mem'): del self.mem
            self.mem = copy.deepcopy(self.state_dict)
        elif self.storage == 'state_ram':
            if hasattr(self, 'mem'): del self.mem
            self.mem = copy.deepcopy(self.state_dict)
            self.mem = {k: v.cpu() for k, v in self.mem.items()}
        elif self.storage == 'model_gpu':
            if hasattr(self, 'mem'): del self.mem
            self.mem = copy.deepcopy(self.model)
        elif self.storage == 'model_ram':
            if hasattr(self, 'mem'): del self.mem
            device = next(self.model.parameters()).device
            self.mem = copy.deepcopy(self.model.cpu())
            self.model.to(device)

    def get_last_epoch(self) -> int:
        name_pre = '_'.join((self.prefix,) + ('',))
        last_epoch = -2

        for fname in os.listdir(self.logger.dir_save):
            fname = str(fname)
            if fname.startswith(name_pre) and fname.endswith('.pth'):
                suffix = fname.replace(name_pre, '').replace('.pth', '')
                if suffix == 'init':
                    this_epoch = -1
                elif suffix.isdigit():
                    this_epoch = int(suffix) - 1
                else:
                    this_epoch = -2
                if this_epoch > last_epoch:
                    last_epoch = this_epoch
        return last_epoch

    # ===== Save during training
    def save_epoch(self, epoch: int, period: int=1) -> None:
        if (epoch + 1) % period == 0:
            self.save(str(epoch+1))

    def save_best(self, score: float, epoch: int=-1,
                  print_log: bool=False) -> int:
        if self.is_best(score, epoch):
            self.save('best')
            if print_log:
                self.logger.print('[best saved] {:>.4f}'.format(self.score_best))
        return self.score_best

    def is_best(self, score: float, epoch: int=-1) -> bool:
        res = (not hasattr(self, 'score_best'))
        if res or self.cmp(score, self.score_best):
            self.score_best = score
            self.epoch_best = epoch
            res = True
        return res

    def is_early_stop(self, epoch: int=-1) -> bool:
        return epoch - self.epoch_best >= self.patience


class LayerNumLogger(object):
    # 【纯计数工具，无修改】
    def __init__(self, name: str=None):
        self.name = name
        self.numel_before = None
        self.numel_after = None

    @property
    def ratio(self) -> float:
        return self.numel_after / self.numel_before

    def __str__(self) -> str:
        s = f"{self.numel_after}/{self.numel_before} ({1-self.ratio:6.2%})"
        return s
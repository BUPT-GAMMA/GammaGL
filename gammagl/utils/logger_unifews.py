import os
from datetime import datetime
import uuid
import json
import numpy as np
from typing import Union, Callable
from dotmap import DotMap

import tensorlayerx as tlx

from tensorlayerx.nn import Module


def prepare_opt(parser) -> DotMap:

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

    def _opt_to_dict(self, opt):
        if isinstance(opt, DotMap):
            return opt.toDict()
        return vars(opt)

    def save_opt(self, opt) -> None:
        if self.lvl_log > 2:
            os.makedirs(self.dir_save, exist_ok=True)
            opt_dict = self._opt_to_dict(opt)
            with open(self.file_config, 'w') as f:
                json.dump(opt_dict, fp=f, indent=4, sort_keys=False)
                f.write('\n')
            print("Option saved.")
            print("Config path: {}".format(self.file_config))
            print("Option dict: {}\n".format(opt_dict))

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
    def __init__(self, logger: Logger, patience: int=99999,
                 prefix: str='model', storage: str='state',
                 cmp: Union[Callable[[float, float], bool], str]='>'):
        super(ModelLogger, self).__init__()
        self.logger = logger
        self.patience = patience
        self.prefix = prefix
        self.model = None

        assert storage in ['model', 'state', 'state_gpu']
        self.storage = storage

        if cmp in ['>', 'max']:
            self.cmp = lambda x, y: x > y
        elif cmp in ['<', 'min']:
            self.cmp = lambda x, y: x < y
        else:
            self.cmp = cmp

    def __set_model(self, model: Module) -> Module:
        self.model = model
        return self.model

    def register(self, model: Module, save_init: bool=True) -> None:
        self.__set_model(model)
        if save_init:
            self.save('0')

    def load(self, *suffix, model: Module=None) -> Module:
        name = '_'.join((self.prefix,) + suffix)
        path = self.logger.path_join(name + '.npz')

        if self.storage in ('state', 'state_gpu'):
            if model is None:
                model = self.model
            loaded = np.load(path)
            for i, w in enumerate(model.trainable_weights):
                new_val = tlx.convert_to_tensor(loaded[f'arr_{i}'], dtype=w.dtype)
                if hasattr(w, 'assign'):
                    w.assign(new_val)
                else:
                    w.data = new_val
        elif self.storage == 'model':
            model = tlx.load(path)

        return self.__set_model(model)

    def save(self, *suffix) -> None:
        name = '_'.join((self.prefix,) + suffix)
        path = self.logger.path_join(name + '.npz')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if self.storage in ('state', 'state_gpu'):
            weights = [tlx.convert_to_numpy(w) for w in self.model.trainable_weights]
            np.savez(path, *weights)
        elif self.storage == 'model':
            tlx.save(self.model, path)

    def get_last_epoch(self) -> int:
        name_pre = '_'.join((self.prefix,) + ('',))
        last_epoch = -2

        for fname in os.listdir(self.logger.dir_save):
            fname = str(fname)
            if fname.startswith(name_pre) and fname.endswith('.npz'):
                suffix = fname.replace(name_pre, '').replace('.npz', '')
                if suffix == 'init':
                    this_epoch = -1
                elif suffix.isdigit():
                    this_epoch = int(suffix) - 1
                else:
                    this_epoch = -2
                if this_epoch > last_epoch:
                    last_epoch = this_epoch
        return last_epoch

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
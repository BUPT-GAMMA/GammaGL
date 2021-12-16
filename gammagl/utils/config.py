class Config:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            self.__setattr__(k, v)
    
    def __setattr__(self, k,v):
        self.__dict__[k]=v

    def self_check(self, dic):
        for k, v in dic.items():
            if not hasattr(self, k) or not isinstance(getattr(self,k), v):
                raise Exception("Config obj not configurate `{}` properly!".format(k))

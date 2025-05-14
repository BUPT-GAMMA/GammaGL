
class DDict(dict):
    def __dir__(self):
        return super().__dir__() + list(self.keys())
    def __setattr__(self, key, value):
        if key in self.keys():
            self[key]=value
        else:
            raise AttributeError('No such attribute: '+key)
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError('No such attribute: '+key)
    def copy(self):
        return self.__class__(self)

class Assignable:
    def __setattr__(self, key, value):
        self[key]=value


class Lambda(DDict):
    _globals={}
    @classmethod
    def update_globals(cls, list_or_dict):
        dictionary = dict((x.__name__,x) 
                              for x in list_or_dict)\
                                if isinstance(list_or_dict, list)\
                                  else list_or_dict
        cls._globals.update(dictionary)
        
    def __init__(self, func):
        super().__init__(tag='<Lambda>', func=func)
    def __call__(self, base):
        lfunc = eval('lambda '+self.func, self._globals)
        return lfunc(base)


class MDict(DDict):
    L = Lambda
    
    def call_macro(self, macro):
        return macro(self)
    
    def is_macro(self, value):
        return callable(value)
        
    def __getattr__(self, key):
        value = super().__getattr__(key)
        if self.is_macro(value):
            value = self.call_macro(value)
        return value
    
    def get_dict(self):
        ret = {}
        for key, value in self.items():
            if self.is_macro(value):
                ret[key] = self.call_macro(value)
            else:
                ret[key] = value
        return ret


class Inherit(DDict):
    def __init__(self, key, default=None, max_nest=None):
        super().__init__(tag='<Inherit>',key=key, default=default, max_nest=max_nest)
    
    def __call__(self, base):
        parent = base
        val = self.default
        for _ in range(self.max_nest 
                           if self.max_nest is not None 
                                else 1000):
            parent = parent.get_parent()
            if parent is None: break
            
            try:
                val = getattr(parent, self.key)
                break
            except AttributeError:
                pass
            
        return val


class HDict(MDict):
    I = Inherit
    
    def set_parent(self, parent):
        self.__dict__['_parent'] = parent
        return self
    
    def get_parent(self):
        return self.__dict__.get('_parent', None)
    
    def __call__(self, parent):
        return self.set_parent(parent)
    
    def get_dict(self):
        ret = super().get_dict()
        for key, value in ret.items():
            if isinstance(value, HDict):
                ret.update({key:value.get_dict()})
        return ret
            



class ADDict(Assignable,DDict):
    pass

class AMDict(Assignable,MDict):
    pass

class AHDict(Assignable,HDict):
    pass



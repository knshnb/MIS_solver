import time

class Timer:
    logs = {}
    ss = {}
    # 複数スレッドで使用するときはactiveをFalseにする！
    active = True

    @staticmethod
    def disable():
        Timer.active = False

    @staticmethod
    def start(key):
        if not Timer.active: return
        Timer.ss[key] = time.time()

    @staticmethod
    def end(key):
        if not Timer.active: return
        assert key in Timer.ss
        s = Timer.ss[key]
        Timer.ss.pop(key)
        e = time.time()
        arr = Timer.logs.get(key, [])
        arr.append(e - s)
        Timer.logs[key] = arr
    
    @staticmethod
    def print():
        if not Timer.active: return
        for key in Timer.logs:
            print("{}: {:.2f}sec".format(key, sum(Timer.logs[key])))

import time

class Timer:
    @staticmethod
    def init():
        Timer.logs = {}
        Timer.ss = {}

    @staticmethod
    def start(key):
        Timer.ss[key] = time.time()

    @staticmethod
    def end(key):
        assert key in Timer.ss
        s = Timer.ss[key]
        Timer.ss.pop(key)
        e = time.time()
        arr = Timer.logs.get(key, [])
        arr.append(e - s)
        Timer.logs[key] = arr
    
    @staticmethod
    def print():
        for key in Timer.logs:
            print("{}: {:.2}sec".format(key, sum(Timer.logs[key])))

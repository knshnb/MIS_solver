class Counter:
    cnt = {}

    @staticmethod
    def count(key):
        Counter.cnt[key] = Counter.cnt.get(key, 0) + 1

    @staticmethod
    def print():
        for key in Counter.cnt:
            print("{}: {}times".format(key, Counter.cnt[key]))

class Counter:
    cnt = {}
    # set active False in multithreading
    active = True

    @staticmethod
    def disable():
        Counter.active = False

    @staticmethod
    def count(key):
        if not Counter.active:
            return
        Counter.cnt[key] = Counter.cnt.get(key, 0) + 1

    @staticmethod
    def print():
        if not Counter.active:
            return
        for key in Counter.cnt:
            print("{}: {}times".format(key, Counter.cnt[key]))

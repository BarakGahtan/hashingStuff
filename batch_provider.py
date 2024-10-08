"""Batch provider. Returns iterator to batches"""

from PIL import Image
from random import shuffle
import matplotlib.pyplot as plt
from scipy import misc
import random
import numpy as np
import lmdb
import pickle
try:
    import queue
except ImportError:
    import Queue as queue
from threading import Thread, Lock, Event
import logging
from PIL import Image
try:
    from BytesIO import BytesIO
except ImportError:
    from io import BytesIO


class BatchProvider:
    """All in memory batch provider for small datasets that fit RAM"""
    def __init__(self, batch_size, items, cycled=True, worker=16, width=224, height=224, lmdb_file=None):
        self.items = items
        shuffle(self.items)
        self.batch_size = batch_size

        self.current_batch = 0
        self.cycled = cycled
        if self.cycled:
            worker = 1
        self.done = False
        self.image_size = (width, height)
        self.lock = Lock()
        self.worker = worker
        self.quit_event = Event()

        self.q = queue.Queue(16)
        self.batches_n = len(self.items)//self.batch_size
        logging.debug("Batches per epoch: {0}", self.batches_n)

        try:
            self.using_lmdb = type(items[0][1]) is str or type(items[0][1]) is unicode
        except:
            self.using_lmdb = type(items[0][1]) is str

        if self.using_lmdb:
            assert(lmdb_file)
            self.env = lmdb.open(lmdb_file, map_size=8 * 1024 * 1024 * 1024, subdir=True, readonly=True, lock=False)


    def get_batches(self):
        workers = []
        for i in range(self.worker):
            worker = Thread(target=self._worker)
            worker.setDaemon(True)
            worker.start()
            workers.append(worker)
        try:
            while True:
                yield self._get_batch()

        except GeneratorExit:
            self.quit_event.set()
            self.done = True
            while not self.q.empty():
                try:
                    self.q.get(False)
                except queue.Empty:
                    continue
                self.q.task_done()

    def _worker(self):
        while not (self.quit_event.is_set() and self.done):
            b = self.__next()
            if b is None:
                break
            self.q.put(b)

    def _get_batch(self):
        if self.q.empty() and self.done:
            return None
        item = self.q.get()
        self.q.task_done()
        return item

    def __next(self):
        self.lock.acquire()
        if self.current_batch == self.batches_n:
            self.done = True
            if self.cycled:
                self.done = False
                self.current_batch = 0
                shuffled = list(self.items)
                shuffle(shuffled)
                self.items = shuffled
            else:
                self.lock.release()
                return None
        cb = self.current_batch
        self.current_batch += 1
        items = self.items
        self.lock.release()

        b_images = []
        b_labels = []

        buffer = BytesIO()
        for i in range(self.batch_size):
            item = items[cb * self.batch_size + i]

            if not self.using_lmdb and len(item[1].shape) == 1:
                image = item[1]
            elif not self.using_lmdb:
                image = Image.fromarray(item[1])
                image = image.resize(self.image_size, Image.BILINEAR)

                # Similar to DVSQ https://github.com/caoyue10/cvpr17-dvsq/blob/master/net.py#L122
                if self.cycled:
                    if random.random() > 0.5:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)
                image = np.array(image)
            else:
                with self.env.begin() as txn:
                    buf = txn.get(item[1].encode('ascii'))
                    if buf is None:
                        print(item[1].encode('ascii'))
                    buffer.seek(0)
                    buffer.write(buf)
                    buffer.seek(0)
                    image = Image.open(buffer).convert('RGB')

                startx = image.size[0] - self.image_size[0]
                starty = image.size[1] - self.image_size[1]
                if self.cycled:
                    startx = random.randint(0, startx)
                    starty = random.randint(0, starty)
                else:
                    startx = startx // 2
                    starty = starty // 2
                image = image.crop((startx, starty, startx + self.image_size[0], starty + self.image_size[1]))

                # Similar to DVSQ https://github.com/caoyue10/cvpr17-dvsq/blob/master/net.py#L122
                if self.cycled:
                    if random.random() > 0.5:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)
                image = np.array(image)

            b_images.append(image)
            b_labels.append([item[0]])
        feed_dict = {"images": b_images, "labels": b_labels}

        return feed_dict


# For testing
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils.cifar10_reader import Reader

    r = Reader('data/cifar10/cifar-10-batches-bin')

    p = BatchProvider(20, r.items)
    # with open('temp/items_train_nuswide_5000.10000.pkl', 'rb') as pkl:
    #     p = BatchProvider(20, pickle.load(pkl))

    b = p.get_batches()

    ims = next(b)["images"]
    for im in ims:
        plt.imshow(im, interpolation='nearest')
        plt.show()

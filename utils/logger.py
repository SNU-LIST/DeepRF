import os
import sys
from pathlib import Path
from datetime import datetime
import traceback

import logging
from subprocess import Popen, PIPE
from termcolor import colored
from tensorboardX import SummaryWriter
from tensorboardX.summary import Summary
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from scipy.io import savemat
import torch
from PIL import Image
import csv

from settings import PROJECT_ROOT, LOG_LEVELS, LOG_DIR


def encode_gif(images, fps=30):
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-r', '%.02f' % fps,
        '-s', '%dx%d' % (images[0].shape[1], images[0].shape[0]),
        '-pix_fmt', 'rgb24',
        '-i', '-',
        '-filter_complex',
        '[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
        '-r', '%.02f' % fps,
        '-f', 'gif',
        '-'
    ]
    proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in images:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        err = '\n'.join([' '.join(cmd), err.decode('utf8')])
        raise IOError(err)
    del proc
    return out


class Logger:
    def __init__(self, name, args=None, log_dir=None):
        self.args = args
        if log_dir is None:
            self.log_dir = os.path.join(
                os.path.join(PROJECT_ROOT, LOG_DIR, self.args.tag),
                datetime.now().strftime("%Y%m%d%H%M%S")
            )
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        else:
            self.log_dir = log_dir

        logger = logging.getLogger(name)
        if not logger.handlers:
            format = logging.Formatter(
                "[%(name)s|%(levelname)s] %(asctime)s > %(message)s"
            )
            streamHandler = logging.StreamHandler()
            streamHandler.setFormatter(format)
            logger.addHandler(streamHandler)
            logger.setLevel(args.log_level)

            filename = os.path.join(self.log_dir, name + '.txt')
            fileHandler = logging.FileHandler(filename, mode="w")
            fileHandler.setFormatter(format)
            logger.addHandler(fileHandler)

        self.logger = logger
        self.writer = SummaryWriter(self.log_dir)
        sys.excepthook = self.excepthook

    def log(self, msg, lvl="INFO"):
        lvl, color = self.get_level_color(lvl)
        self.logger.log(lvl, colored(msg, color))

    def add_level(self, name, lvl, color='white'):
        if name not in LOG_LEVELS.keys() and lvl not in LOG_LEVELS.values():
            LOG_LEVELS[name] = {'lvl': lvl, 'color': color}
            logging.addLevelName(lvl, name)
        else:
            raise AssertionError("log level already exists")

    def get_level_color(self, lvl):
        assert isinstance(lvl, str)
        lvl_num = LOG_LEVELS[lvl]['lvl']
        color = LOG_LEVELS[lvl]['color']
        return lvl_num, color

    def excepthook(self, type_, value_, traceback_):
        e = "{}: {}".format(type_.__name__, value_)
        tb = "".join(traceback.format_exception(type_, value_, traceback_))
        self.log(e, "ERROR")
        self.log(tb, "DEBUG")

    def scalar_summary(self, info, step, lvl="INFO", tag='values'):
        assert isinstance(info, dict), "data must be a dictionary"
        # flush to terminal
        if self.args.log_level <= LOG_LEVELS[lvl]['lvl']:
            key2str = {}
            for key, val in info.items():
                if isinstance(val, float):
                    valstr = "%-8.3g" % (val,)
                else:
                    valstr = str(val)
                key2str[self._truncate(key)] = self._truncate(valstr)

            if len(key2str) == 0:
                self.log("empty key-value dict", 'WARNING')
                return

            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

            dashes = '  ' + '-'*(keywidth + valwidth + 7)
            lines = [dashes]
            for key, val in key2str.items():
                lines.append('  | %s%s | %s%s |' % (
                    key,
                    ' '*(keywidth - len(key)),
                    val,
                    ' '*(valwidth - len(val))
                ))
            lines.append(dashes)
            print('\n'.join(lines))

        # flush to csv
        if self.log_dir is not None:
            filepath = Path(os.path.join(self.log_dir, tag + '.csv'))
            if not filepath.is_file():
                with open(filepath, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['step'] + list(info.keys()))

            with open(filepath, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([step] + list(info.values()))

        # flush to tensorboard
        if self.writer is not None:
            for k, v in info.items():
                self.writer.add_scalar(k, v, step)

    def image_summary(self, image, step, tag='figure'):
        path = os.path.join(self.log_dir, 'images')
        if '/' in tag:
            path = os.path.join(path, tag[:tag.find('/')])
        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(self.log_dir, 'images')
        path = os.path.join(path, '{}_{}.png'.format(tag, step))

        # save matplotlib figure to image
        if isinstance(image, Figure):
            image.savefig(path)
            plt.close(image)
        # save PIL image
        else:
            image.save(path)

        # flush to tensorboard
        if self.writer is not None:
            image = Image.open(path).convert('RGB')
            pix = np.array(image)
            pix = np.transpose(pix, (2, 0, 1))
            self.writer.add_image(tag, pix, step)

    def video_summary(self, images, step, tag='playback', fps=30):
        path = os.path.join(self.log_dir, 'videos')
        if not os.path.exists(path):
            os.makedirs(path)
        string_encode = encode_gif(images, fps=fps)
        path = os.path.join(path, '{}_{}.gif'.format(tag, step))

        # save gif image
        with open(path, 'wb') as f:
            f.write(string_encode)

        # flush to tensorboard
        if self.writer is not None:
            _, h, w, c = images.shape
            video = Summary.Image(
                height=h,
                width=w,
                colorspace=c,
                encoded_image_string=string_encode
            )
            self.writer._get_file_writer().add_summary(
                Summary(value=[Summary.Value(tag=tag, image=video)]),
                step,
                walltime=None
            )

    def savemat(self, filename, array_dict):
        assert isinstance(array_dict, dict)
        for k, v in array_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            assert isinstance(v, np.ndarray)
            array_dict[k] = v

        path = os.path.join(self.log_dir, 'arrays')
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, filename)
        if filename[-4:] == '.mat':
            savemat(path, array_dict)
        else:
            savemat(path + '.mat', array_dict)

    def _truncate(self, s):
        return s[:20] + '...' if len(s) > 23 else s

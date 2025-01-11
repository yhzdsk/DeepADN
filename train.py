import argparse
import functools

from trainer import MAClsTrainer
from utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    '/path/to/configs.yml',        )
add_arg("local_rank",       int,    0,                          )
add_arg("use_gpu",          bool,   True,                      )
add_arg('save_model_path',  str,    '/path/to/model/',              )
add_arg('resume_model',     str,    None,                       )
add_arg('pretrained_model', str,    None,                    )
args = parser.parse_args()
print_arguments(args=args)

trainer = MAClsTrainer(configs=args.configs, use_gpu=args.use_gpu)

trainer.train(save_model_path=args.save_model_path,
              resume_model=args.resume_model,
              pretrained_model=args.pretrained_model)


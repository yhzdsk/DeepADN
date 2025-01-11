# coding: gbk
import argparse
import functools
import os
import shutil
import time  # Import time module for timing

from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments


def classify_audio_files(audio_dir, output_dir, predictor):
    total_inference_time = 0
    file_count = 0

    for root, dirs, files in os.walk(audio_dir):
        for audio_file_name in files:
            audio_path = os.path.join(root, audio_file_name)
            
            # Measure inference time
            start_time = time.time()
            label, score = predictor.predict(audio_data=audio_path)
            inference_time = time.time() - start_time

            total_inference_time += inference_time
            file_count += 1

            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            output_path = os.path.join(label_dir, audio_file_name)
            shutil.copy(audio_path, output_path)

            # Print inference time and prediction results
            #print(f'audio：{audio_path} label：{label}，score：{score}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('configs',          str,    'configs.yml',  )
    add_arg('use_gpu',          bool,   True,                 )
    add_arg('audio_dir',        str,    '/path/to/test/', )
    add_arg('model_path',       str,    ''/path/to/model/, '')
    add_arg('output_dir',       str,    '/path/to/out/', '')
    args = parser.parse_args()
    print_arguments(args=args)

    # Load model configuration and ensure class count is correct
    predictor = MAClsPredictor(configs=args.configs,
                               model_path=args.model_path,
                               use_gpu=args.use_gpu)

    classify_audio_files(args.audio_dir, args.output_dir, predictor)

from pathlib import Path

ROOT_DIR = Path(__file__).parent
CONFIG_DIR = ROOT_DIR.joinpath('config_file')

import numpy as np
import torch
import yaml
from models.generator import OcclusionAwareGenerator
from models.keypoint_detector import KPDetector
import argparse
from models.util import draw_annotation_box
from models.transformer import Audio2kpTransformer
from tools.interface import (
    prediction,
    get_sequences,
    get_tp,
    save_video,
    read_img,
    read_audio,
    get_pose_from_audio,
    get_audio_feature_from_audio,
    parse_phoneme_file,
    load_ckpt
)

# import config
with open(CONFIG_DIR.joinpath('vox-256.yaml').absoulte()) as f:
    config = yaml.full_load(f)

### main

def test_with_input_audio_and_image(
    img_path,
    audio_path,
    phs,
    generator_ckpt,
    audio2pose_ckpt,
    save_dir="samples/results",
):

    # cur_path = os.getcwd() # TODO will be deleted
    with open(CONFIG_DIR.joinpath("audio2kp.yaml").absolute()) as f:  # audio2kp open
        tmp = yaml.full_load(f)
    
    opt = argparse.Namespace(**tmp)
    # image_file, audio_file read
    img = read_img(img_path).cuda()
    temp_audio = read_audio(audio_path=audio_path)

    tp = get_tp(img_path=img_path)
    audio_feature = get_audio_feature_from_audio(temp_audio) # a_{1:T} # shape [time, power]
    
    # N frames
    frames = len(audio_feature) // 4
    frames = min(frames, len(phs["phone_list"]))

    # 머리 포즈 전체 시퀀스
    ref_pose = get_pose_from_audio(tp, audio_feature, audio2pose_ckpt) # h_{1:T}
    torch.cuda.empty_cache()

    # sequences data
    trans_seq = ref_pose[:, 3:] # 프레임 별 transform vector 변화, 3D -> 2D projection
    rot_seq = ref_pose[:, :3] # 프레임 별 rotaion vector 변화, 3D -> 2D projection
    audio_seq = audio_feature  # [40:] # [mfcc, logbank, interpitch(pyworld.harvest)]
    ph_seq = phs["phone_list"]
    audio_f, poses, ph_frames = get_sequences(frames=frames,
                                              trans_seq=trans_seq,
                                              rot_seq=rot_seq,
                                              audio_seq=audio_seq,
                                              ph_seq=ph_seq,
                                              opt=opt)
    
    bs = audio_f.shape[1] # batch size

    # eval 모드
    kp_detector = KPDetector(
        **config["model_params"]["kp_detector_params"],
        **config["model_params"]["common_params"]
    ) # keypoint detector
    
    # 렌더러
    generator = OcclusionAwareGenerator(
        **config["model_params"]["generator_params"],
        **config["model_params"]["common_params"]
    )
    
    kp_detector = kp_detector.cuda()
    generator = generator.cuda()

    ph2kp = Audio2kpTransformer(opt).cuda() # Audio-visual Correlation Transformer

    load_ckpt(generator_ckpt, kp_detector=kp_detector, generator=generator, ph2kp=ph2kp) # 체크포인트

    ph2kp.eval()
    generator.eval()
    kp_detector.eval()

    # 오디오에 따른 머리 예측 시행
    predictions_gen = prediction(bs=bs,
                                 audio_f=audio_f,
                                 poses=poses,
                                 ph_frames=ph_frames,
                                 img=img,
                                 kp_detector=kp_detector,
                                 ph2kp=ph2kp,
                                 generator=generator)

    save_video(save_dir=save_dir,
               img_path=img_path,
               audio_path=audio_path,
               predictions_gen=predictions_gen)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--img_path",
        type=str,
        default='./samples/imgs/d1.jpg',
        help="path of the input image ( .jpg ), preprocessed by image_preprocess.py",
    )
    argparser.add_argument(
        "--audio_path", type=str, default='./samples/audios/abstract.wav', help="path of the input audio ( .wav )"
    )
    argparser.add_argument(
        "--phoneme_path",
        type=str,
        default='./samples/phonemes/abstract.json',
        help="path of the input phoneme. It should be note that the phoneme must be consistent with the input audio",
    )
    argparser.add_argument(
        "--save_dir",
        type=str,
        default="./samples/results",
        help="path of the output video",
    )
    args = argparser.parse_args()

    phoneme = parse_phoneme_file(args.phoneme_path)
    test_with_input_audio_and_image(
        args.img_path,
        args.audio_path,
        phoneme,
        config["GENERATOR_CKPT"],
        config["AUDIO2POSE_CKPT"],
        args.save_dir,
    )

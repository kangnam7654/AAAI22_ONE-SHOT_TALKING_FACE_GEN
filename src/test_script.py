from pathlib import Path

ROOT_DIR = Path(__file__).parent
CONFIG_DIR = ROOT_DIR.joinpath('config_file')

import torch
import yaml
from models.generator import OcclusionAwareGenerator
from models.keypoint_detector import KPDetector
import argparse
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
with open(CONFIG_DIR.joinpath('vox-256.yaml').absolute()) as f:
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

    # Raw data read
    img = read_img(img_path).cuda() # 레퍼런스 이미지 [C, H, W]
    temp_audio = read_audio(audio_path=audio_path) # 오디오, 샘플레이트 16k

    # 처리 시작
    tp = get_tp(img_path=img_path) # 3차원 정보를 2차원으로 전사(projection)한 바이너리 이미지
    audio_feature = get_audio_feature_from_audio(temp_audio) # a_{1:T} # shape [time, power]

    # N frames
    frames = len(audio_feature) // 4
    frames = min(frames, len(phs["phone_list"]))

    # tp, 오디오를 이용하여 전체 시퀀스에 대한 레퍼런스 포즈(의미론적 이미지가 아님)를 추출
    ref_pose = get_pose_from_audio(tp, audio_feature, audio2pose_ckpt) # h_{1:T}
    torch.cuda.empty_cache()

    # 
    trans_seq = ref_pose[:, 3:] # 움직임에 대한 정보 시퀀스
    rot_seq = ref_pose[:, :3] # 회전에 대한 정보 시퀀스
    audio_seq = audio_feature  # [40:] # [mfcc, logbank, interpitch(pyworld.harvest)]
    ph_seq = phs["phone_list"] # 음소에 대한 시퀀스

    audio_f, poses, ph_frames = get_sequences(frames=frames,
                                              trans_seq=trans_seq,
                                              rot_seq=rot_seq,
                                              audio_seq=audio_seq,
                                              ph_seq=ph_seq,
                                              opt=opt)
    # 전체 프레임의 시간축에 정렬한 audio, poses, ph가 출력값으로 나옴
    # 전처리 끝
    
    bs = audio_f.shape[1] # batch size
    kp_detector = KPDetector(
        **config["model_params"]["kp_detector_params"],
        **config["model_params"]["common_params"]
    )
    generator = OcclusionAwareGenerator(
        **config["model_params"]["generator_params"],
        **config["model_params"]["common_params"]
    ) # 렌더러
    
    kp_detector = kp_detector.cuda()
    generator = generator.cuda()

    ph2kp = Audio2kpTransformer(opt).cuda() # Audio-visual Correlation Transformer

    load_ckpt(generator_ckpt, kp_detector=kp_detector, generator=generator, ph2kp=ph2kp) # 체크포인트

    ph2kp.eval()
    generator.eval()
    kp_detector.eval()

    # 전체 프레임에 대한 머리 움직임의 시퀀스 생성
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
        default=ROOT_DIR.joinpath('samples', 'imgs' ,'d2.jpg').absolute(),
        help="path of the input image ( .jpg ), preprocessed by image_preprocess.py",
    )
    argparser.add_argument(
        "--audio_path", type=str, default=ROOT_DIR.joinpath('samples', 'audios' ,'trump.wav').absolute(), help="path of the input audio ( .wav )"
    )
    argparser.add_argument(
        "--phoneme_path",
        type=str,
        default=ROOT_DIR.joinpath('samples', 'phonemes' ,'trump.json').absolute(),
        help="path of the input phoneme. It should be note that the phoneme must be consistent with the input audio",
    )
    argparser.add_argument(
        "--save_dir",
        type=str,
        default=ROOT_DIR.joinpath('samples', 'results').absolute(),
        help="path of the output video",
    )
    args = argparser.parse_args()

    phoneme = parse_phoneme_file(args.phoneme_path) # Parsing Phoneme
    
    test_with_input_audio_and_image(
        args.img_path,
        args.audio_path,
        phoneme,
        config["GENERATOR_CKPT"],
        config["AUDIO2POSE_CKPT"],
        args.save_dir,
    )

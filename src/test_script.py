import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent
CONFIG_DIR = ROOT_DIR.joinpath('config_file')

import numpy as np
import torch
import yaml
from models.generator import OcclusionAwareGenerator
from models.keypoint_detector import KPDetector
import argparse
import imageio
from models.util import draw_annotation_box
from models.transformer import Audio2kpTransformer
from scipy.io import wavfile
from tools.interface import (
    read_img,
    get_img_pose,
    get_pose_from_audio,
    get_audio_feature_from_audio,
    parse_phoneme_file,
    load_ckpt,
    normalize_kp
)

# import config
with open(CONFIG_DIR.joinpath('vox-256.yaml').absoulte()) as f:
    config = yaml.full_load(f)


def save_video(save_dir, img_path, audio_path, predictions_gen):
    log_dir = save_dir
    os.makedirs(os.path.join(log_dir, "temp"), exist_ok=True)

    f_name = (
        os.path.basename(img_path)[:-4]
        + "_"
        + os.path.basename(audio_path)[:-4]
        + ".mp4"
    )
    # kwargs = {'duration': 1. / 25.0}
    video_path = os.path.join(log_dir, "temp", f_name)
    print("save video to: ", video_path)
    imageio.mimsave(video_path, predictions_gen, fps=25.0) # video save, fps 25고정

    # audio_path = os.path.join(audio_dir, x['name'][0].replace(".mp4", ".wav"))
    save_video_path = os.path.join(log_dir, f_name)
    cmd = r'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s"' % (
        video_path,
        audio_path,
        save_video_path,
    )
    os.system(cmd)
    os.remove(video_path)


def prediction(bs, audio_f, poses, ph_frames, img, kp_detector, ph2kp, generator):
    predictions_gen = [] # result image sequence
    with torch.no_grad():
        for frame_idx in range(bs):
            inputs = {}

            inputs["audio"] = audio_f[:, frame_idx].cuda() # a_i
            inputs["pose"] = poses[:, frame_idx].cuda() # h_i
            inputs["ph"] = ph_frames[:, frame_idx].cuda() # p_i
            inputs["id_img"] = img # f^r
            # inputs = {audio, pose, ph, id_image}

            initial_keypoint = kp_detector(img, True) # keypoint of image

            gen_kp = ph2kp(inputs, initial_keypoint) # input (x, initial kp), out = motionfiled
            if frame_idx == 0:
                drive_first = gen_kp

            norm = normalize_kp(
                kp_source=initial_keypoint,
                kp_driving=gen_kp,
                kp_driving_initial=drive_first,
            ) # 2D Projection image
            out_gen = generator(img, kp_source=initial_keypoint, kp_driving=norm) # Renderer, 레퍼런스 이미지 + Motion field 이용하여 렌더링

            predictions_gen.append(
                (
                    np.transpose(
                        out_gen["prediction"].data.cpu().numpy(), [0, 2, 3, 1]
                    )[0]
                    * 255
                ).astype(np.uint8) # [sequence, height, width, colour channel] permutation
            )
    return predictions_gen


### audio
def audio_read(audio_path):
    sr, _ = wavfile.read(audio_path)  # sample rate, data
    
    if sr != 16000: # change sample rate 16k
        temp_audio = ROOT_DIR.joinpath("samples", "temp.wav")
        command = (
            "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s"
            % (audio_path, temp_audio)
        )
        os.system(command)
    else:
        temp_audio = audio_path
    return temp_audio


def get_sequences(frames, trans_seq, rot_seq, audio_seq, ph_seq, opt):
    ph_frames = [] # p_{1:T} # phoneme(음소)
    audio_frames = [] # a_{1:T}
    pose_frames = [] # h_{1:T}

    pad = np.zeros((4, audio_seq.shape[1]), dtype=np.float32) # [4, logbank shape]
    name_len = frames
    
    for rid in range(0, frames):
        ph = []
        audio = [] 
        pose = []
        for i in range(rid - opt.num_w, rid + opt.num_w + 1):
            if i < 0:
                rot = rot_seq[0] # rotate sequence
                trans = trans_seq[0] # transform sequence
                ph.append(31) # phoneme # 31 = SIL = silence
                audio.append(pad)
            elif i >= name_len:
                ph.append(31) # phoneme # 31 = SIL = silence
                rot = rot_seq[name_len - 1]
                trans = trans_seq[name_len - 1]
                audio.append(pad)
            else:
                ph.append(ph_seq[i])
                rot = rot_seq[i]
                trans = trans_seq[i]
                audio.append(audio_seq[i * 4 : i * 4 + 4])
            tmp_pose = np.zeros([256, 256])
            draw_annotation_box(tmp_pose, np.array(rot), np.array(trans))
            pose.append(tmp_pose)

        ph_frames.append(ph)
        audio_frames.append(audio)
        pose_frames.append(pose)

    audio_f = torch.from_numpy(np.array(audio_frames, dtype=np.float32)).unsqueeze(0)
    poses = torch.from_numpy(np.array(pose_frames, dtype=np.float32)).unsqueeze(0)
    ph_frames = torch.from_numpy(np.array(ph_frames)).unsqueeze(0)
    return audio_f, poses, ph_frames
    ### preprocess end (전처리 끝)


def test_with_input_audio_and_image(
    img_path,
    audio_path,
    phs,
    generator_ckpt,
    audio2pose_ckpt,
    save_dir="samples/results",
):

    ### preprocess part (전처리 부분)

    # cur_path = os.getcwd() # TODO will be deleted
    with open(CONFIG_DIR.joinpath("audio2kp.yaml").absolute()) as f:  # audio2kp open
        tmp = yaml.full_load(f)
        
    opt = argparse.Namespace(**tmp)
    img = read_img(img_path).cuda()

    temp_audio = audio_read(audio_path=audio_path)

    # h_r, a_{1:T}
    first_pose = get_img_pose(img_path)  # h_r # [Rx, Ry, Rz, Tx, Ty, Tz]
    audio_feature = get_audio_feature_from_audio(temp_audio) # a_{1:T} # shape [time, power]

    frames = len(audio_feature) // 4
    frames = min(frames, len(phs["phone_list"]))

    tp = np.zeros([256, 256], dtype=np.float32)
    draw_annotation_box(tp, first_pose[:3], first_pose[3:])
    tp = torch.from_numpy(tp).unsqueeze(0).unsqueeze(0).cuda() # [1, 1, 256, 256]
    ref_pose = get_pose_from_audio(tp, audio_feature, audio2pose_ckpt) # h_{1:T}
    torch.cuda.empty_cache()
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

    # load and eval modes
    kp_detector = KPDetector(
        **config["model_params"]["kp_detector_params"],
        **config["model_params"]["common_params"]
    ) # keypoint detector
    
    generator = OcclusionAwareGenerator(
        **config["model_params"]["generator_params"],
        **config["model_params"]["common_params"]
    ) # rendorer
    
    kp_detector = kp_detector.cuda()
    generator = generator.cuda()

    ph2kp = Audio2kpTransformer(opt).cuda() # avct

    load_ckpt(generator_ckpt, kp_detector=kp_detector, generator=generator, ph2kp=ph2kp)

    ph2kp.eval()
    generator.eval()
    kp_detector.eval()

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
        default='G:\project\\talking_face_generation\AAAI22_ONE-SHOT_TALKING_FACE_GEN\src\samples\imgs\d1.jpg',
        help="path of the input image ( .jpg ), preprocessed by image_preprocess.py",
    )
    argparser.add_argument(
        "--audio_path", type=str, default='G:\project\\talking_face_generation\AAAI22_ONE-SHOT_TALKING_FACE_GEN\src\samples\\audios\\abstract.wav', help="path of the input audio ( .wav )"
    )
    argparser.add_argument(
        "--phoneme_path",
        type=str,
        default='G:\project\\talking_face_generation\AAAI22_ONE-SHOT_TALKING_FACE_GEN\src\samples\phonemes\\abstract.json',
        help="path of the input phoneme. It should be note that the phoneme must be consistent with the input audio",
    )
    argparser.add_argument(
        "--save_dir",
        type=str,
        default="samples/results",
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

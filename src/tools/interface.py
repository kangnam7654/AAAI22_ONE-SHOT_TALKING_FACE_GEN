import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[1]
sys.path.append(str(ROOT_DIR))
CONFIG_DIR = os.path.join(ROOT_DIR, 'config_file')
SAMPLE_DIR = os.path.join(ROOT_DIR, 'samples')

from skimage import io, img_as_float32
import cv2
import torch
import numpy as np
import subprocess
import pandas as pd
from models.audio2pose import audio2poseLSTM
from models.util import draw_annotation_box
from scipy.io import wavfile
import python_speech_features
import imageio
import pyworld
import json
from scipy.interpolate import interp1d
import yaml

# import config
with open(os.path.join(CONFIG_DIR, 'vox-256.yaml')) as f:
    config = yaml.full_load(f)

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


def get_tp(img_path):
    # h_r, a_{1:T}
    first_pose = get_img_pose(img_path)  # h_r # [Rx, Ry, Rz, Tx, Ty, Tz]
    tp = np.zeros([256, 256], dtype=np.float32)
    draw_annotation_box(tp, first_pose[:3], first_pose[3:])
    tp = torch.from_numpy(tp).unsqueeze(0).unsqueeze(0).cuda() # [1, 1, 256, 256]
    return tp


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


### audio
def read_audio(audio_path):
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


def normalize_kp(
    kp_source,
    kp_driving,
    kp_driving_initial,
    use_relative_movement=True,
    use_relative_jacobian=True,
):

    kp_new = {k: v for k, v in kp_driving.items()}
    if use_relative_movement:
        kp_value_diff = kp_driving["value"] - kp_driving_initial["value"]
        # kp_value_diff *= adapt_movement_scale
        kp_new["value"] = kp_value_diff + kp_source["value"]

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(
                kp_driving["jacobian"], torch.inverse(kp_driving_initial["jacobian"])
            )
            kp_new["jacobian"] = torch.matmul(jacobian_diff, kp_source["jacobian"])

    return kp_new


def inter_pitch(y, y_flag):
    frame_num = y.shape[0]
    i = 0
    last = -1
    while i < frame_num:
        if y_flag[i] == 0:
            while True:
                if y_flag[i] == 0:
                    if i == frame_num - 1:
                        if last != -1:
                            y[last + 1 :] = y[last]
                        i += 1
                        break
                    i += 1
                else:
                    break
            if i >= frame_num:
                break
            elif last == -1:
                y[:i] = y[i]
            else:
                inter_num = i - last + 1
                fy = np.array([y[last], y[i]])
                fx = np.linspace(0, 1, num=2)
                f = interp1d(fx, fy)
                fx_new = np.linspace(0, 1, inter_num)
                fy_new = f(fx_new)
                y[last + 1 : i] = fy_new[1:-1]
                last = i
                i += 1

        else:
            last = i
            i += 1
    return y


def load_ckpt(checkpoint_path, generator=None, kp_detector=None, ph2kp=None):
    checkpoint = torch.load(checkpoint_path)
    if ph2kp is not None:
        ph2kp.load_state_dict(checkpoint["ph2kp"])
    if generator is not None:
        generator.load_state_dict(checkpoint["generator"])
    if kp_detector is not None:
        kp_detector.load_state_dict(checkpoint["kp_detector"])


def get_img_pose(img_path):
    processor = config['OPENFACE_POSE_EXTRACTOR_PATH']

    tmp_dir = os.path.join(SAMPLE_DIR, "tmp_dir")
    os.makedirs((tmp_dir), exist_ok=True)
    subprocess.call([processor, "-f", img_path, "-out_dir", tmp_dir, "-pose"])

    img_file = os.path.basename(img_path)[:-4] + ".csv"
    csv_file = os.path.join(tmp_dir, img_file)
    pos_data = pd.read_csv(csv_file)
    i = 0
    # pose = [pos_data[" pose_Rx"][i], pos_data[" pose_Ry"][i], pos_data[" pose_Rz"][i],pos_data[" pose_Tx"][i], pos_data[" pose_Ty"][i], pos_data[" pose_Tz"][i]]
    pose = [
        pos_data.loc[i, " pose_Rx"],
        pos_data.loc[i, " pose_Ry"],
        pos_data.loc[i, " pose_Rz"],
        pos_data.loc[i, " pose_Tx"],
        pos_data.loc[i, " pose_Ty"],
        pos_data.loc[i, " pose_Tz"],
    ]
    # pose = [pose]
    pose = np.array(pose, dtype=np.float32)
    return pose


def read_img(path):
    img = io.imread(path)[:, :, :3]
    img = cv2.resize(img, (256, 256))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.array(img_as_float32(img))
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0)
    return img


def parse_phoneme_file(phoneme_path, use_index=True):
    with open(phoneme_path, "r") as f:
        result_text = json.load(f)
    frame_num = int(result_text[-1]["phones"][-1]["ed"] / 100 * 25)
    phoneset_list = []
    index = 0

    word_len = len(result_text)
    word_index = 0
    phone_index = 0
    cur_phone_list = result_text[0]["phones"]
    phone_len = len(cur_phone_list)
    cur_end = cur_phone_list[0]["ed"]

    phone_list = []

    phoneset_list.append(cur_phone_list[0]["ph"])
    i = 0
    while i < frame_num:
        if i * 4 < cur_end:
            phone_list.append(cur_phone_list[phone_index]["ph"])
            i += 1
        else:
            phone_index += 1
            if phone_index >= phone_len:
                word_index += 1
                if word_index >= word_len:
                    phone_list.append(cur_phone_list[-1]["ph"])
                    i += 1
                else:
                    phone_index = 0
                    cur_phone_list = result_text[word_index]["phones"]
                    phone_len = len(cur_phone_list)
                    cur_end = cur_phone_list[phone_index]["ed"]
                    phoneset_list.append(cur_phone_list[phone_index]["ph"])
                    index += 1
            else:
                # print(word_index,phone_index)
                cur_end = cur_phone_list[phone_index]["ed"]
                phoneset_list.append(cur_phone_list[phone_index]["ph"])
                index += 1

    with open("G:\project\\talking_face_generation\AAAI22_ONE-SHOT_TALKING_FACE_GEN\src\phindex.json") as f:
        ph2index = json.load(f)
    if use_index:
        phone_list = [ph2index[p] for p in phone_list]
    saves = {"phone_list": phone_list}

    return saves


def get_audio_feature_from_audio(audio_path):
    sample_rate, audio = wavfile.read(audio_path)
    if len(audio.shape) == 2:
        if np.min(audio[:, 0]) <= 0:
            audio = audio[:, 1]
        else:
            audio = audio[:, 0]

    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    a = python_speech_features.mfcc(audio, sample_rate)
    b = python_speech_features.logfbank(audio, sample_rate)
    c, _ = pyworld.harvest(audio, sample_rate, frame_period=10)
    c_flag = (c == 0.0) ^ 1
    c = inter_pitch(c, c_flag)
    c = np.expand_dims(c, axis=1)
    c_flag = np.expand_dims(c_flag, axis=1)
    frame_num = np.min([a.shape[0], b.shape[0], c.shape[0]])

    cat = np.concatenate(
        [a[:frame_num], b[:frame_num], c[:frame_num], c_flag[:frame_num]], axis=1
    )
    return cat 


def get_pose_from_audio(img, audio, audio2pose):

    num_frame = len(audio) // 4

    minv = np.array([-0.6, -0.6, -0.6, -128.0, -128.0, 128.0], dtype=np.float32)
    maxv = np.array([0.6, 0.6, 0.6, 128.0, 128.0, 384.0], dtype=np.float32)
    generator = audio2poseLSTM().cuda().eval()

    ckpt_para = torch.load(audio2pose)

    generator.load_state_dict(ckpt_para["generator"])
    generator.eval()

    audio_seq = []
    for i in range(num_frame):
        audio_seq.append(audio[i * 4 : i * 4 + 4])

    audio = torch.from_numpy(np.array(audio_seq, dtype=np.float32)).unsqueeze(0).cuda()

    x = {}
    x["img"] = img
    x["audio"] = audio
    poses = generator(x)

    poses = poses.cpu().data.numpy()[0]
    poses = (poses + 1) / 2 * (maxv - minv) + minv

    return poses

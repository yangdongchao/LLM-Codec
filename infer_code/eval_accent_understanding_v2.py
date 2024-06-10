import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
import albumentations
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import yaml
import torch
from omegaconf import OmegaConf
import clip
from models3.MSCodec import MSCodecLM
import torch
import random
import typing as tp
from collections import OrderedDict
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

##############
from llama_inference.llama import Tokenizer, ModelArgs
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.misc as misc

from sklearn.utils import shuffle
import torchaudio

def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav



class EmotionDataset(Dataset):
    def __init__(self, data_root, tsv_path, time, model_path, audio_tokenizer, text_token_embedding, device="cpu", induction=1, vq1_texts=None):

        self.device = device
        self.text_token_embedding = text_token_embedding
        self.data_root = data_root
        self.text_tokenizer = Tokenizer(model_path=model_path + "/tokenizer.model")
        self.induction = induction
        self.vq1_texts = list(vq1_texts)
        #print('self.vq1_texts ', self.vq1_texts)
        self.audio_tokenizer = audio_tokenizer

        self.image_ids = []
        self.class_names = []
        with open(tsv_path) as f:
            for line in f.readlines():
                #image_ids = line.strip('\n').split(",")[:-1]
                image_ids = line.strip('\n').split(",")[:-1]
                cur_images = []
                cur_classes = []
                for image_id in image_ids:
                    class_ind, _ = image_id.split("/")
                    cur_images.append(image_id.split('/')[-1])
                    #cur_images.append(image_id)
                    cur_classes.append(class_ind)
                self.image_ids.append(cur_images)
                self.class_names.append(cur_classes)
        self.image_ids = self.image_ids[time:]
        self.class_names = self.class_names[time:]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        select_texts = []
        select_audios = []
        query_classes = np.unique(self.class_names[index]) # 去重
        ###Instruction
        if self.induction == 0:
            instruction = ''
        else:
            instruction = "For each of the following input output pairs, output is one of ["
            for i in range(0, len(query_classes)):
                instruction = instruction + "'%s' or "%(query_classes[i])
            instruction = instruction[:-4] + ']\n'
        # print('instruction ', instruction)
        # assert 1==2
        prompt_tokens = torch.tensor(self.text_tokenizer.encode(instruction, bos=True, eos=False)).unsqueeze(0).to(self.device)
        in_tokens = torch.tensor(self.text_tokenizer.encode("###\nInput: < ", bos=False, eos=False), dtype=torch.int64).unsqueeze(0).to(self.device)
        out_tokens = torch.tensor(self.text_tokenizer.encode(" >\nOutput: ", bos=False, eos=False), dtype=torch.int64).unsqueeze(0).to(self.device)
        #print('prompt_tokens ', prompt_tokens, prompt_tokens.shape)
        prompt_features = self.text_token_embedding(prompt_tokens)
        # print('prompt_features ', prompt_features.shape)
        # assert 1==2
        in_feature = self.text_token_embedding(in_tokens)
        out_feature = self.text_token_embedding(out_tokens)

        ###Load images and class texts
        image_ids = self.image_ids[index]
        class_names = self.class_names[index]
        last_setence = ''
        for i in range(0, len(image_ids)):
            image_id = image_ids[i]
            class_name = class_names[i]
            wav_root = os.path.join(self.data_root, image_id)
            wav, sr = torchaudio.load(wav_root)
            if sr != 16000:
                wav = convert_audio(wav, sr, 16000, 1)
            wav = wav.unsqueeze(1).to(self.device)
            if wav.shape[2]/16000 > 1:
                wav = wav[:,:,:1*16000]
            else:
                wav_new = torch.zeros(1, 1, 1*16000).type_as(wav)
                wav_new[:,:,:wav.shape[2]] = wav
                wav = wav_new
            my_code = []
            setence = ''
            with torch.no_grad():
                x, codes , _, _,_,_ = self.audio_tokenizer(wav)
                for kk, code in enumerate(codes):
                    if kk != 0:
                        continue
                    for j in range(code.shape[1]):
                        if kk==0:
                            tmp = code[0,j].item() # index
                            wo = self.vq1_texts[tmp] # get word
                            #print('word ', wo)
                            real_code = self.text_tokenizer.encode(str(wo), bos=False, eos=False)
                            my_code += real_code
                            setence += ' ' + str(wo)
                        else:
                            tmp = code[0,j].item()
                            wo = self.text_tokenizer.decode(tmp)
                            setence += ' ' + str(wo)
                            my_code.append(tmp)
                    # ed = self.text_tokenizer.encode(f'<layer_{kk+1}_end>', bos=False, eos=False)
                    # my_code += ed
                    # setence += f' <layer_{kk+1}_end>'
            if i == len(image_ids)-1:
                last_setence = setence
                
            #assert 1==2
            my_code = np.array(my_code)
            my_code = torch.from_numpy(my_code).to(self.device)
            select_audios.append(my_code)
            select_texts.append(class_name)
            if i == len(image_ids)-1:
                x = x.squeeze(0).detach().cpu()
                torchaudio.save('tmp.wav', x, sample_rate=16000, encoding='PCM_S', bits_per_sample=16)
            
        ##The last image serves query image (GT)
        target_texts = select_texts[-1] # the last one as the target
        select_texts = select_texts[:-1] # previous

        ##Generating context examples with other images
        for i in range(0, len(select_texts)):
            text_token = torch.tensor(self.text_tokenizer.encode(select_texts[i], bos=False, eos=False), dtype=torch.int64).to(self.device).unsqueeze(0)
            text_feature = self.text_token_embedding(text_token)
            vis_texts = ""
            vis_token = select_audios[i].unsqueeze(0)
            vis_feature = self.text_token_embedding(vis_token)
            prompt_tokens = torch.cat([prompt_tokens, in_tokens, vis_token, out_tokens, text_token], dim=-1)
            prompt_features = torch.cat( [prompt_features, in_feature, vis_feature , out_feature, text_feature], dim=1)

        ##Adding query token
        vis_texts = ""
        vis_token = select_audios[-1].unsqueeze(0)
        prompt_tokens = torch.cat([prompt_tokens, in_tokens, vis_token, out_tokens], dim=-1)
        prompt_features = torch.cat( [prompt_features, in_feature, self.text_token_embedding(vis_token), out_feature], dim=1)
        # print('prompt_tokens f', prompt_tokens.shape)
        last_image = image_ids[-1]
        
        out_mask = torch.zeros(16, 32000)
        max_len = 0
        for i in range(0, len(query_classes)):
            class_token = self.text_tokenizer.encode(str(query_classes[i]), bos=False, eos=True)
            if len(class_token) > max_len:
                max_len = len(class_token)
            for j, token in enumerate(class_token):
                out_mask[j, token] = 1
        prompt_tokens = prompt_tokens[0].to("cpu")
        prompt_features = prompt_features[0].to("cpu")
        out_mask = out_mask.to("cpu")
        return [prompt_tokens, prompt_features, target_texts, out_mask, max_len, image_ids, last_setence]


def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument("--llama_model_path", default="./llama", type=str, help="path of llama model")
    parser.add_argument("--max_seq_len", type=int, default=2048, metavar="LENGTH", help="the maximum sequence length")

    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    parser.add_argument("--audio_path", default="/data/all", type=str, help="path of llama model")
    parser.add_argument("--file_path", default="/data/all", type=str, help="path of tsv or scp")
    parser.add_argument("--vqgan_path", default="vqgan_weight/vqgan_imagenet_f16_16384", type=str, help="path of llama model")
    
    parser.add_argument("--n_vision_words", default=32000, type=int)
    parser.add_argument("--output_type", default="next_token_prediction", type=str, help="next_token_prediction/classification")
    parser.add_argument("--decode_rate", type=float, default=0, help="Decoding Loss")
    parser.add_argument("--vq_config_path", type=str, default="vqgan_configs/model_16384.yaml", help="Decoding Loss")
    parser.add_argument("--codec_ckpt", type=str, default="stage_1_llama_fix-40.pth", help="the checkpoint of audio codec models")

    parser.add_argument("--induction", type=int, default=1, help="Decoding Loss")
    return parser


def main(args):

    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    vq1_texts = np.load("../layer1.npy", allow_pickle=True)

    ###Load MSLM_codec
    exp_model_config = OmegaConf.load(args.vq_config_path)
    model = MSCodecLM(**exp_model_config.generator.config)  
    parameter_dict = torch.load(args.codec_ckpt)
    model.load_state_dict(parameter_dict['codec_model']) # load model
    model.to(device)
    model.eval()

    ###Load LLaMA
    llama_model_path = args.llama_model_path
    from llama_inference.llama import Llama
    generator = Llama.build(
        ckpt_dir=llama_model_path,
        tokenizer_path=llama_model_path + "/tokenizer.model",
        max_seq_len=args.max_seq_len,
        max_batch_size=2,
    )
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = ""
    print_freq = 10
    text_tokenizer = Tokenizer(model_path=args.llama_model_path + "/tokenizer.model")
    time = 0
    acc = 0
    r_acc = 0
    
    ###Load Dataset
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = EmotionDataset(
        data_root=args.audio_path, tsv_path=args.file_path,  time=0, \
             model_path=args.llama_model_path, audio_tokenizer=model, \
            text_token_embedding=generator.model.tok_embeddings, device=device, induction=args.induction,
            vq1_texts=vq1_texts
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        #sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    ans = {}
    for data_iter_step, [prompt_tokens, prompt_features, target_texts, out_mask, max_lens, image_ids, last_setence] in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):      
        print(image_ids[0], target_texts, last_setence)
        prompt_tokens = prompt_tokens.to(device)
        prompt_features = prompt_features.to(device)
        out_mask = out_mask.to(device)
        ##Auto regression with LLaMA for few-shot classification
        predictions = generator.generate_fewshot(
            prompt_tokens,
            prompt_features,
            induction=args.induction, #No induction settings do not use out_mask (1)
            out_mask = out_mask,
            max_gen_len=16,
            temperature=0,
            top_p=1.0,
        ) # 

        for target_text, prediction in zip(target_texts, predictions):
            pred = prediction['tokens']
            pred_text = prediction['generation']
            time = time + 1
            if pred_text[:len(target_text)] == target_text:
                acc = acc + 1
            print("Prediction: %s \n Ground Truth: %s \n Acc: %.4f \n"%(pred_text[:len(target_text)], target_text, acc / time))
        
    print("Accuracy: ", acc / time)


if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

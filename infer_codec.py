from omegaconf import OmegaConf
from codec.MSCodec import MSCodecLM
from llama_inference.llama import Tokenizer, ModelArgs # 

llama_model_path = '' # download llama 2 7B from https://github.com/meta-llama/llama-recipes/tree/main
text_tokenizer = Tokenizer(model_path=llama_model_path + "/tokenizer.model")
# load model
vq_config_path = 'config.yaml'
codec_ckpt = 'ckpt.pth' # set the ckpt path
device = 'cuda'
exp_model_config = OmegaConf.load()
model = MSCodecLM(**exp_model_config.generator.config)  
parameter_dict = torch.load(codec_ckpt)
model.load_state_dict(parameter_dict['codec_model']) # load model
model.to(device)
model.eval()
vq1_texts = np.load("layer1.npy", allow_pickle=True)
wav_root = ''
wav, sr = torchaudio.load(wav_root)
if sr != 16000:
    wav = convert_audio(wav, sr, 16000, 1)
wav = wav.unsqueeze(1).to(device)
my_code = []
setence = ''
# encode
with torch.no_grad():
    x, codes , _, _,_,_ = model(wav)
    for kk, code in enumerate(codes):
        for j in range(code.shape[1]):
            if kk==0:
                tmp = code[0,j].item() # index
                wo = vq1_texts[tmp] # get word
                real_code = text_tokenizer.encode(str(wo), bos=False, eos=False)
                my_code += real_code
                setence += ' ' + str(wo)
            else:
                tmp = code[0,j].item()
                wo = self.text_tokenizer.decode(tmp)
                setence += ' ' + str(wo)
                my_code.append(tmp)
# decode to wav
x = model.decode(codes)


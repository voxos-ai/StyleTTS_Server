from styletts2.tts import StyleTTS2,word_tokenize, TextCleaner, length_to_mask
import torch
from .logger import configure_logger
import time

logger = configure_logger(__name__)
class TTS(StyleTTS2):
    def __init__(self, model_checkpoint_path=None, config_path=None, phoneme_converter='gruut'):
        super().__init__(model_checkpoint_path, config_path, phoneme_converter)
        self.style_vectors = {}
    def register_voice(self,voice_path:str):
        audio_name = (voice_path.split("/")[-1]).split(".")[0]
        __s = time.time()
        self.style_vectors[audio_name] = self.compute_style(voice_path)
        logger.info(f"add new voice {audio_name} in {time.time() - __s} sec")
    
    def text_to_audio(self,
                  text: str,
                  voice_id:str,
                  alpha=0.3,
                  beta=0.7,
                  diffusion_steps=5,
                  embedding_scale=1):
        
        style_vector = self.style_vectors[voice_id]
        # TEXT filteration section
        text = text.strip()
        text = text.replace('"', '')
        phonemized_text = self.phoneme_converter.phonemize(text)
        ps = word_tokenize(phonemized_text)
        phoneme_string = ' '.join(ps)

        textcleaner = TextCleaner()
        tokens = textcleaner(phoneme_string)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(self.device),
                                  embedding=bert_dur,
                                  embedding_scale=embedding_scale,
                                  features=style_vector, # reference from the same speaker as the embedding
                                  num_steps=diffusion_steps).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * style_vector[:, :128]
            s = beta * s + (1 - beta)  * style_vector[:, 128:]

            # duration prediction
            d = self.model.predictor.text_encoder(d_en,
                                                  s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr,
                                     F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        output = out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later
        
        return output